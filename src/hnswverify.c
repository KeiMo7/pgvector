#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/relation.h"
#include "catalog/pg_am.h"
#include "fmgr.h"
#include "funcapi.h"
#include "hnsw.h"
#include "lib/simplehash.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/tuplestore.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/*
 * Read-only verification and statistics for on-disk HNSW indexes.
 *
 * Everything in this file operates under AccessShareLock plus shared page
 * locks only: no buffers are dirtied, no WAL is generated and no catalogs are
 * modified.
 */

/* Retry budget when an element tuple changes between reads */
#define HNSW_VERIFY_MAX_RETRIES		3

/* Maximum number of page-chain walks when pages are appended concurrently */
#define HNSW_VERIFY_MAX_CHAIN_PASSES 3

/* Severity levels */
#define HNSW_SEVERITY_ERROR			"error"
#define HNSW_SEVERITY_WARNING		"warning"

/* ----------------------------------------------------------------
 * Hash table: element tuple TID -> element state
 * ----------------------------------------------------------------
 */

typedef struct ElementHashEntry
{
	ItemPointerData tid;		/* hash key: element tuple location */
	char		status;
	int			level;
	bool		deleted;
	int			heaptidsLength;
	BlockNumber neighborPage;
	OffsetNumber neighborOffno;
	uint8		version;
	ItemPointerData firstHeapTid;
}			ElementHashEntry;

static inline uint64
hnsw_verify_murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}

static uint32
hnsw_verify_hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	x.i = 0;
	x.tid = tid;

	return hnsw_verify_murmurhash64(x.i);
}

#define SH_PREFIX elementhash
#define SH_ELEMENT_TYPE ElementHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_KEY tid
#define SH_HASH_KEY(tb, key) hnsw_verify_hash_tid(key)
#define SH_EQUAL(tb, a, b) ItemPointerEquals(&a, &b)
#define SH_SCOPE static inline
#define SH_DEFINE
#define SH_DECLARE
#include "lib/simplehash.h"

/* ----------------------------------------------------------------
 * Findings and per-level statistics
 * ----------------------------------------------------------------
 */

typedef struct HnswFinding
{
	char	   *severity;
	char	   *code;
	int			level;			/* -1 when unknown / not applicable */
	BlockNumber blkno;			/* InvalidBlockNumber when NULL */
	OffsetNumber offno;			/* InvalidOffsetNumber when NULL */
	char	   *detail;
}			HnswFinding;

typedef struct HnswLevelStats
{
	int64		elements;
	int64		deleted;
	int64		heaptids;
	int			fanoutMin;
	int			fanoutMax;
	bool		fanoutSet;
	int64		emptyNeighbors;
}			HnswLevelStats;

/* Snapshot of an element tuple copied out of shared buffers */
typedef struct HnswElementSnapshot
{
	ItemPointerData tid;
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	int			heaptidsLength;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	ItemPointerData neighbortid;
	ItemPointerData firstHeapTid;
}			HnswElementSnapshot;

/* Snapshot of a neighbor tuple copied out of shared buffers */
typedef struct HnswNeighborSnapshot
{
	uint8		type;
	uint8		version;
	uint16	count;
	Size		slots;			/* slots that fit inside the tuple */
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			HnswNeighborSnapshot;

typedef struct HnswVerifyState
{
	Relation	index;
	int			m;
	int			dimensions;
	int			efConstruction;
	int			maxLevel;		/* max element level observed */
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int			entryLevel;
	BlockNumber nblocks;
	BlockNumber lastSeenBlkno;	/* blocks up to here are already collected */
	bool		chainStarted;	/* false until the first pass begins */
	BlockNumber invalidPage;	/* first page with a bad page id */

	bool		full;
	int			maxFindings;
	int			maxVisited;

	MemoryContext persistentCtx;	/* survives across pages */
	MemoryContext tmpCtx;		/* reset after every page */

	struct elementhash_hash *elements;
	struct tidhash_hash *heapTids;	/* heap TID dedup */
	HnswLevelStats *levelStats;
	HnswFinding *findings;
	int			findingsLength;
	int			findingsCapacity;
}			HnswVerifyState;

/* Forward declarations */
static void HnswVerifyOpenIndex(Oid indexOid, Relation *index, LOCKMODE lockmode);
static void HnswReadMetaPage(HnswVerifyState * state);
static void HnswAddFinding(HnswVerifyState * state, const char *severity,
						   const char *code, int level, BlockNumber blkno,
						   OffsetNumber offno, const char *detail);
static bool HnswTupleHeaderOk(Page page, OffsetNumber offno, Size minSize);
static void HnswScanPageChain(HnswVerifyState * state);
static void HnswVerifyNeighbors(HnswVerifyState * state);
static void HnswBuildLevelCounters(HnswVerifyState * state);
static void HnswCheckEntryPoint(HnswVerifyState * state);
static void HnswBfs(HnswVerifyState * state);

/* ----------------------------------------------------------------
 * Small helpers
 * ----------------------------------------------------------------
 */

/*
 * Open a relation and make sure it is an HNSW index.
 */
static void
HnswVerifyOpenIndex(Oid indexOid, Relation *index, LOCKMODE lockmode)
{
	Relation	rel;

	rel = index_open(indexOid, lockmode);

	if (!rel->rd_index)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(rel))));

	{
		Oid			hnswAmOid = GetSysCacheOid1(AMNAME, Anum_pg_am_oid,
											   CStringGetDatum("hnsw"));

		if (!OidIsValid(hnswAmOid) || rel->rd_rel->relam != hnswAmOid)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not an hnsw index",
							RelationGetRelationName(rel))));
	}

	*index = rel;
}

/*
 * Record a finding. Strings are copied into the persistent memory context.
 */
static void
HnswAddFinding(HnswVerifyState *state, const char *severity, const char *code,
			   int level, BlockNumber blkno, OffsetNumber offno,
			   const char *detail)
{
	MemoryContext oldCtx;
	HnswFinding *finding;

	if (state->findingsLength >= state->findingsCapacity)
	{
		int			newCapacity = state->findingsCapacity * 2;

		if (newCapacity <= 0)
			newCapacity = 32;

		oldCtx = MemoryContextSwitchTo(state->persistentCtx);
		if (state->findings == NULL)
			state->findings = palloc(sizeof(HnswFinding) * newCapacity);
		else
			state->findings = repalloc(state->findings,
									   sizeof(HnswFinding) * newCapacity);
		MemoryContextSwitchTo(oldCtx);
		state->findingsCapacity = newCapacity;
	}

	finding = &state->findings[state->findingsLength++];

	oldCtx = MemoryContextSwitchTo(state->persistentCtx);
	finding->severity = pstrdup(severity);
	finding->code = pstrdup(code);
	finding->detail = (detail != NULL) ? pstrdup(detail) : NULL;
	MemoryContextSwitchTo(oldCtx);

	finding->level = level;
	finding->blkno = blkno;
	finding->offno = offno;
}

/*
 * Validate the item id and tuple envelope of a slot. Never throws: any
 * malformed slot is reported as tuple_corrupt and skipped by the caller.
 */
static bool
HnswTupleHeaderOk(Page page, OffsetNumber offno, Size minSize)
{
	ItemId		itemid;

	if (offno < FirstOffsetNumber || offno > PageGetMaxOffsetNumber(page))
		return false;

	itemid = PageGetItemId(page, offno);

	if (!ItemIdIsUsed(itemid) || ItemIdGetLength(itemid) == 0)
		return false;

	/* Tuples are stored between pd_upper and pd_special */
	if (ItemIdGetOffset(itemid) < ((PageHeader) page)->pd_upper ||
		ItemIdGetOffset(itemid) + ItemIdGetLength(itemid) >
		((PageHeader) page)->pd_special)
		return false;

	if (ItemIdGetLength(itemid) < minSize)
		return false;

	return true;
}

/*
 * Read and lock a single page under a shared buffer lock. The caller is
 * responsible for UnlockReleaseBuffer().
 */
static Buffer
HnswReadSharedPage(Relation index, BlockNumber blkno)
{
	Buffer		buf = ReadBuffer(index, blkno);

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	return buf;
}

/*
 * Copy an element tuple into caller-supplied local storage while a shared
 * buffer lock is held.
 */
static void
HnswCopyElementTuple(BlockNumber blkno, Page page, OffsetNumber offno,
					 HnswElementSnapshot *snap)
{
	HnswElementTuple etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));
	int			i;

	/* Zero-init so unused heap TID slots read as invalid */
	memset(snap, 0, sizeof(HnswElementSnapshot));

	ItemPointerSet(&snap->tid, blkno, offno);
	snap->type = etup->type;
	snap->level = etup->level;
	snap->deleted = etup->deleted;
	snap->version = etup->version;
	ItemPointerCopy(&etup->neighbortid, &snap->neighbortid);
	snap->heaptidsLength = 0;
	ItemPointerSetInvalid(&snap->firstHeapTid);

	for (i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (!ItemPointerIsValid(&etup->heaptids[i]))
		{
			ItemPointerSetInvalid(&snap->heaptids[i]);
			continue;
		}

		ItemPointerCopy(&etup->heaptids[i], &snap->heaptids[i]);
		snap->heaptidsLength++;
	}

	if (snap->heaptidsLength > 0)
		ItemPointerCopy(&snap->heaptids[0], &snap->firstHeapTid);
}

/*
 * Read a neighbor tuple into a palloc'd snapshot. Returns NULL when the
 * location or tuple envelope is invalid. The returned object lives in the
 * current memory context.
 */
static HnswNeighborSnapshot *
HnswReadNeighborSnapshot(HnswVerifyState *state, BlockNumber blkno,
						 OffsetNumber offno, bool *corrupt)
{
	Buffer		buf;
	Page		page;
	ItemId		itemid;
	Size		tupleSize;
	Size		slots;
	HnswNeighborTuple ntup;
	HnswNeighborSnapshot *snap;

	*corrupt = false;

	if (!BlockNumberIsValid(blkno) || blkno >= state->nblocks ||
		!OffsetNumberIsValid(offno))
		return NULL;

	buf = HnswReadSharedPage(state->index, blkno);
	page = BufferGetPage(buf);

	if (HnswPageGetOpaque(page)->page_id != HNSW_PAGE_ID ||
		!HnswTupleHeaderOk(page, offno,
						   offsetof(HnswNeighborTupleData, indextids)))
	{
		*corrupt = true;
		UnlockReleaseBuffer(buf);
		return NULL;
	}

	itemid = PageGetItemId(page, offno);
	tupleSize = ItemIdGetLength(itemid);
	ntup = (HnswNeighborTuple) PageGetItem(page, itemid);

	/* Slots that physically fit inside the tuple */
	if (tupleSize < offsetof(HnswNeighborTupleData, indextids))
		slots = 0;
	else
		slots = (tupleSize - offsetof(HnswNeighborTupleData, indextids)) /
			sizeof(ItemPointerData);

	snap = palloc(offsetof(HnswNeighborSnapshot, indextids) +
				  sizeof(ItemPointerData) * slots);
	snap->type = ntup->type;
	snap->version = ntup->version;
	snap->count = ntup->count;
	snap->slots = slots;

	memcpy(snap->indextids, ntup->indextids,
		   slots * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return snap;
}

/* ----------------------------------------------------------------
 * Metapage
 * ----------------------------------------------------------------
 */

/*
 * Validate the metapage and load the values all later phases rely on. A bad
 * magic number or version is fatal: later pages cannot be interpreted.
 */
static void
HnswReadMetaPage(HnswVerifyState *state)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = HnswReadSharedPage(state->index, HNSW_METAPAGE_BLKNO);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (metap->magicNumber != HNSW_MAGIC_NUMBER ||
		metap->version != HNSW_VERSION)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hnsw index \"%s\" has invalid magic number or version",
						RelationGetRelationName(state->index))));
	}

	state->m = metap->m;
	state->dimensions = metap->dimensions;
	state->efConstruction = metap->efConstruction;
	state->entryBlkno = metap->entryBlkno;
	state->entryOffno = metap->entryOffno;
	state->entryLevel = metap->entryLevel;

	UnlockReleaseBuffer(buf);

	state->nblocks = RelationGetNumberOfBlocks(state->index);
}

/* ----------------------------------------------------------------
 * Pass 1: page chain scan and element hash
 * ----------------------------------------------------------------
 */

/*
 * Check the heap TID slots of one element: valid TIDs must form a gapless
 * prefix, and each heap TID must be unique across the whole index.
 */
static void
HnswCheckElementHeapTids(HnswVerifyState *state, HnswElementSnapshot *snap)
{
	bool		sawInvalid = false;
	int			i;

	for (i = 0; i < HNSW_HEAPTIDS; i++)
	{
		ItemPointer heaptid = &snap->heaptids[i];

		if (!ItemPointerIsValid(heaptid))
		{
			sawInvalid = true;
			continue;
		}

		if (sawInvalid)
		{
			HnswAddFinding(state, HNSW_SEVERITY_ERROR, "heaptid_gap",
						   snap->level,
						   ItemPointerGetBlockNumber(&snap->tid),
						   ItemPointerGetOffsetNumber(&snap->tid),
						   "valid heap TID follows an invalid slot");
		}
		else if (tidhash_lookup(state->heapTids, *heaptid) != NULL)
		{
			HnswAddFinding(state, HNSW_SEVERITY_ERROR, "heaptid_duplicate",
						   snap->level,
						   ItemPointerGetBlockNumber(&snap->tid),
						   ItemPointerGetOffsetNumber(&snap->tid),
						   "heap TID appears in multiple elements");
		}
		else
		{
			bool		found;

			tidhash_insert(state->heapTids, *heaptid, &found);
		}
	}
}

/*
 * Process one element tuple during the page chain scan.
 */
static void
HnswCollectElement(HnswVerifyState *state, BlockNumber blkno, Page page,
				   OffsetNumber offno)
{
	HnswElementTuple etup;
	HnswElementSnapshot snap;
	ElementHashEntry *entry;
	bool		found;
	ItemPointerData elementTid;
	Size		minSize = offsetof(HnswElementTupleData, data);

	if (!HnswTupleHeaderOk(page, offno, minSize))
	{
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "tuple_corrupt",
					   -1, blkno, offno,
					   "element tuple envelope is invalid");
		return;
	}

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

	if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
	{
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "tuple_corrupt",
					   -1, blkno, offno,
					   "expected element tuple but found another type");
		return;
	}

	/* Skip tuples already collected on an earlier chain pass */
	ItemPointerSet(&elementTid, blkno, offno);
	if (elementhash_lookup(state->elements, elementTid) != NULL)
		return;

	HnswCopyElementTuple(blkno, page, offno, &snap);

	/* Level bounds come from the index itself, not reloptions */
	if (snap.level > HnswGetMaxLevel(state->m))
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "element_level_invalid",
					   snap.level, blkno, offno,
					   "element level exceeds maximum level");

	HnswCheckElementHeapTids(state, &snap);

	entry = elementhash_insert(state->elements, snap.tid, &found);
	entry->level = snap.level;
	entry->deleted = snap.deleted;
	entry->heaptidsLength = snap.heaptidsLength;
	entry->neighborPage = ItemPointerGetBlockNumber(&snap.neighbortid);
	entry->neighborOffno = ItemPointerGetOffsetNumber(&snap.neighbortid);
	entry->version = snap.version;
	entry->firstHeapTid = snap.firstHeapTid;

	if (snap.level > state->maxLevel &&
		snap.level <= HnswGetMaxLevel(state->m))
		state->maxLevel = snap.level;
}

/*
 * Walk the page chain once. When chainStarted is true, blocks at or before
 * startAfter are skipped since earlier passes already collected them.
 * Returns true if the relation gained blocks while the walk was in progress.
 */
static bool
HnswWalkPageChain(HnswVerifyState *state, BlockNumber startAfter,
				  bool chainStarted)
{
	BlockNumber blkno = HNSW_HEAD_BLKNO;
	BlockNumber blocksBefore;
	bool		grew = false;

	blocksBefore = RelationGetNumberOfBlocks(state->index);

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		OffsetNumber offno;
		OffsetNumber maxoffno;
		BlockNumber nextblkno;

		CHECK_FOR_INTERRUPTS();

		/* Already collected by an earlier pass */
		if (chainStarted && blkno <= startAfter)
		{
			if (blkno >= RelationGetNumberOfBlocks(state->index))
				break;

			buf = HnswReadSharedPage(state->index, blkno);
			page = BufferGetPage(buf);
			nextblkno = HnswPageGetOpaque(page)->nextblkno;
			UnlockReleaseBuffer(buf);
			blkno = nextblkno;
			continue;
		}

		if (blkno >= RelationGetNumberOfBlocks(state->index))
		{
			if (!BlockNumberIsValid(state->invalidPage))
			{
				state->invalidPage = blkno;
				HnswAddFinding(state, HNSW_SEVERITY_ERROR, "page_id_invalid",
							   -1, blkno, InvalidOffsetNumber,
							   "page chain points past end of relation");
			}
			break;
		}

		buf = HnswReadSharedPage(state->index, blkno);
		page = BufferGetPage(buf);

		if (HnswPageGetOpaque(page)->page_id != HNSW_PAGE_ID)
		{
			if (!BlockNumberIsValid(state->invalidPage))
			{
				state->invalidPage = blkno;
				HnswAddFinding(state, HNSW_SEVERITY_ERROR, "page_id_invalid",
							   -1, blkno, InvalidOffsetNumber,
							   "page special area does not identify an hnsw page");
			}
			UnlockReleaseBuffer(buf);

			/* Stop following this page; results collected so far survive */
			break;
		}

		maxoffno = PageGetMaxOffsetNumber(page);
		for (offno = FirstOffsetNumber; offno <= maxoffno;
			 offno = OffsetNumberNext(offno))
		{
			ItemId		itemid = PageGetItemId(page, offno);
			uint8		tupleType;

			if (!ItemIdIsUsed(itemid) || ItemIdGetLength(itemid) == 0)
			{
				HnswAddFinding(state, HNSW_SEVERITY_ERROR, "tuple_corrupt",
							   -1, blkno, offno,
							   "unused or zero-length slot in page chain");
				continue;
			}

			/* Safe: item id is used and has a length */
			tupleType = *((uint8 *) PageGetItem(page, itemid));

			if (tupleType == HNSW_ELEMENT_TUPLE_TYPE)
				HnswCollectElement(state, blkno, page, offno);

			/* Neighbor tuples are validated in the second pass */
		}

		nextblkno = HnswPageGetOpaque(page)->nextblkno;

		UnlockReleaseBuffer(buf);

		/* Reset per-page temporary allocations */
		MemoryContextReset(state->tmpCtx);

		if (blkno > state->lastSeenBlkno)
			state->lastSeenBlkno = blkno;

		if (RelationGetNumberOfBlocks(state->index) != blocksBefore)
		{
			grew = true;
			blocksBefore = RelationGetNumberOfBlocks(state->index);
		}

		blkno = nextblkno;
	}

	return grew;
}

/*
 * Page chain scan with tolerance for concurrent page appends.
 */
static void
HnswScanPageChain(HnswVerifyState *state)
{
	int			pass;

	/*
	 * Repeat the walk when blocks are appended underneath us. A pass that
	 * observes no size change settles the scan.
	 */
	for (pass = 0; pass < HNSW_VERIFY_MAX_CHAIN_PASSES; pass++)
	{
		BlockNumber before = RelationGetNumberOfBlocks(state->index);
		bool		grew;

		grew = HnswWalkPageChain(state, state->lastSeenBlkno,
								 state->chainStarted);
		state->chainStarted = true;

		if (!grew && RelationGetNumberOfBlocks(state->index) == before)
			break;
	}

	/* Refresh block count for later range checks */
	state->nblocks = RelationGetNumberOfBlocks(state->index);
}

/* ----------------------------------------------------------------
 * Pass 2: per-element neighbor validation
 * ----------------------------------------------------------------
 */

/*
 * Re-read an element tuple after its neighbor tuple has been read, and check
 * that the tuple has not been reused in between. Retries a bounded number of
 * times.
 */
static bool
HnswRereadElement(HnswVerifyState *state, ElementHashEntry *entry,
				  HnswElementSnapshot *snap)
{
	int			attempt;

	for (attempt = 0; attempt < HNSW_VERIFY_MAX_RETRIES; attempt++)
	{
		Buffer		buf;
		Page		page;
		BlockNumber blkno = ItemPointerGetBlockNumber(&entry->tid);
		OffsetNumber offno = ItemPointerGetOffsetNumber(&entry->tid);

		CHECK_FOR_INTERRUPTS();

		if (!BlockNumberIsValid(blkno) || blkno >= state->nblocks)
			return false;

		buf = HnswReadSharedPage(state->index, blkno);
		page = BufferGetPage(buf);

		if (HnswPageGetOpaque(page)->page_id != HNSW_PAGE_ID ||
			!HnswTupleHeaderOk(page, offno,
							   offsetof(HnswElementTupleData, data)))
		{
			UnlockReleaseBuffer(buf);
			return false;
		}

		HnswCopyElementTuple(blkno, page, offno, snap);

		UnlockReleaseBuffer(buf);

		if (snap->type == HNSW_ELEMENT_TUPLE_TYPE &&
			snap->version == entry->version &&
			snap->level == entry->level &&
			snap->deleted == entry->deleted &&
			ItemPointerEquals(&snap->firstHeapTid, &entry->firstHeapTid))
			return true;
	}

	return false;
}

/*
 * Validate the connections of a single element against the element hash.
 *
 * Concurrent inserts and VACUUM can reuse tuple slots, so the neighbor tuple
 * is read first and the element tuple is re-read immediately afterwards. The
 * two reads describe the same generation only when version, level, deleted
 * flag and the first heap TID still match the element collected earlier. The
 * whole read sequence is retried a bounded number of times.
 */
static void
HnswVerifyElementNeighbors(HnswVerifyState *state, ElementHashEntry *entry)
{
	BlockNumber elementBlkno = ItemPointerGetBlockNumber(&entry->tid);
	OffsetNumber elementOffno = ItemPointerGetOffsetNumber(&entry->tid);
	HnswNeighborSnapshot *neighbors = NULL;
	HnswElementSnapshot snap;
	bool		corrupt;
	bool		consistent = false;
	int			attempt;
	int			level = entry->level;
	int			expectedCount;
	int			idx;
	int			lc;
	int			*layerFanout;
	int			*layerEmpty;
	MemoryContext oldCtx;

	/* Elements with invalid levels skip connection validation */
	if (level > HnswGetMaxLevel(state->m))
		return;

	if (!BlockNumberIsValid(entry->neighborPage) ||
		entry->neighborPage >= state->nblocks ||
		!OffsetNumberIsValid(entry->neighborOffno))
	{
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "neighbortid_invalid",
					   level, elementBlkno, elementOffno,
					   "neighbortid points outside the relation");
		return;
	}

	/*
	 * Read the neighbor tuple first, then re-read the element, and retry the
	 * pair while the element tuple keeps changing underneath us.
	 */
	for (attempt = 0; attempt < HNSW_VERIFY_MAX_RETRIES; attempt++)
	{
		CHECK_FOR_INTERRUPTS();

		MemoryContextReset(state->tmpCtx);
		oldCtx = MemoryContextSwitchTo(state->tmpCtx);
		neighbors = HnswReadNeighborSnapshot(state, entry->neighborPage,
											 entry->neighborOffno, &corrupt);
		MemoryContextSwitchTo(oldCtx);

		if (corrupt || neighbors == NULL)
		{
			HnswAddFinding(state, HNSW_SEVERITY_ERROR, "neighbortid_invalid",
						   level, elementBlkno, elementOffno,
						   "neighbortid does not point to a readable tuple");
			MemoryContextReset(state->tmpCtx);
			return;
		}

		if (neighbors->type != HNSW_NEIGHBOR_TUPLE_TYPE)
		{
			HnswAddFinding(state, HNSW_SEVERITY_ERROR, "neighbor_tuple_type",
						   level, elementBlkno, elementOffno,
						   "neighbortid does not point to a neighbor tuple");
			MemoryContextReset(state->tmpCtx);
			return;
		}

		/* Re-read the element after the neighbor tuple */
		if (!HnswRereadElement(state, entry, &snap))
			continue;

		consistent = true;
		break;
	}

	if (!consistent)
	{
		HnswAddFinding(state, HNSW_SEVERITY_WARNING, "concurrent_modification",
					   level, elementBlkno, elementOffno,
					   "element tuple changed while reading neighbors");
		MemoryContextReset(state->tmpCtx);
		return;
	}

	expectedCount = (level + 2) * state->m;
	if (neighbors->count != expectedCount)
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "neighbor_count_mismatch",
					   level, elementBlkno, elementOffno,
					   psprintf("neighbor count is %u, expected %d",
								neighbors->count, expectedCount));

	/* Per-layer fanout bookkeeping, layers stored high to low in the tuple */
	oldCtx = MemoryContextSwitchTo(state->tmpCtx);
	layerFanout = palloc0(sizeof(int) * (level + 1));
	layerEmpty = palloc0(sizeof(int) * (level + 1));
	MemoryContextSwitchTo(oldCtx);

	idx = 0;
	for (lc = level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(state->m, lc);
		int			validCount = 0;
		bool		gap = false;
		int			i;

		for (i = 0; i < lm; i++)
		{
			ItemPointer indextid;
			int			slot = idx + i;

			if (slot >= neighbors->slots || slot >= neighbors->count)
			{
				/*
				 * Tuple cannot hold this slot. Stats truncate to what fits;
				 * a count mismatch (if any) was already reported above.
				 */
				gap = true;
				continue;
			}

			indextid = &neighbors->indextids[slot];

			if (!ItemPointerIsValid(indextid))
			{
				gap = true;
				continue;
			}

			if (gap)
			{
				HnswAddFinding(state, HNSW_SEVERITY_ERROR, "neighbor_gap",
							   lc, elementBlkno, elementOffno,
							   psprintf("valid neighbor follows an empty slot at level %d", lc));
				continue;
			}

			validCount++;

			/* Range and type checks use only the element hash */
			if (ItemPointerGetBlockNumber(indextid) >= state->nblocks ||
				!OffsetNumberIsValid(ItemPointerGetOffsetNumber(indextid)))
			{
				HnswAddFinding(state, HNSW_SEVERITY_ERROR,
							   "neighbor_tid_out_of_range",
							   lc, elementBlkno, elementOffno,
							   "neighbor TID points outside the relation");
				continue;
			}
			else
			{
				ElementHashEntry *neighborEntry;

				neighborEntry = elementhash_lookup(state->elements,
												   *indextid);
				if (neighborEntry == NULL)
				{
					HnswAddFinding(state, HNSW_SEVERITY_ERROR,
								   "neighbor_not_element",
								   lc, elementBlkno, elementOffno,
								   "neighbor TID does not point to an element tuple");
					continue;
				}

				if (neighborEntry->level < lc)
					HnswAddFinding(state, HNSW_SEVERITY_ERROR,
								   "neighbor_level_too_low",
								   lc, elementBlkno, elementOffno,
								   psprintf("neighbor level %d below connection level %d",
											neighborEntry->level, lc));

				if (neighborEntry->deleted)
					HnswAddFinding(state, HNSW_SEVERITY_WARNING,
								   "deleted_neighbor",
								   lc, elementBlkno, elementOffno,
								   "connection points to a deleted element");
			}
		}

		layerFanout[lc] = validCount;
		if (validCount == 0)
			layerEmpty[lc] = 1;

		idx += lm;
	}

	/*
	 * Per-level fanout statistics describe live elements only; deleted
	 * elements have their connection areas cleared by VACUUM.
	 */
	if (!entry->deleted)
	{
		for (lc = 0; lc <= level; lc++)
		{
			HnswLevelStats *ls = &state->levelStats[lc];

			if (!ls->fanoutSet || layerFanout[lc] < ls->fanoutMin)
				ls->fanoutMin = layerFanout[lc];
			if (!ls->fanoutSet || layerFanout[lc] > ls->fanoutMax)
				ls->fanoutMax = layerFanout[lc];
			ls->fanoutSet = true;
			ls->emptyNeighbors += layerEmpty[lc];
		}
	}

	MemoryContextReset(state->tmpCtx);
}

/*
 * Second pass over every element in the hash.
 */
static void
HnswVerifyNeighbors(HnswVerifyState *state)
{
	elementhash_iterator iter;
	ElementHashEntry *entry;

	elementhash_start_iterate(state->elements, &iter);
	while ((entry = elementhash_iterate(state->elements, &iter)) != NULL)
	{
		CHECK_FOR_INTERRUPTS();
		HnswVerifyElementNeighbors(state, entry);
	}
}

/*
 * Populate element/deleted/heaptid counters per level after both passes.
 */
static void
HnswBuildLevelCounters(HnswVerifyState *state)
{
	elementhash_iterator iter;
	ElementHashEntry *entry;
	int			L;

	if (state->maxLevel < 0)
		return;

	elementhash_start_iterate(state->elements, &iter);
	while ((entry = elementhash_iterate(state->elements, &iter)) != NULL)
	{
		/* Invalid-level elements get no layer counters */
		if (entry->level < 0 || entry->level > HnswGetMaxLevel(state->m))
			continue;

		for (L = 0; L <= entry->level; L++)
		{
			if (entry->deleted)
				state->levelStats[L].deleted++;
			else
			{
				state->levelStats[L].elements++;

				/* Heap TIDs are only accumulated on the L=0 row */
				if (L == 0)
					state->levelStats[L].heaptids += entry->heaptidsLength;
			}
		}
	}
}

/* ----------------------------------------------------------------
 * Entry point checks
 * ----------------------------------------------------------------
 */
static void
HnswCheckEntryPoint(HnswVerifyState *state)
{
	elementhash_iterator iter2;
	ElementHashEntry *cur;
	ElementHashEntry *entry;
	ItemPointerData entryTid;
	bool		haveLiveElements = false;

	/* Any live (non-deleted), valid-level element in the graph? */
	elementhash_start_iterate(state->elements, &iter2);
	while ((cur = elementhash_iterate(state->elements, &iter2)) != NULL)
	{
		if (!cur->deleted && cur->level <= HnswGetMaxLevel(state->m))
		{
			haveLiveElements = true;
			break;
		}
	}

	if (!BlockNumberIsValid(state->entryBlkno))
	{
		if (haveLiveElements)
			HnswAddFinding(state, HNSW_SEVERITY_ERROR, "entry_invalid",
						   -1, InvalidBlockNumber, InvalidOffsetNumber,
						   "graph contains live elements but has no entry point");
		return;
	}

	ItemPointerSet(&entryTid, state->entryBlkno, state->entryOffno);
	entry = elementhash_lookup(state->elements, entryTid);

	if (entry == NULL)
	{
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "entry_invalid",
					   state->entryLevel, state->entryBlkno,
					   state->entryOffno,
					   "entry point does not point to an element");
		return;
	}

	if (entry->level != state->entryLevel || entry->deleted)
		HnswAddFinding(state, HNSW_SEVERITY_ERROR, "entry_invalid",
					   state->entryLevel, state->entryBlkno,
					   state->entryOffno,
					   "entry element level does not match the metapage");
}

/* ----------------------------------------------------------------
 * Full mode: layered BFS reachability
 * ----------------------------------------------------------------
 */
static void
HnswBfs(HnswVerifyState *state)
{
	struct tidhash_hash *visited;
	ItemPointerData *queue;
	int			queueHead = 0;
	int			queueTail = 0;
	int			queueCapacity;
	ItemPointerData entryTid;
	ElementHashEntry *entry;
	int			visitedCount = 0;
	bool		incomplete = false;

	if (!BlockNumberIsValid(state->entryBlkno))
		return;

	ItemPointerSet(&entryTid, state->entryBlkno, state->entryOffno);
	entry = elementhash_lookup(state->elements, entryTid);
	if (entry == NULL || entry->deleted)
		return;						/* entry problems already reported */

	visited = tidhash_create(state->persistentCtx, 256, NULL);

	queueCapacity = 256;
	queue = MemoryContextAlloc(state->persistentCtx,
							   sizeof(ItemPointerData) * queueCapacity);

	/* Seed the queue with the entry point and mark it visited */
	queue[queueTail++] = entryTid;
	{
		bool		found;

		tidhash_insert(visited, entryTid, &found);
		visitedCount++;
	}

	while (queueHead < queueTail)
	{
		ItemPointerData current = queue[queueHead++];
		ElementHashEntry *curEntry;
		HnswNeighborSnapshot *neighbors;
		bool		corrupt;
		int			idx;
		int			lc;
		MemoryContext oldCtx;

		CHECK_FOR_INTERRUPTS();

		if (visitedCount >= state->maxVisited)
		{
			incomplete = true;
			break;
		}

		curEntry = elementhash_lookup(state->elements, current);
		if (curEntry == NULL || curEntry->deleted)
			continue;

		oldCtx = MemoryContextSwitchTo(state->tmpCtx);
		neighbors = HnswReadNeighborSnapshot(state,
											 curEntry->neighborPage,
											 curEntry->neighborOffno,
											 &corrupt);
		MemoryContextSwitchTo(oldCtx);

		if (corrupt || neighbors == NULL ||
			neighbors->type != HNSW_NEIGHBOR_TUPLE_TYPE)
		{
			/* Structural problem already reported; treat as no edges */
			MemoryContextReset(state->tmpCtx);
			continue;
		}

		/* Expand each layer this element participates in */
		idx = 0;
		for (lc = curEntry->level; lc >= 0; lc--)
		{
			int			lm = HnswGetLayerM(state->m, lc);
			int			i;

			for (i = 0; i < lm; i++)
			{
				int			slot = idx + i;
				ItemPointer candidate;
				ElementHashEntry *candidateEntry;
				bool		found;

				if (slot >= neighbors->slots || slot >= neighbors->count)
					break;

				candidate = &neighbors->indextids[slot];
				if (!ItemPointerIsValid(candidate))
					break;

				if (tidhash_lookup(visited, *candidate) != NULL)
					continue;

				tidhash_insert(visited, *candidate, &found);
				visitedCount++;

				if (visitedCount >= state->maxVisited)
				{
					incomplete = true;
					break;
				}

				candidateEntry = elementhash_lookup(state->elements,
													*candidate);
				if (candidateEntry == NULL || candidateEntry->deleted)
					continue;

				/* Grow the queue if needed */
				if (queueTail >= queueCapacity)
				{
					queueCapacity *= 2;
					queue = repalloc(queue,
									 sizeof(ItemPointerData) * queueCapacity);
				}

				queue[queueTail++] = *candidate;
			}

			idx += lm;

			if (incomplete)
				break;
		}

		MemoryContextReset(state->tmpCtx);

		if (incomplete)
			break;
	}

	if (incomplete)
	{
		HnswAddFinding(state, HNSW_SEVERITY_WARNING,
					   "reachability_incomplete", -1,
					   InvalidBlockNumber, InvalidOffsetNumber,
					   psprintf("BFS stopped after visiting %d elements",
								visitedCount));
	}
	else
	{
		/* Report live elements the BFS could not reach */
		elementhash_iterator viter;
		ElementHashEntry *cur;

		elementhash_start_iterate(state->elements, &viter);
		while ((cur = elementhash_iterate(state->elements, &viter)) != NULL)
		{
			CHECK_FOR_INTERRUPTS();

			if (cur->deleted || cur->level > HnswGetMaxLevel(state->m))
				continue;

			if (tidhash_lookup(visited, cur->tid) == NULL)
				HnswAddFinding(state, HNSW_SEVERITY_WARNING,
							   "unreachable_element", cur->level,
							   ItemPointerGetBlockNumber(&cur->tid),
							   ItemPointerGetOffsetNumber(&cur->tid),
							   "live element is not reachable from the entry point");
		}
	}

	tidhash_destroy(visited);
	pfree(queue);
}

/* ----------------------------------------------------------------
 * Finding sort and truncation
 * ----------------------------------------------------------------
 */

static int
HnswFindingCompare(const void *a, const void *b)
{
	const HnswFinding *fa = (const HnswFinding *) a;
	const HnswFinding *fb = (const HnswFinding *) b;
	BlockNumber ba = BlockNumberIsValid(fa->blkno) ? fa->blkno : PG_UINT32_MAX;
	BlockNumber bb = BlockNumberIsValid(fb->blkno) ? fb->blkno : PG_UINT32_MAX;
	OffsetNumber oa = OffsetNumberIsValid(fa->offno) ? fa->offno : PG_UINT16_MAX;
	OffsetNumber ob = OffsetNumberIsValid(fb->offno) ? fb->offno : PG_UINT16_MAX;
	int			la = (fa->level >= 0) ? fa->level : PG_INT16_MAX;
	int			lb = (fb->level >= 0) ? fb->level : PG_INT16_MAX;
	int			cmp;

	if (ba != bb)
		return (ba < bb) ? -1 : 1;

	if (oa != ob)
		return (oa < ob) ? -1 : 1;

	if (la != lb)
		return (la < lb) ? -1 : 1;

	cmp = strcmp(fa->code, fb->code);
	if (cmp != 0)
		return cmp;

	return strcmp(fa->severity, fb->severity);
}

/*
 * Sort findings and enforce the max_findings cap.
 */
static void
HnswFinalizeFindings(HnswVerifyState *state)
{
	if (state->findingsLength == 0)
		return;

	qsort(state->findings, state->findingsLength, sizeof(HnswFinding),
		  HnswFindingCompare);

	if (state->findingsLength > state->maxFindings)
	{
		/* Keep the first N and append a truncation marker */
		state->findingsLength = state->maxFindings;
		HnswAddFinding(state, HNSW_SEVERITY_WARNING, "findings_truncated",
					   -1, InvalidBlockNumber, InvalidOffsetNumber,
					   psprintf("only the first %d findings are shown",
								state->maxFindings));
	}
}

/* ----------------------------------------------------------------
 * Shared verification driver
 * ----------------------------------------------------------------
 */

static void
HnswVerifyInit(HnswVerifyState *state, Oid indexOid, bool full,
			   int maxFindings, int maxVisited)
{
	memset(state, 0, sizeof(HnswVerifyState));

	state->full = full;
	state->maxFindings = maxFindings;
	state->maxVisited = maxVisited;
	state->maxLevel = -1;
	state->entryLevel = -1;
	state->entryBlkno = InvalidBlockNumber;
	state->entryOffno = InvalidOffsetNumber;
	state->lastSeenBlkno = 0;
	state->chainStarted = false;
	state->invalidPage = InvalidBlockNumber;

	HnswVerifyOpenIndex(indexOid, &state->index, AccessShareLock);

	state->persistentCtx = AllocSetContextCreate(CurrentMemoryContext,
												 "Hnsw verify persistent context",
												 ALLOCSET_DEFAULT_SIZES);
	state->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
										  "Hnsw verify page context",
										  ALLOCSET_DEFAULT_SIZES);

	state->elements = elementhash_create(state->persistentCtx, 256, NULL);
	state->heapTids = tidhash_create(state->persistentCtx, 256, NULL);
	state->findings = NULL;
	state->findingsLength = 0;
	state->findingsCapacity = 0;
}

static void
HnswVerifyFree(HnswVerifyState *state)
{
	if (state->tmpCtx != NULL)
		MemoryContextDelete(state->tmpCtx);

	if (state->persistentCtx != NULL)
		MemoryContextDelete(state->persistentCtx);

	if (state->index != NULL)
		index_close(state->index, AccessShareLock);
}

/*
 * Run the full pipeline. Page-level heavyweight locks are held for the whole
 * scan, in the fixed order update lock then scan lock, and released in
 * reverse on every exit path.
 */
static void
HnswRunVerification(HnswVerifyState *state)
{
	bool		locksHeld = false;

	/* Phase 1: metapage validation */
	HnswReadMetaPage(state);

	/* Fixed lock order: HNSW_UPDATE_LOCK, then HNSW_SCAN_LOCK */
	LockPage(state->index, HNSW_UPDATE_LOCK, ShareLock);
	LockPage(state->index, HNSW_SCAN_LOCK, ShareLock);
	locksHeld = true;

	PG_TRY();
	{
		MemoryContext oldCtx;

		/* Phase 2: page chain scan and element hash construction */
		HnswScanPageChain(state);

		if (state->maxLevel >= 0)
		{
			oldCtx = MemoryContextSwitchTo(state->persistentCtx);
			state->levelStats = palloc0(sizeof(HnswLevelStats) *
										(state->maxLevel + 1));
			MemoryContextSwitchTo(oldCtx);
		}

		/* Phase 3: per-element neighbor validation (also fills fanout) */
		HnswVerifyNeighbors(state);

		/* Layer element/deleted/heaptid counters */
		HnswBuildLevelCounters(state);

		/* Phase 4: entry point validation */
		HnswCheckEntryPoint(state);

		/* Phase 5: reachability in full mode */
		if (state->full)
			HnswBfs(state);

		/* Phase 6: sort and truncate */
		HnswFinalizeFindings(state);
	}
	PG_CATCH();
	{
		if (locksHeld)
		{
			/* Reverse acquisition order */
			UnlockPage(state->index, HNSW_SCAN_LOCK, ShareLock);
			UnlockPage(state->index, HNSW_UPDATE_LOCK, ShareLock);
		}
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Reverse acquisition order */
	UnlockPage(state->index, HNSW_SCAN_LOCK, ShareLock);
	UnlockPage(state->index, HNSW_UPDATE_LOCK, ShareLock);
}

/* ----------------------------------------------------------------
 * Result emission
 * ----------------------------------------------------------------
 */

static void
HnswEmitFindings(FunctionCallInfo fcinfo, HnswVerifyState *state)
{
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext oldCtx;
	int			i;
	Datum		values[6];
	bool		nulls[6];
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in context that cannot accept type record")));
	tupdesc = BlessTupleDesc(tupdesc);

	oldCtx = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
	tupstore = tuplestore_begin_heap(true, false, work_mem);
	MemoryContextSwitchTo(oldCtx);

	for (i = 0; i < state->findingsLength; i++)
	{
		HnswFinding *finding = &state->findings[i];
		if (i < 3)

		memset(nulls, 0, sizeof(nulls));

		values[0] = CStringGetTextDatum(finding->severity);
		values[1] = CStringGetTextDatum(finding->code);

		if (finding->level < 0)
		{
			values[2] = (Datum) 0;
			nulls[2] = true;
		}
		else
			values[2] = Int32GetDatum(finding->level);

		if (!BlockNumberIsValid(finding->blkno))
		{
			values[3] = (Datum) 0;
			nulls[3] = true;
		}
		else
			values[3] = Int64GetDatum((int64) finding->blkno);

		if (!OffsetNumberIsValid(finding->offno))
		{
			values[4] = (Datum) 0;
			nulls[4] = true;
		}
		else
			values[4] = Int32GetDatum((int32) finding->offno);

		if (finding->detail == NULL)
		{
			values[5] = (Datum) 0;
			nulls[5] = true;
		}
		else
			values[5] = CStringGetTextDatum(finding->detail);

		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
	}

	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;
}

static void
HnswEmitStats(FunctionCallInfo fcinfo, HnswVerifyState *state)
{
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext oldCtx;
	int			L;
	Datum		values[7];
	bool		nulls[7];
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in context that cannot accept type record")));
	tupdesc = BlessTupleDesc(tupdesc);

	oldCtx = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
	tupstore = tuplestore_begin_heap(true, false, work_mem);
	MemoryContextSwitchTo(oldCtx);

	memset(nulls, 0, sizeof(nulls));

	for (L = 0; L <= state->maxLevel; L++)
	{
		HnswLevelStats *ls = &state->levelStats[L];

		values[0] = Int32GetDatum(L);
		values[1] = Int64GetDatum(ls->elements);
		values[2] = Int64GetDatum(ls->deleted);
		values[3] = Int64GetDatum(ls->heaptids);
		values[4] = Int32GetDatum(ls->fanoutSet ? ls->fanoutMin : 0);
		values[5] = Int32GetDatum(ls->fanoutSet ? ls->fanoutMax : 0);
		values[6] = Int64GetDatum(ls->emptyNeighbors);

		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
	}

	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;
}

/* ----------------------------------------------------------------
 * SQL entry points
 * ----------------------------------------------------------------
 */

PG_FUNCTION_INFO_V1(hnsw_index_info);
Datum
hnsw_index_info(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	TupleDesc	tupdesc;
	Datum		values[9];
	bool		nulls[9];
	HeapTuple	tuple;

	HnswVerifyOpenIndex(indexOid, &index, AccessShareLock);

	buf = HnswReadSharedPage(index, HNSW_METAPAGE_BLKNO);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (metap->magicNumber != HNSW_MAGIC_NUMBER ||
		metap->version != HNSW_VERSION)
	{
		UnlockReleaseBuffer(buf);
		index_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hnsw index \"%s\" has invalid magic number or version",
						RelationGetRelationName(index))));
	}

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in context that cannot accept type record")));
	tupdesc = BlessTupleDesc(tupdesc);

	memset(nulls, 0, sizeof(nulls));

	values[0] = Int32GetDatum((int32) metap->version);
	values[1] = Int32GetDatum((int32) metap->dimensions);
	values[2] = Int32GetDatum((int32) metap->m);
	values[3] = Int32GetDatum((int32) metap->efConstruction);
	values[4] = Int32GetDatum((int32) metap->entryLevel);

	if (BlockNumberIsValid(metap->entryBlkno))
	{
		values[5] = Int64GetDatum((int64) metap->entryBlkno);
		values[6] = Int32GetDatum((int32) metap->entryOffno);
	}
	else
	{
		values[5] = (Datum) 0;
		nulls[5] = true;
		values[6] = (Datum) 0;
		nulls[6] = true;
	}

	values[7] = Int64GetDatum((int64) metap->insertPage);
	values[8] = Int64GetDatum((int64) RelationGetNumberOfBlocks(index));

	UnlockReleaseBuffer(buf);
	index_close(index, AccessShareLock);

	tuple = heap_form_tuple(tupdesc, values, nulls);
	return HeapTupleGetDatum(tuple);
}

PG_FUNCTION_INFO_V1(hnsw_graph_stats);
Datum
hnsw_graph_stats(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	HnswVerifyState state;

	HnswVerifyInit(&state, indexOid, false, 100, 1000000);

	PG_TRY();
	{
		HnswRunVerification(&state);
	}
	PG_CATCH();
	{
		HnswVerifyFree(&state);
		PG_RE_THROW();
	}
	PG_END_TRY();

	HnswEmitStats(fcinfo, &state);

	HnswVerifyFree(&state);
	PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(hnsw_verify);
Datum
hnsw_verify(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	text	   *modeText;
	char	   *mode;
	int			maxFindings;
	int			maxVisited;
	bool		full = false;
	HnswVerifyState state;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) ||
		PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("arguments must not be NULL")));

	modeText = PG_GETARG_TEXT_PP(1);
	mode = text_to_cstring(modeText);

	if (strcmp(mode, "full") == 0)
		full = true;
	else if (strcmp(mode, "structure") != 0)
	{
		pfree(mode);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("mode must be \"structure\" or \"full\"")));
	}

	pfree(mode);

	maxFindings = PG_GETARG_INT32(2);
	maxVisited = PG_GETARG_INT32(3);

	if (maxFindings < 1 || maxVisited < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("max_findings and max_visited must be greater than zero")));

	HnswVerifyInit(&state, indexOid, full, maxFindings, maxVisited);

	PG_TRY();
	{
		HnswRunVerification(&state);
	}
	PG_CATCH();
	{
		HnswVerifyFree(&state);
		PG_RE_THROW();
	}
	PG_END_TRY();

	HnswEmitFindings(fcinfo, &state);

	HnswVerifyFree(&state);
	PG_RETURN_NULL();
}
