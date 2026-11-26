use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim = 128;
my $m = 8;
my $ef = 32;
my $rows = 20000;
my $array_sql = join(",", ('random()') x $dim);

# Initialize node
my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->append_conf(
	'postgresql.conf', qq(
	maintenance_work_mem = '64MB'
	max_parallel_maintenance_workers = 4
	min_parallel_table_scan_size = 0
	min_parallel_index_scan_size = 0
));
$node->start;
$node->safe_psql("postgres", "CREATE EXTENSION vector;");

sub count_errors
{
	my ($idx, $mode) = @_;
	$mode //= 'structure';
	return $node->safe_psql("postgres",
		"SELECT count(*) FROM hnsw_verify('$idx', '$mode') WHERE severity = 'error';"
	);
}

# ----------------------------------------------------------------------
# 1. Serial, parallel, concurrent, and flushed builds have no errors
# ----------------------------------------------------------------------

# Serial build
$node->safe_psql("postgres", "CREATE TABLE ser (v vector($dim));");
$node->safe_psql("postgres",
	"INSERT INTO ser SELECT ARRAY[$array_sql] FROM generate_series(1, $rows) i;");
$node->safe_psql("postgres",
	"CREATE INDEX ser_idx ON ser USING hnsw (v vector_l2_ops) WITH (m = $m, ef_construction = $ef);");
is(count_errors('ser_idx', 'full'), '0', 'serial build has no errors');

# Parallel build
$node->safe_psql("postgres", "CREATE TABLE par (v vector($dim));");
$node->safe_psql("postgres", "ALTER TABLE par SET (parallel_workers = 4);");
$node->safe_psql("postgres",
	"INSERT INTO par SELECT ARRAY[$array_sql] FROM generate_series(1, $rows) i;");
$node->safe_psql("postgres", "ANALYZE par;");
my ($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET client_min_messages = DEBUG;
	CREATE INDEX par_idx ON par USING hnsw (v vector_l2_ops) WITH (m = $m, ef_construction = $ef);
));
is($ret, 0, 'parallel build succeeds');
like($stderr, qr/using \d+ parallel workers/, 'parallel workers used');
is(count_errors('par_idx', 'full'), '0', 'parallel build has no errors');

# Concurrent build
$node->safe_psql("postgres", "CREATE TABLE conc (v vector($dim));");
$node->safe_psql("postgres",
	"INSERT INTO conc SELECT ARRAY[$array_sql] FROM generate_series(1, 1000) i;");
$node->safe_psql("postgres",
	"CREATE INDEX CONCURRENTLY conc_idx ON conc USING hnsw (v vector_l2_ops) WITH (m = $m, ef_construction = $ef);");
is(count_errors('conc_idx', 'full'), '0', 'concurrent build has no errors');

# Flushed (low memory) build
$node->safe_psql("postgres", "CREATE TABLE fl (v vector($dim));");
$node->safe_psql("postgres",
	"INSERT INTO fl SELECT ARRAY[$array_sql] FROM generate_series(1, 2000) i;");
($ret, $stdout, $stderr) = $node->psql("postgres", qq(
	SET maintenance_work_mem = '1MB';
	CREATE INDEX fl_idx ON fl USING hnsw (v vector_l2_ops) WITH (m = $m, ef_construction = $ef);
));
is($ret, 0, 'flushed build succeeds');
like($stderr, qr/no longer fits into maintenance_work_mem/, 'graph flushed to disk');
is(count_errors('fl_idx', 'full'), '0', 'flushed build has no errors');

# ----------------------------------------------------------------------
# 2. Ground layer stats match the table and levels stay within bounds
# ----------------------------------------------------------------------
my $stats = $node->safe_psql("postgres",
	"SELECT elements, heaptids FROM hnsw_graph_stats('ser_idx') WHERE level = 0;");
my ($l0_elements, $l0_heaptids) = split(/\|/, $stats);
is($l0_elements, $rows, 'L0 elements equal table rows');
is($l0_heaptids, $rows, 'L0 heaptids equal table rows');

my $max_level = $node->safe_psql("postgres",
	"SELECT max(level) FROM hnsw_graph_stats('ser_idx');");
cmp_ok($max_level, '<=', 63, 'max level does not exceed 63');

# ----------------------------------------------------------------------
# 3. Concurrent inserts, then full verify
# ----------------------------------------------------------------------
$node->safe_psql("postgres", "CREATE TABLE cins (v vector($dim));");
$node->safe_psql("postgres",
	"CREATE INDEX cins_idx ON cins USING hnsw (v vector_l2_ops) WITH (m = $m, ef_construction = $ef);");
$node->pgbench(
	"--no-vacuum --client=10 --transactions=50",
	0,
	[qr{actually processed}],
	[qr{^$}],
	"concurrent INSERTs",
	{
		"045_hnsw_verify_insert" =>
			"INSERT INTO cins SELECT ARRAY[$array_sql] FROM generate_series(1, 10) i;"
	}
);

is(count_errors('cins_idx', 'full'), '0', 'no errors after concurrent inserts');

my $entry_level = $node->safe_psql("postgres",
	"SELECT (hnsw_index_info('cins_idx')).entry_level;");
cmp_ok($entry_level, '>=', 0, 'entry level is non-negative after inserts');

my $cins_rows = $node->safe_psql("postgres", "SELECT count(*) FROM cins;");
my $cins_heaptids = $node->safe_psql("postgres",
	"SELECT heaptids FROM hnsw_graph_stats('cins_idx') WHERE level = 0;");
is($cins_heaptids, $cins_rows, 'L0 heaptids equal table rows after concurrent inserts');

# unreachable_element warnings are allowed, so only errors are asserted above
my $incomplete = $node->safe_psql("postgres",
	"SELECT count(*) FROM hnsw_verify('ser_idx', 'full', 100, 1) WHERE code = 'reachability_incomplete';");
is($incomplete, '1', 'exactly one reachability_incomplete at max_visited = 1');

# ----------------------------------------------------------------------
# 4. Interleaved VACUUM and inserts leave no errors and no residual locks
# ----------------------------------------------------------------------
$node->safe_psql("postgres", "CREATE TABLE vi (i serial, v vector($dim));");
$node->safe_psql("postgres",
	"INSERT INTO vi (v) SELECT ARRAY[$array_sql] FROM generate_series(1, 1000) i;");
$node->safe_psql("postgres",
	"CREATE INDEX vi_idx ON vi USING hnsw (v vector_l2_ops) WITH (m = $m, ef_construction = $ef);");

for my $round (1 .. 20)
{
	$node->safe_psql("postgres",
		"INSERT INTO vi (v) SELECT ARRAY[$array_sql] FROM generate_series(1, 20) i;");
	$node->safe_psql("postgres",
		"DELETE FROM vi WHERE i IN (SELECT i FROM vi LIMIT 10);");
	$node->safe_psql("postgres", "VACUUM vi;");
	is(count_errors('vi_idx'), '0', "no errors after vacuum/insert round $round");
}

my $locks = $node->safe_psql("postgres",
	"SELECT count(*) FROM pg_locks WHERE relation = 'vi_idx'::regclass;");
is($locks, '0', 'no residual locks on the index');

# ----------------------------------------------------------------------
# 5. All four types, m = 2 and m = 100, continuous levels, stable stats
# ----------------------------------------------------------------------
$node->safe_psql("postgres", "CREATE TABLE tvec (v vector(8));");
$node->safe_psql("postgres", "CREATE TABLE thalf (v halfvec(8));");
$node->safe_psql("postgres", "CREATE TABLE tbit (v bit(32));");
$node->safe_psql("postgres", "CREATE TABLE tsparse (v sparsevec(8));");

my $arr8 = join(",", ('random()') x 8);
$node->safe_psql("postgres",
	"INSERT INTO tvec SELECT ARRAY[$arr8]::vector FROM generate_series(1, 400) i;");
$node->safe_psql("postgres",
	"INSERT INTO thalf SELECT ARRAY[$arr8]::halfvec FROM generate_series(1, 400) i;");
$node->safe_psql("postgres",
	"INSERT INTO tbit SELECT (random()*4294967295)::bigint::bit(32) FROM generate_series(1, 400) i;");
$node->safe_psql("postgres",
	"INSERT INTO tsparse SELECT ARRAY[$arr8]::sparsevec FROM generate_series(1, 400) i;");

my %opclass = (
	tvec => 'vector_l2_ops',
	thalf => 'halfvec_l2_ops',
	tbit => 'bit_hamming_ops',
	tsparse => 'sparsevec_l2_ops'
);

for my $tbl (qw(tvec thalf tbit tsparse))
{
	for my $this_m (2, 100)
	{
		my $idx = "${tbl}_${this_m}_idx";
		$node->safe_psql("postgres",
			"CREATE INDEX $idx ON $tbl USING hnsw (v $opclass{$tbl}) WITH (m = $this_m, ef_construction = " . (2 * $this_m) . ");");
		is(count_errors($idx, 'full'), '0', "$tbl m=$this_m has no errors");

		# Levels are continuous 0..max(level)
		my $levels = $node->safe_psql("postgres",
			"SELECT string_agg(level::text, ',' ORDER BY level) FROM hnsw_graph_stats('$idx');");
		my $top = $node->safe_psql("postgres",
			"SELECT max(level) FROM hnsw_graph_stats('$idx');");
		my @expected = (0 .. $top);
		is($levels, join(",", @expected), "$tbl m=$this_m has continuous levels");

		# Two stats calls return identical results
		my $a = $node->safe_psql("postgres",
			"SELECT * FROM hnsw_graph_stats('$idx') ORDER BY level;");
		my $b = $node->safe_psql("postgres",
			"SELECT * FROM hnsw_graph_stats('$idx') ORDER BY level;");
		is($a, $b, "$tbl m=$this_m stats are stable across calls");
	}
}

# ----------------------------------------------------------------------
# 6. Empty index
# ----------------------------------------------------------------------
$node->safe_psql("postgres", "CREATE TABLE tempty (v vector(4));");
$node->safe_psql("postgres",
	"CREATE INDEX empty_idx ON tempty USING hnsw (v vector_l2_ops);");
my $info = $node->safe_psql("postgres",
	"SELECT (hnsw_index_info('empty_idx')).pages, (hnsw_index_info('empty_idx')).entry_level;");
is($info, "2|-1", 'empty index reports 2 pages and entry_level -1');
is($node->safe_psql("postgres", "SELECT count(*) FROM hnsw_graph_stats('empty_idx');"),
	'0', 'empty index stats return no rows');
is($node->safe_psql("postgres", "SELECT count(*) FROM hnsw_verify('empty_idx', 'full');"),
	'0', 'empty index verify returns no rows');

# Emitted order matches the documented ordering
my $order_diff = $node->safe_psql("postgres", q{
	SELECT count(*) FROM (
		(SELECT severity, code, level, blkno, offno, detail FROM hnsw_verify('ser_idx', 'full', 100, 1))
		EXCEPT ALL
		(SELECT severity, code, level, blkno, offno, detail FROM hnsw_verify('ser_idx', 'full', 100, 1)
		 ORDER BY blkno NULLS LAST, offno NULLS LAST, level NULLS LAST, code)
	) z;
});
is($order_diff, '0', 'verify output follows the documented ordering');

# ----------------------------------------------------------------------
# 7. Error handling: wrong relation type and bad arguments
# ----------------------------------------------------------------------
$node->safe_psql("postgres", "CREATE TABLE btt (id int);");
$node->safe_psql("postgres", "INSERT INTO btt VALUES (1);");
$node->safe_psql("postgres", "CREATE INDEX btree_idx ON btt (id);");
for my $fn (qw(hnsw_index_info hnsw_graph_stats hnsw_verify))
{
	my ($br, $bout, $berr) =
		$node->psql("postgres", "\\set VERBOSITY sqlstate\nSELECT * FROM $fn('btree_idx');");
	isnt($br, 0, "$fn fails on a btree index");
	like($berr, qr/42809/, "$fn rejects a btree index with 42809");
}

my ($mr, $mout, $merr) =
	$node->psql("postgres", "\\set VERBOSITY sqlstate\nSELECT * FROM hnsw_verify('ser_idx', 'bad');");
like($merr, qr/22023/, 'bad mode raises 22023');
($mr, $mout, $merr) =
	$node->psql("postgres", "\\set VERBOSITY sqlstate\nSELECT * FROM hnsw_verify('ser_idx', 'structure', 0);");
like($merr, qr/22023/, 'max_findings = 0 raises 22023');
($mr, $mout, $merr) =
	$node->psql("postgres", "\\set VERBOSITY sqlstate\nSELECT * FROM hnsw_verify('ser_idx', 'structure', 100, 0);");
like($merr, qr/22023/, 'max_visited = 0 raises 22023');

# ----------------------------------------------------------------------
# 8. Read-only behavior: no WAL and no size change inside a transaction
# ----------------------------------------------------------------------
my $readonly = $node->safe_psql("postgres", q{
	DO $$
	DECLARE
		lsn1 pg_lsn;
		lsn2 pg_lsn;
		sz1 bigint;
		sz2 bigint;
	BEGIN
		lsn1 := pg_current_wal_lsn();
		sz1 := pg_relation_size('ser_idx');
		PERFORM * FROM hnsw_verify('ser_idx', 'full');
		PERFORM * FROM hnsw_graph_stats('ser_idx');
		PERFORM hnsw_index_info('ser_idx');
		lsn2 := pg_current_wal_lsn();
		sz2 := pg_relation_size('ser_idx');
		ASSERT lsn1 = lsn2, 'WAL changed';
		ASSERT sz1 = sz2, 'relation size changed';
	END $$;
	SELECT 'ok';
});
is($readonly, 'ok', 'verify is read-only: no WAL or size change');

# Row-by-row determinism
my $v1 = $node->safe_psql("postgres",
	"SELECT * FROM hnsw_verify('ser_idx', 'full') ORDER BY blkno NULLS LAST, offno NULLS LAST, level NULLS LAST, code, severity, detail;");
my $v2 = $node->safe_psql("postgres",
	"SELECT * FROM hnsw_verify('ser_idx', 'full') ORDER BY blkno NULLS LAST, offno NULLS LAST, level NULLS LAST, code, severity, detail;");
is($v1, $v2, 'repeated verify calls return identical rows');

# ----------------------------------------------------------------------
# 9. Statement cancellation leaves no locks and the index remains usable
# ----------------------------------------------------------------------
($ret, my $vstdout, my $vstderr) = $node->psql("postgres", qq(
	\\set VERBOSITY sqlstate
	SET statement_timeout = '1ms';
	SELECT count(*) FROM hnsw_verify('ser_idx', 'full', 100, 100000000);
));
like($vstderr, qr/57014/, 'statement timeout cancels verify with 57014');

$node->safe_psql("postgres", "RESET statement_timeout;");
is(count_errors('ser_idx', 'full'), '0', 'verify works after cancellation');
is($node->safe_psql("postgres",
	"SELECT count(*) FROM pg_locks WHERE relation = 'ser_idx'::regclass;"),
	'0', 'no locks remain after cancellation');

done_testing();
