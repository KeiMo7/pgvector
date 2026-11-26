-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.9.0'" to load this file. \quit

CREATE FUNCTION vector_dims(sparsevec) RETURNS integer
    AS 'MODULE_PATHNAME', 'sparsevec_vector_dims' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION vector_dims(sparsevec) IS 'number of dimensions';

CREATE FUNCTION hnsw_index_info(index regclass, OUT version int, OUT dimensions int, OUT m int, OUT ef_construction int, OUT entry_level int, OUT entry_blkno bigint, OUT entry_offno int, OUT insert_page bigint, OUT pages bigint) RETURNS record
    AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION hnsw_graph_stats(index regclass, OUT level int, OUT elements bigint, OUT deleted bigint, OUT heaptids bigint, OUT fanout_min int, OUT fanout_max int, OUT empty_neighbors bigint) RETURNS SETOF record
    AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE FUNCTION hnsw_verify(index regclass, mode text DEFAULT 'structure', max_findings int DEFAULT 100, max_visited int DEFAULT 1000000, OUT severity text, OUT code text, OUT level int, OUT blkno bigint, OUT offno int, OUT detail text) RETURNS SETOF record
    AS 'MODULE_PATHNAME' LANGUAGE C;
