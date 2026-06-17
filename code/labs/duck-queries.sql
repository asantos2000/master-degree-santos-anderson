D describe select id, prompt, file_source, unnest(elements), created_at from RAW_SECTION_P1_EXTRACTED_ELEMENTS;
┌──────────────────────┬─────────────────────────┬─────────┬─────────┬─────────┬─────────┐
│     column_name      │       column_type       │  null   │   key   │ default │  extra  │
│       varchar        │         varchar         │ varchar │ varchar │ varchar │ varchar │
├──────────────────────┼─────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ id                   │ VARCHAR                 │ YES     │         │         │         │
│ prompt               │ VARCHAR                 │ YES     │         │         │         │
│ file_source          │ VARCHAR                 │ YES     │         │         │         │
│ id                   │ BIGINT                  │ YES     │         │         │         │
│ title                │ VARCHAR                 │ YES     │         │         │         │
│ statement            │ VARCHAR                 │ YES     │         │         │         │
│ terms                │ STRUCT(term VARCHAR, …  │ YES     │         │         │         │
│ verb_symbols         │ VARCHAR[]               │ YES     │         │         │         │
│ verb_symbols_extra…  │ DOUBLE[]                │ YES     │         │         │         │ verb_symbols_extracted_confidence
│ verb_symbols_extra…  │ VARCHAR[]               │ YES     │         │         │         │ verb_symbols_extracted_reason
│ classification       │ VARCHAR                 │ YES     │         │         │         │
│ confidence           │ DOUBLE                  │ YES     │         │         │         │
│ reason               │ VARCHAR                 │ YES     │         │         │         │
│ sources              │ VARCHAR[]               │ YES     │         │         │         │
│ created_at           │ TIMESTAMP WITH TIME Z…  │ YES     │         │         │         │
├──────────────────────┴─────────────────────────┴─────────┴─────────┴─────────┴─────────┤
│ 15 rows                                                                      6 columns │
└────────────────────────────────────────────────────────────────────────────────────────┘
D describe select id, prompt, file_source, unnest(elements), created_at from RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE;
┌────────────────┬───────────────────────────────┬─────────┬─────────┬─────────┬─────────┐
│  column_name   │          column_type          │  null   │   key   │ default │  extra  │
│    varchar     │            varchar            │ varchar │ varchar │ varchar │ varchar │
├────────────────┼───────────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ id             │ VARCHAR                       │ YES     │         │         │         │
│ prompt         │ VARCHAR                       │ YES     │         │         │         │
│ file_source    │ VARCHAR                       │ YES     │         │         │         │
│ id             │ BIGINT                        │ YES     │         │         │         │
│ title          │ VARCHAR                       │ YES     │         │         │         │
│ statement      │ VARCHAR                       │ YES     │         │         │         │
│ terms          │ STRUCT(term VARCHAR, classi…  │ YES     │         │         │         │
│ verb_symbols   │ VARCHAR[]                     │ YES     │         │         │         │
│ classification │ VARCHAR                       │ YES     │         │         │         │
│ sources        │ VARCHAR[]                     │ YES     │         │         │         │
│ created_at     │ TIMESTAMP WITH TIME ZONE      │ YES     │         │         │         │
├────────────────┴───────────────────────────────┴─────────┴─────────┴─────────┴─────────┤
│ 11 rows                                                                      6 columns │
└────────────────────────────────────────────────────────────────────────────────────────┘
D 

WITH P1_EXTRACTED_ELEMENTS
AS (
SELECT id,
prompt,
file_source,
unnest(elements) as element,
created_at
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS
),
P1_EXTRACTED_ELEMENTS_TRUE AS (
SELECT id,
prompt,
file_source,
unnest(elements) as element,
created_at
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE
)
select 
id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
terms,
verb_symbols,
verb_symbols_extracted_confidence,
verb_symbols_extracted_reason,
classification,
confidence,
reason,
sources as statement_sources,
created_at
from P1_EXTRACTED_ELEMENTS
UNION
select
id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
[] as terms,
verb_symbols,
[0::DOUBLE] as verb_symbols_extracted_confidence,
[NULL] as verb_symbols_extracted_reason,
classification,
0::DOUBLE as confidence,
'' as reason,
sources as statement_sources,
created_at
from P1_EXTRACTED_ELEMENTS_TRUE
;

-- Statements
WITH P1_EXTRACTED_ELEMENTS
AS (
SELECT id,
prompt,
file_source,
unnest(elements) as element,
created_at
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS
),
P1_EXTRACTED_ELEMENTS_TRUE AS (
SELECT id,
prompt,
file_source,
unnest(elements) as element,
created_at
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE
)
select 
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
classification,
confidence,
reason,
sources as statement_sources,
created_at
from P1_EXTRACTED_ELEMENTS
UNION
select
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
classification,
0::DOUBLE as confidence,
'' as reason,
sources as statement_sources,
created_at
from P1_EXTRACTED_ELEMENTS_TRUE
;

-- Terms
WITH P1_EXTRACTED_ELEMENTS
AS (
SELECT id,
prompt,
file_source,
unnest(elements) as element,
created_at
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS
),
P1_EXTRACTED_ELEMENTS_TRUE AS (
SELECT id,
prompt,
file_source,
unnest(elements) as element,
created_at
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE
)
select 
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
classification,
confidence,
reason,
sources as statement_sources,
created_at
from P1_EXTRACTED_ELEMENTS
UNION
select
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
classification,
0::DOUBLE as confidence,
'' as reason,
sources as statement_sources,
created_at
from P1_EXTRACTED_ELEMENTS_TRUE
;

-- Terms
select 
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
unnest(terms) as term,
verb_symbols,
classification,
sources as statement_sources,
created_at
from (
SELECT id, prompt, file_source, UNNEST(elements) as element, created_at 
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE
)

WITH P1_EXTRACTED_ELEMENTS_TRUE AS 
( 
    SELECT id, 
    prompt, 
    file_source, 
    unnest(elements) AS element, 
    created_at 
    FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE 
) 
SELECT id, 
( SELECT list_aggregate( ARRAY( SELECT struct_pack( term := x.term, classification := x.classification, confidence := 0.0, reason := NULL, extracted_confidence := 0.0, extracted_reason := NULL ) FROM UNNEST(element.terms) x ) ) ) AS terms 
FROM P1_EXTRACTED_ELEMENTS_TRUE;

WITH P1_EXTRACTED_ELEMENTS_TRUE AS 
( 
    SELECT id, 
    prompt, 
    file_source, 
    unnest(elements) AS element, 
    created_at 
    FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE 
) 
P1_EXTRACTED_ELEMENTS_TRUE_TERM (
SELECT id,
(select term.term, term.classification from unnest(terms) limit 1) as term
--( SELECT list_aggregate( ARRAY( SELECT struct_pack( term := term, classification := classification, confidence := 0.0, reason := NULL, extracted_confidence := 0.0, extracted_reason := NULL ) FROM UNNEST(terms) ) ) )
FROM P1_EXTRACTED_ELEMENTS_TRUE;

CREATE OR REPLACE VIEW RAW_SECTION_P1_EXTRACTED_ELEMENTS_VW AS
(
select
doc_id,
prompt,
checkpoint,
statement_id,
statement_title,
statement_text,
--term,
--term_classification,
array_agg(struct_pack( term := term, classification := term_classification, confidence := 0.0, reason := NULL, extracted_confidence := 0.0, extracted_reason := NULL )) as terms,
verb_symbols,
classification,
statement_sources,
created_at
from (
    select 
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
unnest(terms).term as term,
unnest(terms).classification as term_classification,
verb_symbols,
classification,
sources as statement_sources,
created_at
from (
SELECT id, prompt, file_source, UNNEST(elements) as element, created_at 
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE
)
)
GROUP BY 
doc_id,
prompt,
checkpoint,
statement_id,
statement_title,
statement_text,
verb_symbols,
classification,
statement_sources,
created_at
UNION
select
id.replace('_P1', '').replace('_P2', '') as doc_id,
prompt,
file_source as checkpoint,
id_1 as statement_id,
title as statement_title,
statement as statement_text,
terms,
verb_symbols,
classification,
sources as statement_sources,
created_at
FROM (
SELECT id, prompt, file_source, UNNEST(elements) as element, created_at 
FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS
)
);