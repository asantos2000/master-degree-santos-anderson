CREATE OR REPLACE VIEW RAW_SECTION_P1_EXTRACTED_ELEMENTS_VW AS (
SELECT
    id,
    doc_id,
    checkpoint,
    statement_id,
    statement_title,
    statement_text,
    array_agg(struct_pack(term := term,
    classification := term_classification,
    confidence := 0.0,
    reason := NULL,
    extracted_confidence := 0.0,
    extracted_reason := NULL)) AS terms,
    verb_symbols,
    classification AS statement_classification_type,
    statement_sources,
    created_at
FROM
    (
    SELECT
        id,
        "replace"(id."replace"('_P1',
        ''),
        '_P2',
        '') AS doc_id,
        prompt,
        file_source AS checkpoint,
        id_1 AS statement_id,
        title AS statement_title,
        statement AS statement_text,
        (unnest(terms)).term AS term,
        (unnest(terms)).classification AS term_classification,
        verb_symbols,
        classification,
        sources AS statement_sources,
        created_at
    FROM (
	        SELECT
	            id,
	            prompt,
	            file_source,
	            unnest(elements) AS ELEMENT,
	            created_at
	        FROM
	            RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE
        )
    )
GROUP BY
    id,
    doc_id,
    checkpoint,
    statement_id,
    statement_title,
    statement_text,
    verb_symbols,
    classification,
    statement_sources,
    created_at
UNION ALL
	SELECT
	id,
	"replace"(id."replace"('_P1',
	''),
	'_P2',
	'') AS doc_id,
	file_source AS checkpoint,
	id_1 AS statement_id,
	title AS statement_title,
	statement AS statement_text,
	terms,
	verb_symbols,
	classification AS statement_classification_type,
	sources AS statement_sources,
	created_at
	FROM
	(
	SELECT
	    id,
	    prompt,
	    file_source,
	    unnest(elements) AS ELEMENT,
	    created_at
	FROM
	    RAW_SECTION_P1_EXTRACTED_ELEMENTS)
);

SELECT * FROM RAW_SECTION_P1_EXTRACTED_ELEMENTS_VW;