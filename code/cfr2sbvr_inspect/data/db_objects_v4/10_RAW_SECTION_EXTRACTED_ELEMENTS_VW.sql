CREATE OR REPLACE VIEW RAW_SECTION_EXTRACTED_ELEMENTS_VW AS (
-- True
WITH P1 AS (
SELECT
	id,
	doc_id,
	prompt,
	file_source AS "checkpoint",
	id_1 AS statement_id,
	title AS statement_title,
	"statement" AS statement_text,
	(UNNEST(terms)).term AS term,
	(UNNEST(terms)).classification AS term_classification,
	verb_symbols,
	classification,
	sources AS statement_sources,
	created_at
FROM
	(
	SELECT
		id,
		id."replace"('_P1',
		'') AS doc_id,
		prompt,
		file_source,
		UNNEST(elements) AS ELEMENT,
		created_at
	FROM
		RAW_SECTION_P1_EXTRACTED_ELEMENTS_TRUE))
SELECT
	P1.id,
	P1.doc_id,
	P1."checkpoint",
	P1.statement_id,
	P1.statement_title,
	P1.statement_text,
	P1.verb_symbols,
	P1.classification AS statement_classification_type,
	CAST(1.00 AS DOUBLE) AS statement_classification_type_confidence,
	CAST('' AS VARCHAR) AS statement_classification_type_explanation,
	P1.statement_sources,
	P1.created_at,
	array_agg(
		struct_pack(
			term := P1.term,
			classification := term_classification,
			isLocalScope := term_isLocalScope,
			definition := term_definition,
			confidence := 0.0,
			reason := NULL,
			isLocalScope_confidence := 0.0,
			isLocalScope_reason := NULL,
			templates_ids := [],
			transformed := NULL,
			transform_confidence := 0.0,
			transform_reason := NULL,
			from_checkpoint := NULL
		)
	) AS terms,
	P1.classification."replace"(' ', '_') AS SOURCE
FROM P1
LEFT JOIN (
	SELECT
		id,
		id."replace"('_P2',
		'') AS doc_id,
		prompt,
		file_source AS "checkpoint",
		terms.term AS term,
		terms.definition AS term_definition,
		terms.isLocalScope AS term_isLocalScope,
		created_at
	FROM RAW_SECTION_P2_EXTRACTED_NOUN_TRUE) AS P2 
ON (((P1.doc_id = P2.doc_id)
	AND (P1."checkpoint" = P2."checkpoint")
		AND (P1.term = P2.term)))
GROUP BY
	P1.id,
	P1.doc_id,
	P1."checkpoint",
	P1.statement_id,
	P1.statement_title,
	P1.statement_text,
	P1.verb_symbols,
	P1.classification,
	P1.statement_sources,
	P1.created_at
UNION ALL
-- PREDICT
(WITH P1 AS (
	SELECT
		id,
		doc_id,
		prompt,
		file_source AS "checkpoint",
		id_1 AS statement_id,
		title AS statement_title,
		"statement" AS statement_text,
		(UNNEST(terms)).term AS term,
		(UNNEST(terms)).classification AS term_classification,
		verb_symbols,
		classification,
		confidence,
		reason,
		sources AS statement_sources,
		created_at
	FROM (
			SELECT
				id,
				id."replace"('_P1',
				'') AS doc_id,
				prompt,
				file_source,
				UNNEST(elements) AS ELEMENT,
				created_at
			FROM
				RAW_SECTION_P1_EXTRACTED_ELEMENTS
	)
)
SELECT
	P1.id,
	P1.doc_id,
	P1."checkpoint",
	P1.statement_id,
	P1.statement_title,
	P1.statement_text,
	P1.verb_symbols,
	P1.classification AS statement_classification_type,
	P1.confidence AS statement_classification_type_confidence,
	P1.reason AS statement_classification_type_explanation,
	P1.statement_sources,
	P1.created_at,
	array_agg(
		struct_pack(
			term := P1.term,
			classification := term_classification,
			isLocalScope := term_isLocalScope,
			definition := term_definition,
			confidence := term_confidence,
			reason := term_reason,
			isLocalScope_confidence := isLocalScope_confidence,
			isLocalScope_reason := isLocalScope_reason,
			templates_ids := COALESCE(NAMES.content.templates_ids, TERMS.content.templates_ids, []),
			transformed := COALESCE(NAMES.content.transformed, TERMS.content.transformed, NULL),
			transform_confidence := COALESCE(NAMES.content.confidence, TERMS.content.confidence, 0.0),
			transform_reason := COALESCE(NAMES.content.reason, TERMS.content.reason, NULL),
			from_checkpoint := COALESCE(NAMES.file_source, TERMS.file_source, NULL)
    	)
    ) AS terms,
	P1.classification."replace"(' ', '_') AS SOURCE
FROM P1
LEFT JOIN (
	SELECT
		id,
		id."replace"('_P2',
		'') AS doc_id,
		prompt,
		file_source AS "checkpoint",
		terms.term AS term,
		terms.definition AS term_definition,
		terms.isLocalScope AS term_isLocalScope,
		terms.confidence AS term_confidence,
		terms.reason AS term_reason,
		terms.local_scope_confidence AS isLocalScope_confidence,
		terms.local_scope_reason AS isLocalScope_reason,
		created_at
	FROM RAW_SECTION_P2_EXTRACTED_NOUN
) AS P2 
ON ((P1.doc_id = P2.doc_id)
	AND (P1."checkpoint" = P2."checkpoint")
		AND (P1.term = P2.term))
LEFT JOIN RAW_TRANSFORM_NAMES AS NAMES
ON (P1.doc_id = NAMES.content.doc_id
		--AND P1.checkpoint = NAMES.file_source
		AND P2.term::VARCHAR = NAMES.content.statement_id::VARCHAR
		AND list_has_any(P1.statement_sources, NAMES.content.statement_sources)
)
LEFT JOIN RAW_TRANSFORM_TERMS AS TERMS
ON ( P1.doc_id = TERMS.content.doc_id
		--AND P1.checkpoint = TERMS.file_source
		AND P2.term::VARCHAR = TERMS.content.statement_id::VARCHAR
		AND list_has_any(P1.statement_sources, TERMS.content.statement_sources)
)
GROUP BY
	P1.id,
	P1.doc_id,
	P1."checkpoint",
	P1.statement_id,
	P1.statement_title,
	P1.statement_text,
	P1.verb_symbols,
	P1.classification,
	P1.confidence,
	P1.reason,
	P1.statement_sources,
	P1.created_at)
);