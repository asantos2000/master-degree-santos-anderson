CREATE OR REPLACE VIEW RAW_CLASSIFY_P1_OPERATIVE_RULES_VW AS (
SELECT
	CLASS.id,
	CLASS.checkpoint,
	CLASS.doc_id,
	CLASS.statement_id,
	CLASS.statement_title,
	CLASS.statement_text,
	CLASS.statement_sources,
	CLASS.created_at,
	max(CLASS.statement_classification_type) AS statement_classification_type,
	max(CLASS.statement_classification_explanation) AS statement_classification_explanation,
	max(CLASS.statement_classification_confidence) AS statement_classification_confidence,
	EXTRACT.terms,
	EXTRACT.verb_symbols
FROM
	(
	WITH expanded AS (
	SELECT
		t.id,
		t.file_source AS checkpoint,
		t.content.doc_id,
		t.content.statement_id,
		t.content.statement_title,
		t.content.statement_text,
		t.content.statement_sources,
		t.created_at,
		c.item ->> 'type' AS statement_classification_type,
		c.item ->> 'explanation' AS statement_classification_explanation,
		(c.item ->> 'confidence')::DOUBLE AS statement_classification_confidence
	FROM
		main.RAW_CLASSIFY_P1_OPERATIVE_RULES t,
		UNNEST(t.content.classification) c(item) ),
	ranked_classifications AS (
	SELECT
		*,
		ROW_NUMBER() OVER ( PARTITION BY doc_id,
		statement_id,
		checkpoint
	ORDER BY
		statement_classification_confidence DESC ) AS rn
	FROM
		expanded )
	SELECT
		*
	FROM
		ranked_classifications
	WHERE
		rn = 1) AS CLASS
LEFT JOIN main.RAW_SECTION_EXTRACTED_ELEMENTS_VW as EXTRACT
  ON
	(CLASS.statement_id::STRING = EXTRACT.statement_id::STRING)
	AND (CLASS.doc_id = EXTRACT.doc_id)
	--AND CLASS.checkpoint = EXTRACT.checkpoint
	AND EXTRACT.checkpoint = 'documents_true_table.json'
	AND list_has_any(CLASS.statement_sources, EXTRACT.statement_sources)
	--AND (CLASS.statement_text = EXTRACT.statement_text)
GROUP BY
	CLASS.id,
	CLASS.checkpoint,
	CLASS.doc_id,
	CLASS.statement_id,
	CLASS.statement_title,
	CLASS.statement_text,
	CLASS.statement_sources,
	CLASS.created_at,
	EXTRACT.terms,
	EXTRACT.verb_symbols
UNION ALL
SELECT
	CLASS.id,
	CLASS.file_source as checkpoint,
	CLASS.content.doc_id,
	CLASS.content.statement_id,
	EXTRACT.statement_title as statement_title,
	CLASS.content.statement as statement_text,
	CLASS.content.sources as statement_sources,
	CLASS.created_at,
	CLASS.content.type as statement_classification_type,
	'From true table' as statement_classification_explanation,
	1 as statement_classification_confidence,
	EXTRACT.terms as terms,
	EXTRACT.verb_symbols as verb_symbols
FROM
	main.RAW_CLASSIFY_P1_OPERATIVE_RULES_TRUE as CLASS
LEFT JOIN main.RAW_SECTION_EXTRACTED_ELEMENTS_VW as EXTRACT
  ON
	(CLASS.content.statement_id::STRING = EXTRACT.statement_id::STRING)
	AND (CLASS.content.doc_id = EXTRACT.doc_id)
	--AND CLASS.file_source = EXTRACT.checkpoint
	AND EXTRACT.checkpoint = 'documents_true_table.json'
	AND list_has_any(CLASS.content.sources,
	EXTRACT.statement_sources)
	AND (CLASS.content.statement = EXTRACT.statement_text)
);

SELECT * FROM RAW_CLASSIFY_P1_OPERATIVE_RULES_VW;



