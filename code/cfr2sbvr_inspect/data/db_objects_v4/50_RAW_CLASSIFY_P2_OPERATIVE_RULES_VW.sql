CREATE OR REPLACE VIEW RAW_CLASSIFY_P2_OPERATIVE_RULES_VW AS (
SELECT
	CLASS.id,
	CLASS.checkpoint,
	CLASS.doc_id,
	CLASS.statement_id,
	CLASS.statement_title,
	CLASS.statement_text,
	CLASS.statement_sources,
	CLASS.created_at,
	MAX(CLASS.statement_classification_subtype) as statement_classification_subtype,
	MAX(CLASS.statement_classification_explanation) as statement_classification_explanation,
	MAX(CLASS.statement_classification_confidence) as statement_classification_confidence,
	MAX(CLASS.statement_classification_templates_ids) as statement_classification_templates_ids,
	EXTRACT.terms,
	EXTRACT.verb_symbols
FROM
	(
	SELECT
		id,
		file_source as checkpoint,
		content.doc_id,
		content.statement_id,
		content.statement_title,
		content.statement_text,
		content.statement_sources,
		UNNEST(content.classification).subtype as statement_classification_subtype,
		UNNEST(content.classification).explanation as statement_classification_explanation,
		UNNEST(content.classification).confidence as statement_classification_confidence,
		UNNEST(content.classification).templates_ids as statement_classification_templates_ids,
		created_at
	FROM
		main.RAW_CLASSIFY_P2_OPERATIVE_RULES
) as CLASS
LEFT JOIN main.RAW_SECTION_EXTRACTED_ELEMENTS_VW as EXTRACT
  ON
	(CLASS.statement_id::STRING = EXTRACT.statement_id::STRING)
	AND (CLASS.doc_id = EXTRACT.doc_id)
	--AND CLASS.checkpoint = EXTRACT.checkpoint
	AND EXTRACT.checkpoint = 'documents_true_table.json'
	AND list_has_any(CLASS.statement_sources,
	EXTRACT.statement_sources)
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
	CLASS.content.subtype as statement_classification_subtype,
	'From true table' as statement_classification_explanation,
	1 as statement_classification_confidence,
	CLASS.content.templates_ids as statement_classification_templates_ids,
	EXTRACT.terms as terms,
	EXTRACT.verb_symbols as verb_symbols
FROM
	main.RAW_CLASSIFY_P2_OPERATIVE_RULES_TRUE CLASS
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


SELECT * FROM RAW_CLASSIFY_P2_OPERATIVE_RULES_VW;
