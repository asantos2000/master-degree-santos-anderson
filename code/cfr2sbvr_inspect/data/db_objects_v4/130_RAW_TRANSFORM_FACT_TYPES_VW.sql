CREATE OR REPLACE VIEW RAW_TRANSFORM_FACT_TYPES_VW AS (
SELECT
	TRANSF.id,
	TRANSF.file_source as checkpoint,
	TRANSF.content.doc_id,
	TRANSF.content.statement_id,
	TRANSF.content.statement_title,
	TRANSF.content.statement as statement_text,
	TRANSF.content.statement_sources,
	TRANSF.content.templates_ids as transformation_template_ids,
	TRANSF.content.transformed,
	TRANSF.content.confidence as transformation_confidence,
	TRANSF.content.reason as transformation_reason,
	TRUE as isLocalScope,
	CLASS.statement_classification_type,
	CLASS.statement_classification_type_confidence,
	CLASS.statement_classification_type_explanation,
	CLASS.statement_classification_subtype,
	CLASS.statement_classification_subtype_confidence,
	CLASS.statement_classification_subtype_explanation,
	CLASS.terms,
	CLASS.verb_symbols,
	TRANSF.created_at
FROM
	main.RAW_TRANSFORM_FACT_TYPES AS TRANSF
LEFT JOIN main.RAW_CLASSIFY_VW as CLASS
  ON
	(TRANSF.content.statement_id::STRING = CLASS.statement_id::STRING)
	AND (CLASS.source = 'Fact_Types')
	AND (TRANSF.content.doc_id = CLASS.doc_id)
	--AND TRANSF.file_source = CLASS.checkpoint
	AND CLASS.checkpoint = 'documents_true_table.json'
	AND list_has_any(TRANSF.content.statement_sources, CLASS.statement_sources)
	AND (TRANSF.content.statement = CLASS.statement_text)
);

SELECT * FROM RAW_TRANSFORM_FACT_TYPES_VW;