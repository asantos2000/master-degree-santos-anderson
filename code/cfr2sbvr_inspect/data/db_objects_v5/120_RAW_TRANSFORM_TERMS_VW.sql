CREATE OR REPLACE VIEW RAW_TRANSFORM_TERMS_VW AS (
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
EXT_P2.isLocalScope as isLocalScope,
CLASS.statement_classification_type,
CLASS.statement_classification_type_confidence,
CLASS.statement_classification_type_explanation,
CLASS.statement_classification_subtype,
CLASS.statement_classification_subtype_confidence,
CLASS.statement_classification_subtype_explanation,
CLASS.terms,
CLASS.verb_symbols,
TRANSF.created_at 
FROM main.RAW_TRANSFORM_TERMS as TRANSF
LEFT JOIN main.RAW_CLASSIFY_VW as CLASS
  ON
	(TRANSF.content.statement_id::STRING = CLASS.statement_id::STRING)
	AND (CLASS.source = 'Terms')
	AND (TRANSF.content.doc_id = CLASS.doc_id)
	AND TRANSF.file_source = CLASS.checkpoint
	--AND CLASS.checkpoint = 'documents_true_table.json'
	AND list_has_any(TRANSF.content.statement_sources, CLASS.statement_sources)
	AND (TRANSF.content.statement = CLASS.statement_text)
LEFT JOIN main.RAW_SECTION_P2_EXTRACTED_NOUN_VW as EXT_P2
  ON
	(TRANSF.content.statement_id::STRING = EXT_P2.statement_title::STRING)
	AND (CLASS.source = 'Terms')
	AND (TRANSF.content.doc_id = EXT_P2.doc_id)
	AND TRANSF.file_source = CLASS.checkpoint
	--AND EXT_P2.checkpoint = 'documents_true_table.json'
	--AND list_has_any(TRANSF.content.statement_sources, EXT_P2.statement_sources)
	AND (TRANSF.content.statement = EXT_P2.statement_text)
);

SELECT * FROM RAW_TRANSFORM_TERMS_VW;