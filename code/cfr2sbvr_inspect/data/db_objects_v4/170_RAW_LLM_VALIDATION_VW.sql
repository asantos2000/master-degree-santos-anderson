CREATE OR REPLACE VIEW RAW_LLM_VALIDATION_VW AS (
SELECT
	VAL.id,
	VAL.file_source as checkpoint,
	VAL.content.doc_id,
	VAL.content.statement_id,
	TRANSF.statement_title,
	VAL.content.statement as statement_text,
	VAL.content.sources as statement_sources,
	VAL.content.semscore,
	VAL.content.similarity_score,
	VAL.content.similarity_score_confidence,
	VAL.content.transformation_accuracy,
	VAL.content.grammar_syntax_accuracy,
	VAL.content.findings,
	VAL.created_at,
	TRANSF.transformed,
	TRANSF.transformation_template_ids,
	TRANSF.transformation_confidence,
	TRANSF.transformation_reason,
	-- CLASS
	TRANSF.statement_classification_type,
	TRANSF.statement_classification_type_confidence,
	TRANSF.statement_classification_type_explanation,
	TRANSF.statement_classification_subtype,
	TRANSF.statement_classification_subtype_confidence,
	TRANSF.statement_classification_subtype_explanation,
	TRANSF.isLocalScope,
	-- EXTRACT
	TRANSF.terms,
	TRANSF.verb_symbols,
	VAL.id.REPLACE('validation_judge_', '') as source,
	IF (TRANSF.statement_text = VAL.content.statement, TRUE, FALSE) as dbl_check
FROM
	main.RAW_LLM_VALIDATION as VAL
LEFT JOIN main.RAW_TRANSFORM_ELEMENTS_VW as TRANSF
  ON
	(VAL.content.statement_id::STRING = TRANSF.statement_id::STRING)
	AND (VAL.id.REPLACE('validation_judge_', '') = TRANSF.source)
	AND (VAL.content.doc_id = TRANSF.doc_id)
	AND (VAL.file_source = TRANSF.checkpoint)
	AND list_has_any(VAL.content.sources, TRANSF.statement_sources)
	AND (VAL.content.statement = TRANSF.statement_text)
);

SELECT * FROM RAW_LLM_VALIDATION_VW;