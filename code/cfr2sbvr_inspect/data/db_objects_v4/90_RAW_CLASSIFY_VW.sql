CREATE OR REPLACE VIEW RAW_CLASSIFY_VW AS (
SELECT
	CLASS_P2.id,
	CLASS_P2.checkpoint,
	CLASS_P2.doc_id,
	CLASS_P2.statement_id,
	CLASS_P2.statement_title,
	CLASS_P2.statement_text,
	CLASS_P2.statement_sources,
	CLASS_P2.created_at,
	'Definitional rules' as statement_classification_type,
	'By the book' as statement_classification_type_explanation,
	1 as statement_classification_type_confidence,
	CLASS_P2.statement_classification_subtype,
	CLASS_P2.statement_classification_explanation as statement_classification_subtype_explanation,
	CLASS_P2.statement_classification_confidence as statement_classification_subtype_confidence,
	CLASS_P2.statement_classification_templates_ids as transformation_template_ids,
	[] as terms,
	[] as verb_symbols,
	'Terms' as source
FROM
	main.RAW_CLASSIFY_P2_DEFINITIONAL_TERMS_VW as CLASS_P2
UNION ALL
SELECT
	CLASS_P2.id,
	CLASS_P2.checkpoint,
	CLASS_P2.doc_id,
	CLASS_P2.statement_id,
	CLASS_P2.statement_title,
	CLASS_P2.statement_text,
	CLASS_P2.statement_sources,
	CLASS_P2.created_at,
	'Definitional rules' as statement_classification_type,
	'By the book' as statement_classification_type_explanation,
	1 as statement_classification_type_confidence,
	CLASS_P2.statement_classification_subtype,
	CLASS_P2.statement_classification_explanation as statement_classification_subtype_explanation,
	CLASS_P2.statement_classification_confidence as statement_classification_subtype_confidence,
	CLASS_P2.statement_classification_templates_ids as transformation_template_ids,
	[] as terms,
	[] as verb_symbols,
	'Names' as source
FROM
	main.RAW_CLASSIFY_P2_DEFINITIONAL_NAMES_VW as CLASS_P2
UNION ALL
SELECT
	CLASS_P2.id,
	CLASS_P2.checkpoint,
	CLASS_P2.doc_id,
	CLASS_P2.statement_id,
	CLASS_P2.statement_title,
	CLASS_P2.statement_text,
	CLASS_P2.statement_sources,
	CLASS_P2.created_at,
	'Definitional rules' as statement_classification_type,
	'By the book' as statement_classification_type_explanation,
	1 as statement_classification_type_confidence,
	CLASS_P2.statement_classification_subtype,
	CLASS_P2.statement_classification_explanation as statement_classification_subtype_explanation,
	CLASS_P2.statement_classification_confidence as statement_classification_subtype_confidence,
	CLASS_P2.statement_classification_templates_ids as transformation_template_ids,
	CLASS_P2.terms,
	CLASS_P2.verb_symbols,
	'Fact_Types' as source
FROM
	main.RAW_CLASSIFY_P2_DEFINITIONAL_FACTS_VW as CLASS_P2
UNION ALL
SELECT
	CLASS_P2.id,
	CLASS_P2.checkpoint,
	CLASS_P2.doc_id,
	CLASS_P2.statement_id,
	CLASS_P2.statement_title,
	CLASS_P2.statement_text,
	CLASS_P2.statement_sources,
	CLASS_P2.created_at,
	CLASS_P1.statement_classification_type,
	CLASS_P1.statement_classification_explanation as statement_classification_type_explanation,
	CLASS_P1.statement_classification_confidence as statement_classification_type_confidence,
	CLASS_P2.statement_classification_subtype,
	CLASS_P2.statement_classification_explanation as statement_classification_subtype_explanation,
	CLASS_P2.statement_classification_confidence as statement_classification_subtype_confidence,
	CLASS_P2.statement_classification_templates_ids as transformation_template_ids,
	CLASS_P2.terms,
	CLASS_P2.verb_symbols,
	'Operative_Rules' as source
FROM
	main.RAW_CLASSIFY_P2_OPERATIVE_RULES_VW as CLASS_P2
LEFT JOIN main.RAW_CLASSIFY_P1_OPERATIVE_RULES_VW as CLASS_P1
  ON (CLASS_P2.statement_id = CLASS_P1.statement_id)
	 AND (CLASS_P2.checkpoint = CLASS_P1.checkpoint)
 	 AND list_has_any(CLASS_P2.statement_sources, CLASS_P1.statement_sources)
 	 AND (CLASS_P2.statement_text = CLASS_P1.statement_text)
);

SELECT * FROM RAW_CLASSIFY_VW;
