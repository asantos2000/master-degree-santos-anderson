CREATE OR REPLACE VIEW RAW_SECTION_P2_EXTRACTED_NOUN_VW as (
SELECT
	'section_P2' as id,
	doc_id,
	checkpoint,
	statement_title,
	statement_text,
	confidence,
	reason,
	isLocalScope,
	local_scope_confidence,
	local_scope_reason,
	created_at
FROM
	(
	SELECT
		id.REPLACE('_P1',
		'').REPLACE('_P2',
		'') as doc_id,
		prompt,
		file_source as checkpoint,
		terms.term as statement_title,
		terms.definition as statement_text,
		terms.confidence as confidence,
		terms.reason as reason,
		terms.isLocalScope as isLocalScope,
		terms.local_scope_confidence as local_scope_confidence,
		terms.local_scope_reason as local_scope_reason,
		created_at
	FROM
		main.RAW_SECTION_P2_EXTRACTED_NOUN rspen 
)
UNION
SELECT
	'section_P2' as id,
	doc_id,
	checkpoint,
	statement_title,
	statement_text,
	confidence,
	reason,
	isLocalScope,
	local_scope_confidence,
	local_scope_reason,
	created_at
FROM
	(
	SELECT
		id.REPLACE('_P1',
		'').REPLACE('_P2',
		'') as doc_id,
		file_source as checkpoint,
		terms.term as statement_title,
		terms.definition as statement_text,
		1 as confidence,
		'From true table' as reason,
		terms.isLocalScope,
		1 as local_scope_confidence,
		'From true table' as local_scope_reason,
		created_at
	FROM
		main.RAW_SECTION_P2_EXTRACTED_NOUN_TRUE rspen 
)
);

SELECT * FROM RAW_SECTION_P2_EXTRACTED_NOUN_VW;