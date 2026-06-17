CREATE OR REPLACE VIEW RAW_LLM_VALIDATION_BEST_VW AS (
SELECT *
FROM
	(
	SELECT
		*,
		ROW_NUMBER() OVER ( PARTITION BY source,
		doc_id,
		statement_sources
	ORDER BY
		semscore DESC,
		similarity_score DESC ) AS rn
	FROM
		RAW_LLM_VALIDATION_VW 
	WHERE transformed is not NULL) t
WHERE
	rn = 1
ORDER BY
	source,
	doc_id,
	statement_sources
);

SELECT * FROM main.RAW_LLM_VALIDATION_BEST_VW;