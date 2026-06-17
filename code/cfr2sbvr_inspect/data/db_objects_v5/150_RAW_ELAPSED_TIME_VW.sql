CREATE OR REPLACE VIEW RAW_ELAPSED_TIME_VW AS (
SELECT
id,
file_source as checkpoint,
elapsed_times,
created_at
FROM main.RAW_ELAPSED_TIME
);

SELECT * FROM RAW_ELAPSED_TIME_VW;