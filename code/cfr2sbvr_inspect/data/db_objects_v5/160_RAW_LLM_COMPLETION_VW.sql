CREATE OR REPLACE VIEW RAW_LLM_COMPLETION_VW AS (
SELECT 
id,
checkpoint,
completions_id,
completions_choices,
completions_usage.prompt_tokens,
completions_usage.total_tokens,
--completions_usage.completion_tokens_details,
--completions_usage.prompt_tokens_details,
created_at
FROM (
SELECT
id,
file_source as checkpoint,
UNNEST(completions) as completions,
completions.id as completions_id,
completions.choices as completions_choices,
completions.usage as completions_usage,
created_at
FROM main.RAW_LLM_COMPLETION
)
);

SELECT * FROM RAW_LLM_COMPLETION_VW;