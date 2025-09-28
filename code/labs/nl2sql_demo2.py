# %% [markdown]
# # NLP with Disaster Tweets: Advanced Cleaning with FlockMTL
# 
# This notebook demonstrates advanced text processing of disaster-related tweets using the **FlockMTL** extension for DuckDB. The pipeline includes cleaning, feature extraction, and data visualization.

# %%
import duckdb
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# %% [markdown]
# ## Set Up DuckDB and Load FlockMTL
# 
# Initialize an in-memory DuckDB connection and load the FlockMTL extension.

# %%
# Initialize DuckDB
con = duckdb.connect(':memory:')

# Load FlockMTL extension
con.sql("INSTALL flockmtl FROM community")
con.sql("LOAD flockmtl")

# %% [markdown]
# ## Define Model and API Key for OpenAI Integration
# Set up Model and the OpenAI API key to enable advanced NLP processing using FlockMTL.

#%%
# Create model
con.sql("""
CREATE MODEL(
   'CFR2SBVR_GPT4O',
   'gpt-4o', 
   'openai', 
   {"model_parameters": {"temperature": 0.1}}
);
""")

# %%
# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

con.sql(f"""
CREATE SECRET (
    TYPE OPENAI,
    API_KEY '{openai_api_key}'
); 
""")

#%% [markdown]
# ## Test LLM Completion
# Source: https://dais-polymtl.github.io/flockmtl/docs/scalar-functions/llm-complete

#%%
con.sql("""
SELECT llm_complete(
           {'model_name': 'CFR2SBVR_GPT4O'},
    {'prompt': 'Talk like a duck 🦆 and explain what a database is 📚'}
) AS greetings;
""")

# %% [markdown]
# ## Extract terms from transformed tables

# %%
# Define terms list prompt
con.execute("""
CREATE PROMPT ('terms-list', '
Transform tables has all elements transformed (terms, names, operative rules, and fact types)
The content.statement_name has the term name, and file_source has the checkpoint name.
Extract terms per checkpoint?
Output JSON with:
{
    "terms": ["term1", "term2", ...]},
}
If nothing is found, return "[]".
');
""")

#%%
# Process terms extraction
query = """
CREATE TABLE terms AS
SELECT 
    file_source AS checkpoint_name,
    llm_complete(
        {'model_name': 'CFR2SBVR_GPT4O'},
        {'prompt_name': 'terms-list'},
        {'content': content.statement_title}
    ) AS terms_json
FROM RAW_TRANSFORM_TERMS;
"""
con.execute(query)

#%% [markdown]
# ## View Extracted Terms

#%%
con.execute("SELECT * FROM terms").df()

