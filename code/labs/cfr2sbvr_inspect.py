import re
from pathlib import Path
import html

import duckdb
import jellyfish
import numpy as np
import pandas as pd
import streamlit as st

# only for labs
import sys
sys.path.append(r'../src')

import rules_taxonomy_provider.main as rules_taxonomy_provider
from rules_taxonomy_provider.main import RuleInformationProvider


def disconnect_db():
    st.write("called")
    # conn.close()


# @st.cache_data
def load_data(table, checkpoints, doc_ids, process_selected):
    where_clause = ""

    # where_clause += f' AND file_source=\'{filters["checkpoint"]}\''
    checkpoints_string = ", ".join(f"'{item}'" for item in checkpoints)
    doc_ids_string = ", ".join(f"'{item}'" for item in doc_ids)
    if checkpoints_string:
        where_clause += f" AND checkpoint in ({checkpoints_string})"

    if doc_ids_string:
        where_clause += f" AND doc_id in ({doc_ids_string})"

    data_query = f"""
    SELECT *
    FROM {table}
    WHERE 1 = 1
    {where_clause}
    ORDER BY *
    ;
    """

    # print(data_query)
    st.write(f"Content from: {table}")
    return conn.sql(query=data_query).fetchdf()


def calculate_statements_similarity(statement1, statement2):
    # Calculate the similarity score using the Levenshtein distance
    score = jellyfish.levenshtein_distance(statement1.lower(), statement2.lower())
    similarity_score = 1 - (
        score / max(len(statement1), len(statement2))
    )  # Normalize to a similarity score
    return similarity_score


def highlight_statement(
    prefix,
    doc_id,
    element_id,
    classification_type,
    classification_subtype,
    terms,
    verb_symbols,
    statement,
    sources,
):
    sources_links = []
    for source in sources:
        doc_id_url = doc_id.replace("§ ", "")
        url = f"https://www.ecfr.gov/current/title-17/part-275#p-{doc_id_url}{source}"
        sources_links.append(f'<a href="{url}">{source}</a>')
    sources = ", ".join(sources_links)

    classification = classification_type
    if classification_subtype:
        classification += f" | {classification_subtype}"

    # Clean up the statement
    statement = statement.replace("$", "\\$")

    # Highlight term in the statement
    def highlight_match_term(term_info):
        def replace_term(match):
            original = match.group(0)
            if term_info["classification"] == "Common Noun":
                # Aqui apenas inserimos um placeholder. Depois criaremos o tooltip.
                return (
                    f'<span style="text-decoration: underline; '
                    f'text-decoration-color: green;">{original}</span>'
                )
            elif term_info["classification"] == "Proper Noun":
                return (
                    f'<span style="text-decoration: underline double; '
                    f'text-decoration-color: green;">{original}</span>'
                )
            return original
        return replace_term

    # Apply term substitutions (without inserting the tooltip yet).
    for t in terms:
        term_regex = rf"\b{re.escape(t['term'])}\b"
        statement = re.sub(term_regex,
                           highlight_match_term(t),
                           statement,
                           flags=re.IGNORECASE)

    # Highlight verb symbols
    def highlight_match_verb(match):
        original = match.group(0)
        return f'<span style="font-style: italic; color: blue;">{original}</span>'

    for verb in verb_symbols:
        verb_regex = rf"\b{re.escape(verb)}\b"
        statement = re.sub(verb_regex, highlight_match_verb, statement, flags=re.IGNORECASE)

    def add_tooltip(term_info):
        definition = html.escape(term_info.get("definition", "") or "Missing")
        confidence = term_info.get("confidence", "")
        reason = html.escape(term_info.get("reason", "") or "")
        isLocalScope = term_info.get("isLocalScope", False)
        if isLocalScope:
            scope = "📍"
        else:
            scope = "🌎"

        tooltip_content = (
            f"{scope} "
            f"Definition: {definition} - "
            f"Confidence: {confidence} - "
            f"Reason: {reason}"
        )

        # Regext to capture the tag generated above (simple or double underline) with the corresponding text.
        if term_info["classification"] == "Common Noun":
            # Simple underline
            tag_pattern = (
                r'<span style="text-decoration: underline; text-decoration-color: green;">'
                rf"(?P<content>{re.escape(term_info['term'])}|{re.escape(term_info['term'].lower())}|{re.escape(term_info['term'].capitalize())})"
                r"</span>"
            )
        else:
            # Double underline
            tag_pattern = (
                r'<span style="text-decoration: underline double; text-decoration-color: green;">'
                rf"(?P<content>{re.escape(term_info['term'])}|{re.escape(term_info['term'].lower())}|{re.escape(term_info['term'].capitalize())})"
                r"</span>"
            )

        def insert_title(match):
            original_span = match.group(0)
            content = match.group("content")
            # Insert the title attribute without changing the inner text
            return original_span.replace(
                ';">' + content,  # ponto de inserção
                f';" title="{tooltip_content}">' + content
            )

        return tag_pattern, insert_title

    for t in terms:
        tag_pattern, insert_title_fn = add_tooltip(t)
        statement = re.sub(tag_pattern, insert_title_fn, statement, flags=re.IGNORECASE)

    sup = f" [{classification}]"
    final_text = f"{prefix}{element_id}: <strong>{sources}</strong> {statement}{sup}"
    return final_text



def display_section(doc_id):
    doc_id_url = doc_id.replace("§ ", "")
    section_url = f"https://www.ecfr.gov/current/title-17/section-{doc_id_url}"

    content = conn.sql(
        f"SELECT content FROM RAW_SECTION WHERE id = '{doc_id}'"
    ).fetchall()[0][0]
    content = content.replace("\n", "<br>")
    content = content.replace("$", "\$")
    content = content.replace(doc_id, f'<a href="{section_url}">{doc_id}</a>')

    st.write(content, unsafe_allow_html=True)

@st.dialog("Witt (2012) taxonomy", width="large")
def witt_taxonomy_dialog(classification): 
    rule_provider = RuleInformationProvider("../data")
    markdown_data = rule_provider.get_classification_and_templates(classification, return_forms="fact_type")
    st.markdown(markdown_data)

def list_to_markdown(list, ordered=True):
    if ordered:
       prefix = "1."
    else:
         prefix = "-"
    return "\n".join([f"{prefix} {item}" for item in list])

#
# Main
#

st.set_page_config(page_title="CFR2SBVR Inspect", page_icon="🏛️", layout="wide")

st.title("CFR2SBVR Inspect")

st.sidebar.title("Checkpoints")

# Connect to the database
conn = duckdb.connect("../data/cfr2sbvr_db/database_v3.db", read_only=True)
# conn.close()

# Sidebar selectbox to choose a process
process_dict = {
    "Extraction": "extraction",
    "Classification": "classification",
    "Transformation": "transformation",
    "Validation": "validation",
}

process_selected = st.sidebar.selectbox("Choose a process", process_dict.keys())

doc_ids = conn.sql(
    "select distinct id.replace('_P1', '') as doc_id from RAW_SECTION_P1_EXTRACTED_ELEMENTS order by id"
)

# Sidebar selectbox to choose a file
doc_id_selected = st.sidebar.multiselect("Choose a doc_id", doc_ids)

query = f"""
SELECT DISTINCT TABLE_NAME,
FROM CHECKPOINT_METADATA
WHERE doc_source in ('both')
AND process='{process_dict[process_selected]}'
ORDER BY 1;
"""

all_tables = conn.sql(query).fetchall()

table_names = [table_name[0] for table_name in all_tables]

# Sidebar selectbox to choose a file
table_selected = st.sidebar.selectbox("Choose a table", table_names)

table_markdown = "\n".join([f"- {table}" for table in table_names])

# Display the tables available for the selected process
st.markdown(
    f"""
CFR2SBVR Inspect is a tool to inspect CFR2SBVR checkpoint files.

Tables available for {process_selected}:

{table_markdown}
"""
)

checkpoints = conn.sql(f"select distinct checkpoint from {table_selected} order by 1")

# Sidebar selectbox to choose a file
checkpoints_selected = st.sidebar.multiselect("Choose checkpoints", checkpoints)

st.write("Selected checkpoints:")
st.write(checkpoints_selected)

#
# Load the selected data
#
data_df = load_data(
    table_selected, checkpoints_selected, doc_id_selected, process_selected
)

event = st.dataframe(
    data_df,
    key="id",
    on_select="rerun",
    use_container_width=True,
    selection_mode=["multi-row"],
)

st.write(f"Load {len(data_df)} line(s)")

st.subheader("Compare", divider=True)

if event.selection:
    try:
        columns = st.columns(len(event.selection["rows"]))

        # Loop through the rows and assign each row to a column
        for col, row in zip(columns, event.selection["rows"]):
            with col:
                # Extract values
                doc_id = data_df.at[row, "doc_id"]
                title = data_df.at[row, "statement_title"]
                statement = data_df.at[row, "statement_text"]
                element_id = data_df.at[row, "statement_id"]
                checkpoint = data_df.at[row, "checkpoint"]
                sources = data_df.at[row, "statement_sources"]

                st.write("#### Statement")
                # Display selected row number
                st.write(f"Selected checkpoint '{checkpoint}', row: {row}")

                # Display frozen section
                if doc_id:
                    with st.expander(f"Section {doc_id} (frozen)"):
                        display_section(doc_id)
                
                # Try get values dependent of the process
                missing_messages = []
                try:
                    transformed_statement = data_df.at[row, "transformed"]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    transformed_statement = None
                try:
                    classification_type = data_df.at[
                        row, "statement_classification_type"
                    ]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    classification_type = ""
                # statement_classification_type_confidence
                try:
                    classification_type_confidence = data_df.at[
                        row, "statement_classification_type_confidence"
                    ]
                    classification_type_explanation = data_df.at[
                        row, "statement_classification_type_explanation"
                    ]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    classification_type_confidence = None
                    classification_type_explanation = None
                # statement_classification_subtype
                try:
                    classification_subtype = data_df.at[
                        row, "statement_classification_subtype"
                    ]
                    classification_subtype_confidence = data_df.at[
                        row, "statement_classification_subtype_confidence"
                    ]
                    classification_subtype_explanation = data_df.at[
                        row, "statement_classification_subtype_explanation"
                    ]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    classification_subtype = ""
                    classification_subtype_confidence = None
                    classification_subtype_explanation = None
                try:
                    terms = data_df.at[row, "terms"]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    terms = []
                try:
                    verb_symbols = data_df.at[row, "verb_symbols"]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    verb_symbols = []
                try:
                    template_ids = data_df.at[row, "transformation_template_ids"]
                    transformation_confidence = data_df.at[row, "transformation_confidence"]
                    transformation_reason = data_df.at[row, "transformation_reason"]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    template_ids = []
                    transformation_confidence = None
                    transformation_reason = None
                try:
                    transformation_semscore = data_df.at[row, "semscore"]
                    transformation_similarity_score = data_df.at[row, "similarity_score"]
                    transformation_similarity_score_confidence = data_df.at[row, "similarity_score_confidence"]
                    transformation_findings = data_df.at[row, "findings"]
                    transformation_accuracy = data_df.at[row, "transformation_accuracy"]
                    transformation_grammar_syntax_accuracy = data_df.at[row, "grammar_syntax_accuracy"]
                except Exception as e:
                    missing_messages.append(f"{e}")
                    transformation_semscore = None
                    transformation_similarity_score = None
                    transformation_similarity_score_confidence = None
                    transformation_findings = None
                    transformation_accuracy = None
                    transformation_grammar_sintaxe_accuracy = None

                # Display the title
                if title:
                    st.write(f"Title: **{title}**")

                # Generate highlighted text
                highlighted_text = highlight_statement(
                    "OS",
                    doc_id,
                    element_id,
                    classification_type,
                    classification_subtype,
                    terms,
                    verb_symbols,
                    statement,
                    sources,
                )
                # Display the statement_text
                st.write(highlighted_text, unsafe_allow_html=True)

                # Display the transformed statement
                if transformed_statement:
                    st.write("**Rule**")

                    highlighted_text = highlight_statement(
                        "TS",
                        doc_id,
                        element_id,
                        classification_type,
                        classification_subtype,
                        terms,
                        verb_symbols,
                        transformed_statement,
                        sources,
                    )
                    st.markdown(highlighted_text, unsafe_allow_html=True)

                # display the terms and verb symbols
                terms_list = [
                    dict(t) for t in terms
                ]  # if each item in terms is already a dict
                verb_symbols_list = verb_symbols.tolist()
                st.json(
                    {"terms": terms_list, "verb_symbols": verb_symbols_list}, expanded=1
                )

                # display the classification type and subtype
                with st.expander(f"More info"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Classification**")
                        if classification_type:
                            if st.button(f"type: {classification_type}", key=f"classification_type_{row}", type="tertiary", icon=":material/info:", help="Click to see more information about the classification type"):
                                witt_taxonomy_dialog(classification_type)
                            st.write(f"Confidence: {classification_type_confidence}")
                            st.write(f"Explanation: {classification_type_explanation}")
                        if classification_subtype:
                            if st.button(f"subtype: {classification_subtype}", key=f"classification_subtype_{row}", type="tertiary", icon=":material/info:", help="Click to see more information about the classification subtype"):
                                witt_taxonomy_dialog(classification_subtype)
                            st.write(f"Confidence: {classification_subtype_confidence}")
                            st.write(f"Explanation: {classification_subtype_explanation}")
                    with col2:
                        if template_ids:
                            st.write("**Transformation**")
                            st.write(f"Template(s): \n\n {list_to_markdown(template_ids, ordered=False)}")
                            #st.slider(value=transformation_confidence, min_value=0.00, max_value=1.00, label="Confidence", help="Confidence in the transformation", key=f"transformation_confidence_{row}")
                            st.write(f"Confidence: {transformation_confidence}")
                            st.write(f"Reason: {transformation_reason}")
                        if transformation_semscore:
                            st.write("**Validation**")
                            #st.write(f"Semantic Score: {transformation_semscore}")
                            st.metric(label="Semantic Score", value=round(transformation_semscore, 2), delta=round(transformation_semscore - transformation_similarity_score, 2), help="Semantic score of the transformation and the difference with the similarity score")
                            #st.write(f"Similarity Score: {transformation_similarity_score}")
                            st.metric(label="Similarity Score", value=round(transformation_similarity_score, 2), delta=round(transformation_similarity_score - transformation_semscore, 2), help="Similarity score of the transformation and the difference with the semantic score")
                            st.write(f"Similarity score confidence: {transformation_similarity_score_confidence}")
                            st.write(f"Similarity score findings: \n\n {list_to_markdown(transformation_findings)}")
                            st.write(f"Accuracy: {transformation_accuracy}")
                            st.write(f"Grammar/Sintax Accuracy: {transformation_grammar_syntax_accuracy}")

                # Display what is missing
                if missing_messages:
                    st.write("#### Missing")
                    for missing_message in missing_messages:
                        st.write(f":material/error: {missing_message}")

                # Display similarity scores
                if len(event.selection["rows"]) > 1:
                    st.write("#### Levenshtein Distance (LD)")
                    for other_row in event.selection["rows"]:
                        if other_row != row:
                            for column in ("statement_title", "statement_text", "transformed"):
                                try:
                                    score = calculate_statements_similarity(
                                        str(data_df.at[row, column]),
                                        str(data_df.at[other_row, column]),
                                    )
                                    col1 = column.replace('_', '\_')
                                    st.markdown(
                                        f"- $LD({col1}_{{{row}}}, {col1}_{{{other_row}}} = {score:.2f}$"
                                    )
                                except Exception as e:
                                    st.write(f"No {e} available")
    except Exception as e:
        st.write(f"Nothing to see yet, because {e}.")
