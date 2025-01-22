import html
import logging
import os

# Highlight term in the statement
import re

import duckdb
import jellyfish
import rules_taxonomy_provider.main as rules_taxonomy_provider
import streamlit as st
from openai import OpenAI
from rules_taxonomy_provider.main import RuleInformationProvider

logger = logging.getLogger(__name__)

# def highlight_statement(
#     line_id,
#     doc_id,
#     element_id,
#     classification_type,
#     classification_subtype,
#     terms,
#     verb_symbols,
#     statement,
#     sources,
# ):
#     keywords = [
#         "the",
#         "a",
#         "an",
#         "another",
#         "a given",
#         "that",
#         "who",
#         "what",
#         "and",
#         "or",
#         "but not both",
#         "if",
#         "if and only if",
#         "not",
#         "does not",
#         "must",
#         "must not",
#         "need not",
#         "always",
#         "never",
#         "can",
#         "cannot",
#         "may",
#         "might",
#         "can not",
#         "could not",
#         "only if",
#         "it is obligatory that",
#         "it is prohibited that",
#         "it is impossible that",
#         "it is possible that",
#         "it is permitted that",
#         "not both",
#         "neither",
#         "either",
#         "nor",
#         "whether or not",
#         "each",
#         "some",
#         "at least one",
#         "at least",
#         "at most one",
#         "at most",
#         "exactly one",
#         "exactly",
#         "at least",
#         "and at most",
#         "more than one",
#         "no",
#         "the",  # repetido, mas sem problemas
#         "a",  # repetido, mas sem problemas
#     ]

#     sources_links = []
#     for source in sources:
#         doc_id_url = doc_id.replace("§ ", "")
#         url = f"https://www.ecfr.gov/current/title-17/part-275#p-{doc_id_url}{source}"
#         sources_links.append(f'<a href="{url}">{source}</a>')
#     sources = ", ".join(sources_links)

#     classification = classification_type
#     if classification_subtype:
#         classification += f" | {classification_subtype}"

#     # Clean up the statement
#     statement = statement.replace("$", "\\$")

#     def highlight_match_term(term_info):
#         def replace_term(match):
#             original = match.group(0)
#             if term_info["classification"] == "Common Noun":
#                 # Aqui apenas inserimos um placeholder. Depois criaremos o tooltip.
#                 return (
#                     f'<span style="text-decoration: underline; '
#                     f'text-decoration-color: green;">{original}</span>'
#                 )
#             elif term_info["classification"] == "Proper Noun":
#                 return (
#                     f'<span style="text-decoration: underline double; '
#                     f'text-decoration-color: green;">{original}</span>'
#                 )
#             return original

#         return replace_term

#     # Apply term substitutions (sem inserir tooltip ainda).
#     for t in terms:
#         term_regex = rf"\b{re.escape(t['term'])}\b"
#         statement = re.sub(
#             term_regex, highlight_match_term(t), statement, flags=re.IGNORECASE
#         )

#     # Highlight verb symbols (em itálico azul)
#     def highlight_match_verb(match):
#         original = match.group(0)
#         return f'<span style="font-style: italic; color: blue;">{original}</span>'

#     for verb in verb_symbols:
#         verb_regex = rf"\b{re.escape(verb)}\b"
#         statement = re.sub(
#             verb_regex, highlight_match_verb, statement, flags=re.IGNORECASE
#         )

#     # Destaque das keywords em laranja
#     def highlight_match_keyword(match):
#         original = match.group(0)
#         return f'<span style="color: orange;">{original}</span>'

#     for kw in keywords:
#         kw_regex = rf"\b{re.escape(kw)}\b"
#         statement = re.sub(
#             kw_regex, highlight_match_keyword, statement, flags=re.IGNORECASE
#         )

#     # Função para adicionar tooltip aos termos (Common ou Proper Noun)
#     def add_tooltip(term_info):
#         definition = html.escape(term_info.get("definition", "") or "Missing")
#         confidence = term_info.get("confidence", "")
#         reason = html.escape(term_info.get("reason", "") or "Missing")
#         transformed = html.escape(term_info.get("transformed", "") or "Missing")
#         transformed_confidence = term_info.get("transform_confidence", "") or "0.0"
#         transformed_reason = html.escape(term_info.get("transform_reason", "") or "Missing")
#         isLocalScope = term_info.get("isLocalScope", False)
#         if isLocalScope:
#             scope = "📍Local\n"
#         else:
#             scope = "🌎 Elsewhere\n"

#         tooltip_content = (
#             f"{scope} "
#             f"• Definition: {definition}\n"
#             f"• D. confidence: {confidence}\n"
#             f"• D. reason: {reason}\n"
#             f"• Transformed: {transformed}\n"
#             f"• T. confidence: {transformed_confidence}\n"
#             f"• T. Reason: {transformed_reason}"
#         )

#         # Regex para capturar a tag já gerada acima (simples ou dupla) com o texto correspondente.
#         if term_info["classification"] == "Common Noun":
#             # Sublinha simples
#             tag_pattern = (
#                 r'<span style="text-decoration: underline; text-decoration-color: green;">'
#                 rf"(?P<content>{re.escape(term_info['term'])}|{re.escape(term_info['term'].lower())}|{re.escape(term_info['term'].capitalize())})"
#                 r"</span>"
#             )
#         else:
#             # Sublinha dupla
#             tag_pattern = (
#                 r'<span style="text-decoration: underline double; text-decoration-color: green;">'
#                 rf"(?P<content>{re.escape(term_info['term'])}|{re.escape(term_info['term'].lower())}|{re.escape(term_info['term'].capitalize())})"
#                 r"</span>"
#             )

#         def insert_title(match):
#             original_span = match.group(0)
#             content = match.group("content")
#             # Insere o atributo title sem alterar o texto interno
#             return original_span.replace(
#                 ';">' + content,  # ponto de inserção
#                 f';" title="{tooltip_content}">' + content,
#             )

#         return tag_pattern, insert_title

#     # Agora adicionamos o tooltip aos termos que foram sublinhados
#     for t in terms:
#         tag_pattern, insert_title_fn = add_tooltip(t)
#         statement = re.sub(tag_pattern, insert_title_fn, statement, flags=re.IGNORECASE)

#     sup = f" [{classification}]"
#     final_text = f"{line_id}: <strong>{sources}</strong> {statement}{sup}"
#     return final_text


def highlight_statement(
    line_id,
    doc_id,
    element_id,
    classification_type,
    classification_subtype,
    terms,
    verb_symbols,
    statement,
    sources,
):
    keywords = [
        "the",
        "a",
        "an",
        "another",
        "a given",
        "that",
        "who",
        "what",
        "and",
        "or",
        "but not both",
        "if",
        "if and only if",
        "not",
        "does not",
        "must",
        "must not",
        "need not",
        "always",
        "never",
        "can",
        "cannot",
        "may",
        "might",
        "can not",
        "could not",
        "only if",
        "it is obligatory that",
        "it is prohibited that",
        "it is impossible that",
        "it is possible that",
        "it is permitted that",
        "not both",
        "neither",
        "either",
        "nor",
        "whether or not",
        "each",
        "some",
        "at least one",
        "at least",
        "at most one",
        "at most",
        "exactly one",
        "exactly",
        "at least",
        "and at most",
        "more than one",
        "no",
        "the",  # repetido
        "a",  # repetido
    ]

    sources_links = []
    for source in sources:
        doc_id_url = doc_id.replace("§ ", "")
        url = f"https://www.ecfr.gov/current/title-17/part-275#p-{doc_id_url}{source}"
        sources_links.append(f'<a href="{url}">{source}</a>')
    sources = ", ".join(sources_links)

    classification = classification_type
    if classification_subtype:
        classification += f" | {classification_subtype}"

    statement = statement.replace("$", "\\$")

    def highlight_match_term(term_info):
        def replace_term(match):
            original = match.group(0)
            if term_info["classification"] == "Common Noun":
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

    for t in terms:
        term_regex = rf"\b{re.escape(t['term'])}\b"
        statement = re.sub(
            term_regex, highlight_match_term(t), statement, flags=re.IGNORECASE
        )

    def highlight_match_verb(match):
        return f'<span style="font-style: italic; color: blue;">{match.group(0)}</span>'

    for verb in verb_symbols:
        verb_regex = rf"\b{re.escape(verb)}\b"
        statement = re.sub(
            verb_regex, highlight_match_verb, statement, flags=re.IGNORECASE
        )

    # Função auxiliar para destacar keywords apenas fora de <span>
    def highlight_keywords_outside_spans(text, kw_list):
        # Divide em segmentos que são ou não <span ...>...</span>
        segments = re.split(
            r"(<span.*?>.*?</span>)", text, flags=re.IGNORECASE | re.DOTALL
        )
        # Em cada segmento fora de <span>, faz a substituição das keywords
        for i, seg in enumerate(segments):
            if seg.startswith("<span"):
                continue
            for kw in kw_list:
                kw_pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                seg = kw_pattern.sub(
                    lambda m: f'<span style="color: orange;">{m.group(0)}</span>', seg
                )
            segments[i] = seg
        return "".join(segments)

    # Agora destacamos keywords apenas fora de spans
    statement = highlight_keywords_outside_spans(statement, keywords)

    def add_tooltip(term_info):
        definition = html.escape(term_info.get("definition", "") or "Missing")
        confidence = term_info.get("confidence", "")
        reason = html.escape(term_info.get("reason", "") or "Missing")
        transformed = html.escape(term_info.get("transformed", "") or "Missing")
        transformed_confidence = term_info.get("transform_confidence", "") or "0.0"
        transformed_reason = html.escape(
            term_info.get("transform_reason", "") or "Missing"
        )
        isLocalScope = term_info.get("isLocalScope", False)
        if isLocalScope:
            scope = "📍Local\n"
        else:
            scope = "🌎 Elsewhere\n"

        tooltip_content = (
            f"{scope} "
            f"• Definition: {definition}\n"
            f"• D. confidence: {confidence}\n"
            f"• D. reason: {reason}\n"
            f"• Transformed: {transformed}\n"
            f"• T. confidence: {transformed_confidence}\n"
            f"• T. Reason: {transformed_reason}"
        )

        if term_info["classification"] == "Common Noun":
            tag_pattern = (
                r'<span style="text-decoration: underline; text-decoration-color: green;">'
                rf"(?P<content>{re.escape(term_info['term'])}|"
                rf"{re.escape(term_info['term'].lower())}|"
                rf"{re.escape(term_info['term'].capitalize())})"
                r"</span>"
            )
        else:
            tag_pattern = (
                r'<span style="text-decoration: underline double; text-decoration-color: green;">'
                rf"(?P<content>{re.escape(term_info['term'])}|"
                rf"{re.escape(term_info['term'].lower())}|"
                rf"{re.escape(term_info['term'].capitalize())})"
                r"</span>"
            )

        def insert_title(match):
            original_span = match.group(0)
            content = match.group("content")
            return original_span.replace(
                ';">' + content,
                f';" title="{tooltip_content}">' + content,
            )

        return tag_pattern, insert_title

    for t in terms:
        tag_pattern, insert_title_fn = add_tooltip(t)
        statement = re.sub(tag_pattern, insert_title_fn, statement, flags=re.IGNORECASE)

    sup = f" [{classification}]"
    final_text = f"{line_id}: <strong>{sources}</strong> {statement}{sup}"
    return final_text


def display_section(conn, doc_id):
    doc_id_url = doc_id.replace("§ ", "")
    section_url = f"https://www.ecfr.gov/current/title-17/section-{doc_id_url}"

    content = conn.sql(
        f"SELECT content FROM RAW_SECTION WHERE id = '{doc_id}'"
    ).fetchall()[0][0]
    content = content.replace("\n", "<br>")
    content = content.replace("$", "\$")
    content = content.replace(doc_id, f'<a href="{section_url}">{doc_id}</a>')
    return content


@st.dialog("Witt (2012) taxonomy", width="large")
def witt_taxonomy_dialog(classification):
    rule_provider = RuleInformationProvider("code/cfr2sbvr_inspect/data")
    markdown_data = rule_provider.get_classification_and_templates(
        classification, return_forms="fact_type"
    )
    st.markdown(markdown_data)


@st.dialog("ⓘ Info", width="large")
def info_dialog(topic):
    if topic == "process":
        st.markdown(
            """
            ### Process
            - Extraction: Extracts the fact types and operative rules statements from the CFR sections.
            - Classification: Classifies the statements extracted with Witt (2012) taxonomy.
            - Transformation: Transform the statements into SBVR using Witt (2012) templates.
            - Validation: Validates the extraction, classification, and transformation processes against a golden dataset calculating precision, accuracy, and other scores.
            """
        )
        st.image("code/cfr2sbvr_inspect/static/cfr2sbvr-process.png")
        st.write(
            """
                Version considerations:
                 - Version 4 (database_v4.db): The process is as shown in the picture; the true table is used as input for each process after extraction.
                 - Version 5 (database_v5.db): The process is slightly different from the picture, the output checkpoint from one process is used as input for the next.
                 """
        )
        st.write("> Witt, Graham. Writing effective business rules. Elsevier, 2012.")


def list_to_markdown(list, ordered=True):
    if ordered:
        prefix = "1."
    else:
        prefix = "-"
    return "\n".join([f"{prefix} {item}" for item in list])


def disconnect_db(conn):
    st.write("called")
    # conn.close()


def get_databases(local_db):
    if local_db:
        return ["database_v5.db", "database_v4.db"]
    else:
        return ["md:cfr2sbvr_db"]


def db_connection(db_name, local_db, default_data_dir="data"):
    # Connect to the database
    if local_db:
        conn = duckdb.connect(f"{default_data_dir}/{db_name}.db", read_only=True)
    else:
        mother_duck_token = os.getenv("MOTHER_DUCK_TOKEN")
        conn = duckdb.connect(
            f"{db_name}?motherduck_token=md:{mother_duck_token}", read_only=True
        )
    return conn, db_name


# @st.cache_data
def load_data(conn, table, checkpoints, doc_ids, statement_sources, process_selected):
    where_clause = ""
    if checkpoints:
        checkpoints_string = ", ".join(f"'{item}'" for item in checkpoints)
        where_clause += f" AND checkpoint in ({checkpoints_string})"

    if doc_ids:
        doc_ids_string = ", ".join(f"'{item}'" for item in doc_ids)
        where_clause += f" AND doc_id in ({doc_ids_string})"

    if statement_sources:
        statement_sources_string = ", ".join(f"'{item}'" for item in statement_sources)
        where_clause += (
            f" AND list_has_any([{statement_sources_string}], statement_sources)"
        )

    data_query = f"""
    SELECT *
    FROM {table}
    WHERE 1 = 1
    {where_clause}
    ORDER BY *
    ;
    """

    logger.debug(data_query)
    df = conn.sql(query=data_query).fetchdf()
    return df


def calculate_statements_similarity(statement1, statement2):
    # Calculate the similarity score using the Levenshtein distance
    score = jellyfish.levenshtein_distance(statement1.lower(), statement2.lower())
    similarity_score = 1 - (
        score / max(len(statement1), len(statement2))
    )  # Normalize to a similarity score
    return similarity_score


def get_table_names(conn, process_dict, process_selected):
    query = f"""
    SELECT DISTINCT TABLE_NAME,
    FROM CHECKPOINT_METADATA
    WHERE doc_source in ('both')
    AND process='{process_dict[process_selected]}'
    ORDER BY 1 DESC;
    """

    all_tables = conn.sql(query).fetchall()

    return [table_name[0] for table_name in all_tables]


def get_doc_ids(conn):
    return conn.sql(
        """
        select distinct id as doc_id
        from RAW_SECTION
        order by id
        """)


def get_checkpoints(conn, table_selected):
    return conn.sql(
        f"""
        select distinct checkpoint as checkpoint
        from RAW_SECTION_EXTRACTED_ELEMENTS_VW --{table_selected}
        order by checkpoint
        """
    )


def get_statement_sources(conn, table_selected):
    return conn.sql(
        f"""
        select distinct unnest(statement_sources) as statement_source
        from RAW_SECTION_EXTRACTED_ELEMENTS_VW --{table_selected}
        order by statement_source
        """
    )


def extract_row_values(data_df, row):
    # Try get values dependent of the process
    missing_messages = []
    row_values = {}

    row_values["doc_id"] = data_df.at[row, "doc_id"]
    row_values["statement_title"] = data_df.at[row, "statement_title"]
    row_values["statement_text"] = data_df.at[row, "statement_text"]
    row_values["statement_id"] = data_df.at[row, "statement_id"]
    row_values["checkpoint"] = data_df.at[row, "checkpoint"]
    row_values["statement_sources"] = data_df.at[row, "statement_sources"]

    # transformed
    try:
        row_values["transformed"] = data_df.at[row, "transformed"]
    except Exception as e:
        missing_messages.append(f"{e}")

    # statement_classification_type
    try:
        row_values["statement_classification_type"] = data_df.at[
            row, "statement_classification_type"
        ]
    except Exception as e:
        missing_messages.append(f"{e}")
    # statement_classification_type_confidence
    try:
        row_values["statement_classification_type_confidence"] = data_df.at[
            row, "statement_classification_type_confidence"
        ]
        row_values["statement_classification_type_explanation"] = data_df.at[
            row, "statement_classification_type_explanation"
        ]
    except Exception as e:
        missing_messages.append(f"{e}")
    # statement_classification_subtype
    try:
        row_values["statement_classification_subtype"] = data_df.at[
            row, "statement_classification_subtype"
        ]
        row_values["statement_classification_subtype_confidence"] = data_df.at[
            row, "statement_classification_subtype_confidence"
        ]
        row_values["statement_classification_subtype_explanation"] = data_df.at[
            row, "statement_classification_subtype_explanation"
        ]
    except Exception as e:
        missing_messages.append(f"{e}")
    # terms
    try:
        row_values["terms"] = data_df.at[row, "terms"]
    except Exception as e:
        missing_messages.append(f"{e}")

    # verb_symbols
    try:
        row_values["verb_symbols"] = data_df.at[row, "verb_symbols"]
    except Exception as e:
        missing_messages.append(f"{e}")

    # transformation_template_ids
    try:
        row_values["transformation_template_ids"] = data_df.at[
            row, "transformation_template_ids"
        ]
    except Exception as e:
        missing_messages.append(f"{e}")

    # transformation_confidence
    try:
        row_values["transformation_confidence"] = data_df.at[
            row, "transformation_confidence"
        ]
        row_values["transformation_reason"] = data_df.at[row, "transformation_reason"]
    except Exception as e:
        missing_messages.append(f"{e}")

    # transformation scores
    try:
        row_values["transformation_semscore"] = data_df.at[row, "semscore"]
        row_values["transformation_similarity_score"] = data_df.at[
            row, "similarity_score"
        ]
        row_values["transformation_similarity_score_confidence"] = data_df.at[
            row, "similarity_score_confidence"
        ]
        row_values["transformation_findings"] = data_df.at[row, "findings"]
        row_values["transformation_accuracy"] = data_df.at[
            row, "transformation_accuracy"
        ]
        row_values["transformation_grammar_syntax_accuracy"] = data_df.at[
            row, "grammar_syntax_accuracy"
        ]
    except Exception as e:
        missing_messages.append(f"{e}")

    # source of statement
    try:
        row_values["statament_from"] = data_df.at[row, "source"]
    except Exception as e:
        missing_messages.append(f"from {e}")

    return row_values, missing_messages


def format_score(score, THRESHOLD):
    if not score:
        score = 0.0
    if score < THRESHOLD:
        return f'<span style="color:red;">{score:.2f}</span>'
    else:
        return f"{score:.2f}"


def chatbot_widget():
    st.caption("🤖 Chatbot powered by OpenAI")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(
            model="gpt-4o", messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


def log_config(home_dir):
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # Console log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Custom date format
        filename=f"{home_dir}/streamlit_app.log",
    )

    logger = logging.getLogger(__name__)

    return logger
