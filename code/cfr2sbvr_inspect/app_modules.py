import html
import logging
import os
from datetime import datetime

# Highlight term in the statement
import re

import duckdb
import jellyfish
import rules_taxonomy_provider.main as rules_taxonomy_provider
import streamlit as st
from openai import OpenAI
from rdflib import Graph, Namespace, Literal, URIRef, XSD, RDF, RDFS
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from rules_taxonomy_provider.main import RuleInformationProvider

logger = logging.getLogger(__name__)


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
        classification, return_forms="rule"
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


def db_connection(db_name, default_data_dir="data"):
    # Connect to the database
    if db_name.startswith("md:"):
        mother_duck_token = os.getenv("MOTHER_DUCK_TOKEN")
        conn = duckdb.connect(
            f"{db_name}?motherduck_token={mother_duck_token}", read_only=True
        )
    else:
        conn = duckdb.connect(f"{default_data_dir}/{db_name}", read_only=True)

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


def chatbot_widget(row_values):
    st.caption("🤖 Chatbot powered by OpenAI")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]


    prompt = ""
    if row_values:
        prompt = st.selectbox("Prompt suggestions", [
        "",
        f"Give a better title up to 3 words for the statement '{row_values['statement_text']}'", 
        f"Could you explain the {row_values['statement_title']} statement? The definition is '{row_values['statement_text']}'",
        ],
        key="prompt_templates")

    #prompt = st.chat_input()

    with st.form("my_chat_form", clear_on_submit=True):

        user_prompt = st.text_input("Type your message here", value=prompt)

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.chat_message("user").write(user_prompt)
            response = client.chat.completions.create(
                model="gpt-4o", messages=st.session_state.messages
            )
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

        if st.form_submit_button('Press enter to send', type="tertiary"):
            user_prompt = ""

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

# Define namespaces with original prefixes
SBVR = Namespace("https://www.omg.org/spec/SBVR/20190601#")
CFR_SBVR = Namespace("http://cfr2sbvr.com/cfr#")
FRO_CFR = Namespace("http://finregont.com/fro/cfr/Code_Federal_Regulations.ttl#")

class Term(BaseModel):
    term: str
    classification: Optional[str]

class RuleAndFact(BaseModel):
    statement_id: str
    statement: str
    concept_type: str
    terms: Optional[List[Term]]
    verb_symbols: Optional[List[str]]
    vocabulary_namespace: str
    sources: Optional[List[str]]
    doc_id: Optional[str]
    metadata_cfr2sbvr: Optional[Dict[str, Any]]


class Designation(BaseModel):
    signifier: str
    statement: str
    concept_type: str
    closeMatch: Optional[List[str]]
    exactMatch: Optional[List[str]]
    vocabulary_namespace: str
    sources: Optional[List[str]]
    doc_id: Optional[str]
    metadata_cfr2sbvr: Optional[Dict[str, Any]]


def transform_to_rdf_subject(input_string: str) -> str:
    """
    Transform the input string to a valid RDF subject by converting it to camel case
    and replacing invalid characters.

    Args:
        input_string (str): The string to transform.

    Returns:
        rdf_subject (str): The transformed RDF subject name.
    """
    # Convert to title case (camel case)
    camel_case_string = ''.join(word.capitalize() for word in input_string.split())
    # Replace invalid characters (retain only alphanumeric and underscore)
    rdf_subject = re.sub(r'[^a-zA-Z0-9_]', '', camel_case_string)

    return rdf_subject

def now_as_xsd_dateTime() -> str:
    return datetime.now().isoformat()

def remove_section_symbol(doc_id: str) -> str:
    return doc_id.replace("§", "")

def triples_rule_and_fact(graph: Graph, rule_fact_model: RuleAndFact):
    designation_class = CFR_SBVR[transform_to_rdf_subject(rule_fact_model.statement_id)]
    label = rule_fact_model.statement_id
    concept_type = rule_fact_model.concept_type
    designation_type = SBVR.DefinitionalRule if concept_type == "Fact" else SBVR.BehavioralBusinessRule

    # Remove existing triples
    graph.remove((designation_class, None, None))

    # Add new triples
    graph.add((designation_class, RDF.type, SBVR[concept_type]))
    graph.add((designation_class, RDF.type, designation_type))
    graph.add((designation_class, SBVR.statement, Literal(rule_fact_model.statement)))
    graph.add((designation_class, RDFS.label, Literal(label)))
    graph.add((designation_class, SBVR.designationIsInNamespace, URIRef(rule_fact_model.vocabulary_namespace)))
    graph.add((designation_class, CFR_SBVR.createDate, Literal(now_as_xsd_dateTime(), datatype=XSD.dateTime)))

    if rule_fact_model.terms:
        for term in rule_fact_model.terms:
            term_uri = CFR_SBVR[transform_to_rdf_subject(term.term)]
            graph.add((designation_class, CFR_SBVR.hasTerm, term_uri))

    if rule_fact_model.verb_symbols:
        for verb_symbol in rule_fact_model.verb_symbols:
            verb_symbol_uri = CFR_SBVR[transform_to_rdf_subject(verb_symbol)]
            graph.add((designation_class, CFR_SBVR.hasVerbSymbol, verb_symbol_uri))

    if rule_fact_model.sources:
        for source in rule_fact_model.sources:
            source_literal = Literal(f"{rule_fact_model.doc_id}{source}")
            graph.add((designation_class, SBVR.referenceSupportsMeaning, source_literal))

    if rule_fact_model.metadata_cfr2sbvr:
        metadata = rule_fact_model.metadata_cfr2sbvr
        graph.add((designation_class, CFR_SBVR.extractOriginalStatement, Literal(metadata.get("extract_original_statement"))))
        graph.add((designation_class, CFR_SBVR.transformationSemscore, Literal(f"{metadata.get('transformation_semscore'):.2f}", datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationSimilarityScore, Literal(f"{metadata.get('transformation_similarity_score'):.2f}", datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationSimilarityScoreConfidence, Literal(f"{metadata.get('transformation_similarity_score_confidence'):.2f}", datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationAccuracy, Literal(f"{metadata.get('transformation_accuracy'):.2f}", datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationGrammarSyntaxAccuracy, Literal(f"{metadata.get('transformation_grammar_syntax_accuracy'):.2f}", datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.classificationType, Literal(metadata.get("classification_type"))))
        graph.add((designation_class, CFR_SBVR.classificationSubtype, Literal(metadata.get("classification_subtype"))))
        graph.add((designation_class, CFR_SBVR.classificationSubtypeConfidence, Literal(f"{metadata.get('classification_subtype_confidence'):.2f}", datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.classificationSubtypeExplanation, Literal(metadata.get("classification_subtype_explanation"))))

        findings = rule_fact_model.metadata_cfr2sbvr.get("transformation_findings")
        if findings is not None and len(findings) > 0:
            for finding in findings:
                graph.add((designation_class, CFR_SBVR.transformationFinding, Literal(finding)))

        templates = rule_fact_model.metadata_cfr2sbvr.get("classification_templates_ids")
        if templates is not None and len(templates) > 0:
            for template in templates:
                graph.add((designation_class, CFR_SBVR.classificationTemplatesId, Literal(template)))

        if concept_type == "Rule":
            graph.add((designation_class, CFR_SBVR.classificationTypeConfidence, Literal(f"{metadata.get('classification_type_confidence'):.2f}", datatype=XSD.decimal)))
            graph.add((designation_class, CFR_SBVR.classificationTypeExplanation, Literal(metadata.get("classification_type_explanation"))))


def generate_insert_sparql_query(graph: Graph) -> str:
    # Generate SPARQL INSERT DATA query
    sparql_insert_query = """
    PREFIX sbvr: <https://www.omg.org/spec/SBVR/20190601#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX cfr-sbvr: <http://cfr2sbvr.com/cfr#>
    PREFIX fro-cfr: <http://finregont.com/fro/cfr/Code_Federal_Regulations.ttl#>

    INSERT DATA {
        GRAPH cfr-sbvr:CFR_SBVR {
    """

    # Add triples to the SPARQL query
    for triple in graph:
        subject, predicate, obj = triple
        # Use prefix notation for URIs
        subject_str = subject.n3(graph.namespace_manager)
        predicate_str = predicate.n3(graph.namespace_manager)
        obj_str = obj.n3(graph.namespace_manager)
        sparql_insert_query += f"    {subject_str} {predicate_str} {obj_str} .\n"

    sparql_insert_query += "}\n}"

    return sparql_insert_query


def triples_verb_symbol(graph: Graph, designation: Designation) -> None:
    signifier = designation.signifier
    transformed_statement = designation.statement
    statement = "missing"  # Placeholder, as per original function
    concept_type = designation.concept_type  # sbvr:VerbConcept
    vocabulary_namespace = designation.vocabulary_namespace
    doc_id = designation.doc_id
    metadata_cfr2sbvr = designation.metadata_cfr2sbvr

    designation_class = URIRef(f"{CFR_SBVR}{transform_to_rdf_subject(f'{signifier}-{remove_section_symbol(doc_id)}')}")

    # Remove existing triples
    graph.remove((designation_class, None, None))

    # Add new triples
    graph.add((designation_class, RDF.type, SBVR.VerbSymbol))
    graph.add((designation_class, RDF.type, SBVR[concept_type]))
    graph.add((designation_class, SBVR.signifier, Literal(signifier)))
    graph.add((designation_class, SBVR.statement, Literal(statement)))
    graph.add((designation_class, SBVR.designationIsInNamespace, URIRef(vocabulary_namespace)))
    graph.add((designation_class, CFR_SBVR.extractOriginalStatement, Literal(metadata_cfr2sbvr.get("extract_original_statement"))))
    graph.add((designation_class, CFR_SBVR.transformedStatement, Literal(transformed_statement)))
    graph.add((designation_class, CFR_SBVR.createDate, Literal(now_as_xsd_dateTime(), datatype=XSD.dateTime)))

    if designation.sources:
        for source in designation.sources:
            source_literal = Literal(f"{doc_id}{source}")
            graph.add((designation_class, SBVR.referenceSupportsMeaning, source_literal))


def triples_term_and_name(graph: Graph, designation: Designation) -> None:
    signifier = designation.signifier
    statement = designation.statement
    concept_type = designation.concept_type
    vocabulary_namespace = designation.vocabulary_namespace
    doc_id = designation.doc_id
    metadata_cfr2sbvr = designation.metadata_cfr2sbvr

    designation_class = URIRef(f"{CFR_SBVR}{transform_to_rdf_subject(f'{signifier}-{remove_section_symbol(doc_id)}')}")

    if concept_type == "Name":
        designation_type = SBVR.IndividualNounConcept
    else:
        designation_type = SBVR.GeneralConcept

    # Determine rule type based on metadata
    classification_subtype = metadata_cfr2sbvr.get("classification_subtype")
    if classification_subtype == "Formal intensional definitions":
        rule_type = SBVR.IntensionalDefinition
    elif classification_subtype == "Formal extensional definitions":
        rule_type = SBVR.Extensionaldefinition
    elif classification_subtype == "Categorization scheme enumerations":
        rule_type = SBVR.Categorizationscheme
    else:
        rule_type = SBVR.DefinitionalRule

    # Remove existing triples
    graph.remove((designation_class, None, None))

    # Add new triples
    graph.add((designation_class, RDF.type, designation_type))
    graph.add((designation_class, RDF.type, rule_type))
    graph.add((designation_class, RDF.type, SBVR[concept_type]))
    graph.add((designation_class, SBVR.signifier, Literal(signifier)))
    graph.add((designation_class, SBVR.statement, Literal(statement)))
    graph.add((designation_class, SBVR.designationIsInNamespace, URIRef(vocabulary_namespace)))
    graph.add((designation_class, CFR_SBVR.createDate, Literal(now_as_xsd_dateTime(), datatype=XSD.dateTime)))

    if designation.closeMatch:
        for close_match in designation.closeMatch:
            graph.add((designation_class, SBVR.closeMatch, URIRef(close_match)))

    if designation.exactMatch:
        for exact_match in designation.exactMatch:
            graph.add((designation_class, SBVR.exactMatch, URIRef(exact_match)))

    if designation.sources:
        for source in designation.sources:
            source_literal = Literal(f"{doc_id}{source}")
            graph.add((designation_class, SBVR.referenceSupportsMeaning, source_literal))

    if metadata_cfr2sbvr:
        graph.add((designation_class, CFR_SBVR.extractOriginalStatement, Literal(metadata_cfr2sbvr.get("extract_original_statement"))))
        graph.add((designation_class, CFR_SBVR.transformationSemscore, Literal(metadata_cfr2sbvr.get("transformation_semscore"), datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationSimilarityScore, Literal(metadata_cfr2sbvr.get("transformation_similarity_score"), datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationSimilarityScoreConfidence, Literal(metadata_cfr2sbvr.get("transformation_similarity_score_confidence"), datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationAccuracy, Literal(metadata_cfr2sbvr.get("transformation_accuracy"), datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.transformationGrammarSyntaxAccuracy, Literal(metadata_cfr2sbvr.get("transformation_grammar_syntax_accuracy"), datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.classificationType, Literal(metadata_cfr2sbvr.get("classification_type"))))
        graph.add((designation_class, CFR_SBVR.classificationSubtype, Literal(metadata_cfr2sbvr.get("classification_subtype"))))
        graph.add((designation_class, CFR_SBVR.classificationSubtypeConfidence, Literal(metadata_cfr2sbvr.get("classification_subtype_confidence"), datatype=XSD.decimal)))
        graph.add((designation_class, CFR_SBVR.classificationSubtypeExplanation, Literal(metadata_cfr2sbvr.get("classification_subtype_explanation"))))

        findings = metadata_cfr2sbvr.get("transformation_findings")
        if findings is not None and len(findings) > 0:
            for finding in findings:
                graph.add((designation_class, CFR_SBVR.transformationFinding, Literal(finding)))

        templates = metadata_cfr2sbvr.get("classification_templates_ids")
        if templates is not None and len(templates) > 0:
            for template in templates:
                graph.add((designation_class, CFR_SBVR.classificationTemplatesId, Literal(template)))


def now_as_xsd_dateTime():
    # Get the current datetime in UTC
    current_time = datetime.utcnow().isoformat()

    # Remove microseconds for compliance
    if '.' in current_time:
        current_time = current_time.split('.')[0]

    # Append the UTC timezone indicator
    current_time += 'Z'

    return current_time

def get_metadata_cfr2sbvr(element):
    return {
        "extract_original_statement":element.get('statement_text', 'missing'),
        "transformation_semscore": element.get("semscore", 0),
        "transformation_similarity_score":element.get("similarity_score", 0),
        "transformation_similarity_score_confidence":element.get("similarity_score_confidence", 0),
        "transformation_accuracy":element.get("transformation_accuracy", 0),
        "transformation_grammar_syntax_accuracy":element.get("grammar_syntax_accuracy", 0),
        "transformation_findings":element.get("findings", []),
        # from classification
        "classification_type":element.get("statement_classification_type", 'missing'),
        "classification_subtype":element.get("statement_classification_subtype", 'missing'),
        "classification_type_confidence":element.get("statement_classification_type_confidence", 0),
        "classification_type_explanation":element.get("statement_classification_type_explanation", 'not available'),
        "classification_subtype_confidence":element.get("statement_classification_subtype_confidence", 0),
        "classification_subtype_explanation":element.get("statement_classification_subtype_explanation", 'not available'),
        "classification_templates_ids":element.get("transformation_template_ids", [])
    }

def normalize_ns_string(input_string: str) -> str:
    """
    Transform the input string to title case, which capitalizes the first letter of each word.

    Args:
        input_string (str): The string to normalize.

    Returns:
        normalized_string (str): The normalized string.
    """
    normalized_string = remove_section_symbol(input_string)

    # Remove all spaces, change points and hyphens to underscores
    return normalized_string.replace(" ", "").replace("-", "_").replace(".", "_")

def define_vocabulary_ns(doc_id: str, is_local_scope: bool) -> str:
    """
    Determines the vocabulary section ID based on the term's source section.

    Args:
        section_id (str): The section ID of the current document.
        source_section: The section id.

    Returns:
        str: The appropriate vocabulary section ID.

    Raises:
        KeyError: If 'source' or 'section' key is missing in the term.
        TypeError: If 'section_id' is not a string or 'term' is not a dictionary.
    """

    doc_id = remove_section_symbol(normalize_ns_string(doc_id))

    if is_local_scope:
        ns = f"cfr-sbvr:CFR_SBVR_{doc_id}_NS"
    else:
        ns = "fro-cfr:CFR_Title_17_Part_275_NS"

    logger.info(f"Vocabulary namespace: {ns}")
    
    return ns


# Define namespaces with original prefixes
SBVR = Namespace("https://www.omg.org/spec/SBVR/20190601#")
CFR_SBVR = Namespace("http://cfr2sbvr.com/cfr#")
FRO_CFR = Namespace("http://finregont.com/fro/cfr/Code_Federal_Regulations.ttl#")

def df_to_rdf_triples(data_df) -> str:
    pred_terms_names = data_df[data_df['source'].isin(['Terms','Names'])].to_dict(orient="records")
    pred_operative_rules_fact_types = data_df[data_df['source'].isin(['Operative_Rules','Fact_Types'])].to_dict(orient="records")

    g = Graph()
    g.bind("sbvr", SBVR)
    g.bind("cfr-sbvr", CFR_SBVR)
    g.bind("fro-cfr", FRO_CFR)
    g.bind("rdfs", RDFS)

    for index, element in enumerate(pred_terms_names):
        logger.info(f"{index=}")

        # from extraction
        doc_id = element.get('doc_id') # section

        statement_id = element.get('statement_id')
        
        statement = element.get('transformed')
        statement = statement if statement else "missing" # Change None to "missing"

        # SBVR ontology
        concept_type = "Term" if element.get('source') == "Terms" else "Name"
        
        sources = element.get("statement_sources") # paragraphs
        
        is_local_scope = element.get("isLocalScope")
        
        # from transformation
        metadata_cfr2sbvr = get_metadata_cfr2sbvr(element)

        # create vocabulary and namespace if not exists
        vocabulary = define_vocabulary_ns(doc_id, is_local_scope)

        # similar search
        exact_match, close_match = [], []#get_similar_signifiers(kg_conn, statement_id)

        # create designation
        designation = Designation(
            signifier=statement_id,
            statement=statement,
            concept_type=concept_type,
            closeMatch=close_match,
            exactMatch=exact_match,
            vocabulary_namespace=vocabulary,
            sources=sources,
            doc_id=doc_id,
            metadata_cfr2sbvr=metadata_cfr2sbvr
        )
        triples_term_and_name(g, designation)

        logger.debug(g.serialize(format="turtle"))


    for index, element in enumerate(pred_operative_rules_fact_types):
        logger.info(f"{index=}")

        # from extraction
        doc_id = element.get('doc_id') # section

        statement_id = element.get('statement_id')
        
        statement = element.get('transformed')
        statement = statement if statement else "missing" # Change None to "missing"

        # SBVR ontology
        concept_type = "Rule" if element.get('source') == "Operative_Rules" else "Fact"

        statement_title = element.get('statement_title')

        sources = element.get("statement_sources") # paragraph

        terms = element.get("terms")

        verb_symbols = element.get("verb_symbols")
        
        # from transformation
        metadata_cfr2sbvr = get_metadata_cfr2sbvr(element)

        # create vocabulary and namespace if not exists
        vocabulary = define_vocabulary_ns(doc_id, True)

        # create Fact model
        rule_fact_model = RuleAndFact(
            statement_id=statement_title,
            statement=statement,
            concept_type=concept_type,
            terms=terms,
            verb_symbols=verb_symbols,
            vocabulary_namespace=vocabulary,
            sources=sources,
            doc_id=doc_id,
            metadata_cfr2sbvr=metadata_cfr2sbvr
        )
        triples_rule_and_fact(g, rule_fact_model)

        logger.debug(g.serialize(format="turtle"))


    for index, element in enumerate(pred_operative_rules_fact_types):
        logger.info(f"{index=}")

        # from extraction
        doc_id = element.get('doc_id') # section

        # There is no extracted / transformed statement for verb symbols
        # Storging the fact or rule statement as cfr-sbvr:transformedStatement
        statement = element.get('transformed')
        statement = statement if statement else "missing" # Change None to "missing"

        # SBVR ontology
        concept_type = "VerbConcept"
        
        sources = element.get("statement_sources") # paragraph

        verb_symbols = element.get("verb_symbols")
        
        # from transformation
        metadata_cfr2sbvr = get_metadata_cfr2sbvr(element)

        vocabulary = define_vocabulary_ns(doc_id, True)

        for verb_symbol in verb_symbols:
            # create Fact model
            designation_model = Designation(
                signifier=verb_symbol,
                statement=statement, # There is no extracted / transformed statement
                concept_type=concept_type,
                closeMatch=[],
                exactMatch=[],
                vocabulary_namespace=vocabulary,
                sources=sources,
                doc_id=doc_id,
                metadata_cfr2sbvr=metadata_cfr2sbvr
            )

            triples_verb_symbol(g, designation_model)

            logger.debug(g.serialize(format="turtle"))

    logger.debug(g.serialize(format="turtle"))
    return g.serialize(format="turtle")