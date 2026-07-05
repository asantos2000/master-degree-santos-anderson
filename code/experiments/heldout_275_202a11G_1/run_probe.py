"""
Held-out generalization probe for CFR2SBVR.

Runs the FROZEN pipeline (prompts, models, and parameters exactly as in
code/src/chap_6_* notebooks) a SINGLE time, with NO gold-standard correction
at any checkpoint, on the unseen section 17 CFR 275.202(a)(11)(G)-1
(Family offices), and reports reference-free indicators:

  1. Grounding (hallucination) check: extracted statements and definitions
     matched against the source text (no gold standard required).
  2. SemScore: embedding cosine similarity between each original statement
     or definition and its SBVR-SE transformation (as in
     chap_7_validation_rules_transformation.ipynb, cell "compare_sentences").
  3. LLM-as-a-Judge: similarity, transformation accuracy, and grammar/syntax
     scores of the transformed statement against the source statement and
     the writing templates (frozen judge prompts).

The supervised metrics of the article (precision/recall/F1 against the gold
standard) are intentionally NOT computed: no gold standard exists for this
section, which is the point of the probe.

Usage:
    export OPENAI_API_KEY=...   (or set in the environment)
    python run_probe.py             # runs pipeline + indicators + report
    python run_probe.py --selftest  # checks imports, data files, prompts (no API calls)

The script is resumable: every stage is checkpointed in outputs/, and stages
already present in the checkpoint are skipped. Delete outputs/ for a clean run.

Provenance of frozen artifacts (copied verbatim):
  - Extraction prompts P1/P2 and response models:
      chap_6_semantic_annotation_elements_extraction.ipynb (system_prompt_v4_1/v4_2)
  - Classification prompts P1/P2 and response models:
      chap_6_semantic_annotation_rules_classification.ipynb
  - Transformation prompts, template formulation, and response models:
      chap_6_nlp2sbvr_transform.ipynb
  - Judge prompts and response models, SemScore functions:
      chap_7_validation_rules_transformation.ipynb
  - LLM parameters: config.yaml (MODEL=gpt-4o, TEMPERATURE=0, MAX_TOKENS=8192)
"""

import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from itertools import islice
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Set

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_CODE_DIR = EXPERIMENT_DIR.parent.parent          # .../code
SRC_DIR = REPO_CODE_DIR / "src"
DATA_DIR = REPO_CODE_DIR / "data"
OUT_DIR = EXPERIMENT_DIR / "outputs"
CHECKPOINT_FILE = OUT_DIR / "probe_checkpoint.json"
SECTION_FILE = EXPERIMENT_DIR / "section_275_202a11G_1.txt"
SECTION_ID = "§ 275.202(a)(11)(G)-1"

sys.path.insert(0, str(SRC_DIR))

from pydantic import BaseModel, Field  # noqa: E402

from checkpoint.main import (  # noqa: E402
    Document,
    DocumentManager,
    DocumentProcessor,
    restore_checkpoint,
    save_checkpoint,
)
from rules_taxonomy_provider.main import (  # noqa: E402
    RuleInformationProvider,
    RulesTemplateProvider,
)

# Frozen LLM parameters (config.yaml)
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 8192
EMBEDDING_MODEL = "text-embedding-3-large"  # as in chap_7 compare_sentences

# Acceptance threshold used in the article for flagging statements for SME review
ACCEPTANCE_THRESHOLD = 0.8


# --------------------------------------------------------------------------
# Response models (verbatim from the notebooks)
# --------------------------------------------------------------------------

# Extraction P1 (chap_6_semantic_annotation_elements_extraction.ipynb)
class Item(BaseModel):
    term: str = Field(..., description="The term is a word or a group of words that represents a specific concept, entity, or subject in a particular context")
    classification: str = Field(..., description="The classification of the term, either 'Common Noun' or 'Proper Noun'.")
    confidence: float = Field(..., description="The confidence score of the classification.")
    reason: Optional[str] = Field(None, description="The reason for the confidence score.")
    extracted_confidence: float = Field(..., description="The confidence scores of the terms extracted from the statement.")
    extracted_reason: Optional[str] = Field(None, description="The reasons for the confidence scores of the terms extracted from the statement.")


class Element(BaseModel):
    id: int = Field(..., description="A unique numeric identifier for each fact, fact type, or rule.")
    title: str = Field(..., description="The title for statement.")
    statement: str = Field(..., description="The full statement or phrase representing the fact, fact type, or rule.")
    terms: List[Item] = Field(..., description="A list of terms involved in the fact, fact type, or rule.")
    verb_symbols: List[str] = Field(..., description="A list of vers, verb phrases or prepositions connecting the terms.")
    verb_symbols_extracted_confidence: List[float] = Field(..., description="The confidence scores of the verb symbols extracted from the statement.")
    verb_symbols_extracted_reason: List[str] = Field(..., description="The reasons for the confidence scores of the verb symbols extracted from the statement.")
    classification: str = Field(..., description="Indicates whether the statement is classified as 'Fact', 'Fact Type', or 'Operative Rule'.")
    confidence: float = Field(..., description="The confidence score of the classification.")
    reason: Optional[str] = Field(None, description="The reason for the confidence score.")
    sources: List[str] = Field(..., description="The paragraph ID of the document where the fact, fact type, or rule is located (e.g., '(a)', '(b)(2)').")


class ElementsDocumentModel(BaseModel):
    section: str = Field(..., description="The section ID of the document.")
    summary: str = Field(..., description="The summary of the document.")
    elements: List[Element] = Field(..., description="A list of facts, fact types, and rules extracted from the document.")


# Extraction P2
class Term(BaseModel):
    term: str = Field(..., description="The term is a word or a group of words that represents a specific concept, entity, or subject in a particular context")
    definition: Optional[str] = Field(None, description="Definition is a explanation or description of the meaning of the term.")
    confidence: float = Field(..., description="The confidence score of the definition extracted from the document.")
    reason: Optional[str] = Field(None, description="The reason for the confidence score.")
    isLocalScope: bool = Field(..., description="Indicates whether the statement is a local scope or not.")
    local_scope_confidence: float = Field(..., description="The confidence score of the local scope.")
    local_scope_reason: Optional[str] = Field(None, description="The reason for the local scope confidence score.")


class TermsRelationship(BaseModel):
    term_1: str = Field(..., description="First term in the relationship.")
    term_2: str = Field(..., description="Second term in the relationship.")
    relation: str = Field(..., description="The type of relationship between the terms.")
    confidence: float = Field(..., description="The confidence score of the relationship extracted from the document.")
    reason: Optional[str] = Field(None, description="The reason for the confidence score.")


class TermsDocumentModel(BaseModel):
    terms: List[Term] = Field(..., description="A list of terms.")
    terms_relationship: List[TermsRelationship] = Field(..., description="A list of relationships between terms.")


# Classification P1/P2 (chap_6_semantic_annotation_rules_classification.ipynb)
class Classification(BaseModel):
    type: str = Field(..., description="Type of the rule (e.g., Party, Data, Activity)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level of the classification")
    explanation: str = Field(..., description="Explanation of why the classification was made")


class StatementClassification(BaseModel):
    doc_id: str = Field(..., description="Document ID associated with the statement")
    statement_id: str = Field(..., description="A provided string that identifies the statement. e.g., '1', 'Person'")
    statement_title: str = Field(..., description="The title of the statement")
    statement_text: str = Field(..., description="The statement to be classified")
    statement_sources: List[str] = Field(..., description="List of statement's source")
    classification: List[Classification] = Field(..., description="List of classifications with explanations")


class StatementClassifications(BaseModel):
    StatementClassifications: List[StatementClassification]


class SubClassification(BaseModel):
    subtype: str = Field(..., description="Subtype of the rule. The title of the section/subsection.")
    templates_ids: List[str] = Field(..., description="List of template IDs that matched the statement.")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level of the classification")
    explanation: str = Field(..., description="Explanation of why the classification was made")


class StatementSubClassification(BaseModel):
    doc_id: str = Field(..., description="Document ID associated with the statement")
    statement_id: str = Field(..., description="A provided string that identifies the statement. e.g., '1', 'Person'")
    statement_title: str = Field(..., description="The title of the statement")
    statement_text: str = Field(..., description="The statement to be classified")
    statement_sources: List[str] = Field(..., description="List of statement's source")
    classification: List[SubClassification] = Field(..., description="List of classifications with explanations")


class StatementSubClassifications(BaseModel):
    StatementSubClassifications: List[StatementSubClassification]


# Transformation (chap_6_nlp2sbvr_transform.ipynb)
class TransformedStatement(BaseModel):
    doc_id: str = Field(..., description="Document ID associated with the statement.")
    statement_id: str = Field(..., description="A provided string that identifies the statement. e.g., '1', 'Person'.")
    statement_title: str = Field(..., description="Title of the statement.")
    statement: str = Field(..., description="The statement to be transformed.")
    statement_sources: List[str] = Field(..., description="Sources of the statement.")
    templates_ids: List[str] = Field(..., description="List of template IDs.")
    transformed: str = Field(..., description="The transformed statement.")
    confidence: float = Field(..., description="Confidence of the transformation.")
    reason: str = Field(..., description="Reason for confidence score of the transformation.")


class TransformedStatements(BaseModel):
    TransformedStatements: List[TransformedStatement] = Field(..., description="List of transformed statements.")


# Judge (chap_7_validation_rules_transformation.ipynb)
class JudgeStatement(BaseModel):
    doc_id: str = Field(..., description="Document ID associated with the statement.")
    statement_id: str = Field(..., description="A provided string that identifies the statement. e.g., '1', 'Person'.")
    statement: str = Field(..., description="The statement to be transformed.")
    sources: List[str] = Field(..., description="Sources of the statement.")
    semscore: float = Field(..., description="just a copy from input semscore.")
    similarity_score: float = Field(..., description="Similarity score between the original and transformed sentences.")
    similarity_score_confidence: float = Field(..., description="Confidence score for the similarity score.")
    transformation_accuracy: float = Field(..., description="Accuracy score for the transformation.")
    grammar_syntax_accuracy: float = Field(..., description="Accuracy score for the grammar and syntax.")
    findings: List[str] = Field(..., description="List of findings.")


class JudgeStatements(BaseModel):
    JudgeStatements: List[JudgeStatement] = Field(..., description="List of judge statements.")


# --------------------------------------------------------------------------
# Frozen prompts (verbatim from the notebooks)
# --------------------------------------------------------------------------

SYSTEM_PROMPT_EXTRACT_P1 = """
You are tasked with extracting elements from a given legal document. Please follow these steps carefully and ensure all instructions are adhered to:

# Steps

1. **Summarize the document** to understand its purpose and use it to verify if all important terms,facts, fact types, and rules are identified in subsequent steps.

2. **Identify elements**:
   - **About the elements**:
     - **Fact**: A specific instance or statement that describes an event or condition without any directive element. Facts often involve relationships between terms or entities. Example: "John works for X Inc."
     - **Fact Type**: A general, abstract template that describes potential relationships between terms or entities, serving as a model for generating specific facts. Example: "Person works for Company."
     - **Operative Rule**: A statement that governs or constrains some aspect of the business, specifying what must be done or what is not allowed. Rules enforce compliance, limit possibilities, or prescribe specific behaviors in response to business situations. Operative rules (otherwise known as normative rules or prescriptive rules) state what must or must not happen in particular circumstances. Operative rules can be contravened: required information may be omitted, inappropriate information supplied, or an attempt may be made to perform a process that is prohibited. Example: "A customer must provide identification before opening an account."
     - **Term**: A word or a group of words that represents a specific concept, entity, or subject in a particular context.
     - Terms, Fact, Fact Type, and Operative Rule are statements that should allow only full compliance or full contravention; partial compliance is not possible. The presence of "or" or "and" often suggests the need to separate a statement into two.
   - **For each fact, fact type, or rule**:
     - **Extract the statement**: Identify the exact statement or phrase from the document representing the fact, fact type, or rule.
     - **Give a unique title to the statement**.
     - **Extract and classify Terms**:
       - **Extract all the terms involved in the statement**. Record the level of confidence in the extraction, ranging from 0 to 1, and provide a brief reason for the confidence score.
       - **Classify each term** as either **Common Noun** or **Proper Noun**.
       - If a Term contains nouns separated by "and," ",", or "or," split it into two or more terms. For example, "Principal office and place of business" should be split into "Principal office" and "Place of business".
     - **Extract Verb Symbols**: Identify verbs, verb phrases, or prepositions that connect the terms in the statement. Record the level of confidence in the extraction, ranging from 0 to 1, and provide a brief reason for the confidence score.
     - **Classification**: Classify the statement as either a **Fact**, **Fact Type**, or **Rule**.
     - **Confidence**: Record the level of confidence in the classification, ranging from 0 to 1.
     - **Reason**: Provide a brief reason for the classification score.
     - **Source**: Note the specific paragraph or section of the document where the statement is found (e.g., "(a)(1)", "(b)").

3. **Provide JSON Output**:
   - Format your answer as per the output example below.
   - **All values are optional**: Include as much information as is available based on the document.
   - **Do not include any additional text or explanation outside the JSON structure**.

**Output Example**:

```json
{
  "section": "§ 123.4-5",
  "elements": [
    {
      "id": 1,
      "title": "some title",
      "statement": "A person serves a non-resident investment adviser by furnishing the Commission with process, pleadings, or papers.",
      "terms": [
        {
          "term": "Person",
          "classification": "Common Noun",
          "confidence": 0.9,
          "reason": "The term is ..."
          "extract_confidence": 0.9,
          "extract_reason": "The term is ..."
        },
        {
          "term": "Non-resident investment adviser",
          "classification": "Common Noun",
          "confidence": 0.8,
          "reason": "The term is ...",
          "extract_confidence": 0.8,
          "extract_reason": "The term is ..."
        },
        ...
      ],
      "verb_symbols": ["serves", "by furnishing", "with"],
      "verb_symbols_extracted_confidence": [0.9, 0.8, 0.7],
      "verb_symbols_extracted_reason": ["The verb is ...", "The verb is ...", "The verb is ..."],
      "classification": "Fact Type",
      "confidence": 0.8,
      "reason": "The statement is ...",
      "sources": ["(a)"]
    },
    ...
  ]
}
```

# Notes
1. Level of Granularity: Extract and analyze every potential statement from the document;
2. Contextual Interpretation: Extract explicitly stated facts, fact types, and rules;
3. Scope of Terms: Classify every noun phrase as a term, even if it is peripheral to the main statement;
4. Verb Symbols: Include prepositions and auxiliary phrases;
5. Classification Nuances: Record the level of confidence range from 0 to 1;
6. Section Handling: Strictly tie every element to its section (e.g., § 275.0-7(a)(1));
7. Order of Presentation: Follow the sequence of the document strictly;
8. Edge Cases: Classify it with a lower level of confidence;
9. Output Preferences: Add timestamp and processing notes;
10. Formatting Precision: Ensure the JSON adheres strictly to a specific schema (e.g., for use in a system).
"""

SYSTEM_PROMPT_EXTRACT_P2 = """
You are tasked with extracting definitions and **relationships** of terms in the terms list searching a given legal document. Please follow these steps carefully and ensure all instructions are adhered to:

# Steps

1. **Summarize the document** to understand its purpose and use it to verify if all important terms, term definitions, facts, fact types, and rules are identified in subsequent steps.

2. **Define terms**:
  - For each term:
    - Search the entire document for the term's definition, explanation, or meaning. Also, look in the document summary.
    - If the definition is found, include it.
    - If the definition is not found in the document, use null.
    - Record the level of confidence in the definition, ranging from 0 to 1.
    - Explain the reason for the confidence level.

3. **isLocalScope**: Is there an indication that the definition is exclusive to this section? Example: "For purposes of this section...", "as described in this section", "as defined in this section". If yes answer only with true. Otherwise, the answer is false.

4. **Identify synonym relationships between terms**:
  - For each term in the terms list:
    - Compare it against other terms in the text to find synonyms.
    - Ensure both terms exist within the same document context.
  - List all valid synonym pairs identified.
  - Record the level of confidence in the synonym relationship, ranging from 0 to 1.
  - Explain the reason for the confidence level.

5. **Provide JSON Output**:
  - Format your answer as per the output example below.
  - **All values are optional**: Include as much information as is available based on the document.
  - **Do not include any additional text or explanation outside the JSON structure**.

**Output Example**:

```json
{
  "terms": [
    {
      "term": "Person",
      "definition": "A person is a person.",
      "confidence": 0.9,
      "reason": "The definition was ...",
      "isLocalScope": true,
      "local_scope_confidence": 0.9,
      "local_scope_reason": "The scope is ..."
    },
    {
      "term": "Capital",
      "definition": "The total assets of a person.",
      "confidence": 0.8,
      "reason": "The definition is ...",
      "isLocalScope": false,
      "local_scope_confidence": 0.9,
      "local_scope_reason": "The scope is ..."
    },
    ...
  ],
  "relationships": [
    {
      "term_1": "Person",
      "term_2": "Capital",
      "relationship": "Synonym",
      "confidence": 0.8,
      "reason": "The relationship is ...",
    },
    {
      "term_1": "Capital",
      "term_2": "Person",
      "relationship": "Synonym",
      "confidence": 0.5,
      "reason": "The relationship is ...",
    },
    ...
  ]
}
```
"""


def get_system_prompt_classify_p1():
    return """
You are an expert in SBVR (Semantics of Business Vocabulary and Business Rules).

You are working for regulatory bodies, auditors, or process managers.

You will be provided with a list of statements formatted as JSON.

Your task is to classify each statement into one or more Operative Rules types according to the given definitions.

# Steps

1. **Summarize statement**: Summarize the given statement to understand its structure and content.

2. **Classify statement**: Classify each Operative Rule statement into one or more of the provided rule types. The **Operative rules** govern actions or constraints that must or must not happen under certain conditions, such as Data Rules, Activity Rules, and Party Rules. types to classify are:

- **Party rules**: A "Party rule" is a type of operative rule that establishes distinctions or constraints involving parties or the roles they perform. To identify a party rule, it is important to recognize its defining characteristics, as these rules often specify who can carry out certain activities, access particular information, or hold specific responsibilities. Party rules may include restrictions on who is permitted to perform specific roles or processes. For example, a rule might state that a person can serve as the pilot in command only if they hold a current command endorsement. Additionally, these rules may enforce role separation to prevent conflicts of interest, such as a requirement that the cabin crew member verifying an aircraft door's disarmed status cannot be the same individual who initially disarmed it. In other cases, party rules may require role binding, ensuring continuity by stipulating that the consultant who signs a quality review report must be the one who conducted the review. Party rules can also govern information access, specifying who is authorized to view, create, or modify certain data. For instance, a rule might state that an employee’s leave record can only be accessed by the employee, their supervisor, or a human resources officer. Furthermore, responsibility rules fall under this category by defining accountability for specific actions or obligations, such as requiring the receiving parties in a property transfer to pay the associated stamp duty. These rules can be identified by linking actions or processes (predicates) to subjects, like roles or data, while applying conditions that qualify or limit their application.
- **Data rules**: A "Data rule" imposes constraints or requirements on the data used in transactions, records, or systems. Identifying a data rule involves analyzing its structure, purpose, and type, which include cardinality, content, and update rules. Data cardinality rules govern the presence and multiplicity of data items. These may include mandatory rules requiring data items, such as specifying at least one passenger name in a flight booking confirmation. They can also restrict data, such as ensuring a one-way flight booking does not include a return date, or enforce limits on the number of data instances in a transaction. Data content rules regulate the values within data items. Examples include value set rules, which require a data item to match one of a specified set of valid values, and range rules that constrain a data item's value to within a specific range. Equality rules ensure consistency between related data items, such as requiring that an origin city matches the corresponding booking request. Additionally, uniqueness constraints ensure that a data item's value does not duplicate within a dataset, and consistency rules maintain logical relationships between multiple data points. Lastly, data update rules constrain modifications to existing data. These rules may prohibit updates entirely, restrict the scope of permissible changes (such as maintaining valid state transitions), or enforce monotonic trends like numeric values that can only increase or decrease. To identify a data rule, it is essential to examine the specific constraints or requirements applied to data items, their interrelationships, and their contexts within a given system or transaction. Templates and formalized structures, can aid in distinguishing these rules effectively.
- **Activity rules**: An "Activity rule" is an operative rule designed to constrain the operation of business processes or activities. Identifying an activity rule involves understanding its subcategories, which define how activities are regulated or mandated. The primary types of activity rules are activity restriction rules, activity obligation rules, and process decision rules. Activity restriction rules are used to place limitations on when or under what conditions an activity can occur. For example, time-based restrictions may stipulate that online check-in for a flight can only occur within a specific time window, such as the 24 hours before departure. Similarly, exclusion period rules prohibit activities during certain times, such as a restriction on operating machinery during nighttime hours. Activity pre-condition rules ensure that an event must occur before another activity can take place, such as requiring passengers to complete a security screening before boarding. Activity obligation rules, on the other hand, specify activities that must be performed either within a maximum time after a triggering event or as soon as practical. For instance, acknowledging an order may be mandated to occur within 24 hours of receipt. These rules enforce timely action and compliance with operational requirements. Process decision rules determine actions in response to specific situations. These rules guide devices or processes, such as ensuring that a ticket barrier retains invalid tickets to prevent misuse. To identify an activity rule, it is essential to look for statements that define constraints, obligations, or decision-making criteria for activities. Such rules are often structured using specific templates that articulate the conditions, timeframes, or triggers associated with the activity. Recognizing these elements helps in categorizing activity rules effectively.

2. Assess a **confidence level** for each classification between 0 and 1. Assign confidence scores to each class for the given statement, ensuring that no two classes receive the same score. If one class is assigned a score (e.g., 0.6), the others must have distinct values that are either higher or lower. The scores should reflect the likelihood of each class while avoiding ties.

3. **Explain classification**: You also need to record a confidence level for each classification and provide an explanation for why the classification was made.

4. **Repeat for each statement**: Repeat the process for each statement in the list.

5. **Output format**: Your output must also be in JSON format. It should contain, for each statement:

- The `doc_id`
- The `statement_id`
- The `statement_title`
- The original `statement_text`
- The `statement_sources` of the statement
- A list of classifications (`classification`), each containing:
  - The `type` of the rule.
  - The `confidence` in your classification.
  - An `explanation` detailing why you made the classification decision.

Here is an example of the expected output:

```
[
    {
        "doc_id": "some doc id",
        "statement_id": "some id",
        "statement_title": "some title",
        "statement_text": "some text",
        "statement_sources": ["some source"],
        "classification": [
            {
                "type": "Activity rules",
                "confidence": 0.9,
                "explanation": "This statement defines ..."
            },
            {
                "type": "Party rules",
                "confidence": 0.2,
                "explanation": "There is little reference ..."
            },
            ...
        ]
    },
    {
        "doc_id": ...,
        "statement_id": ...,
        "statement_title": ...,
        "statement_text": ...,
        "statement_sources": ...,
        "classification": ...
    }
]
```

# Notes
- **Detail the Reasoning**: Make sure to provide explanations that justify why a particular rule type was chosen.
- **Confidence Values**: The confidence value should genuinely represent how strongly you believe the classification is correct, with 1 being an absolute match and 0 meaning unlikely.

Make sure that every statement is analyzed thoroughly, and the final justification for each classification is straightforward and adequately supports both the type choice and confidence level.
"""


def get_user_prompt_classify(rules_to_classify):
    return f"""
# Classification Task:

Analyze the following statements based on the above guidelines:

{json.dumps(rules_to_classify, indent=2)}
"""


def get_system_prompt_classify_p2(element_count, element_type, rule_type, statement_type):
    rule_information_provider = RuleInformationProvider(str(DATA_DIR))

    subclassification_text = rule_information_provider.get_classification_and_templates(f"{statement_type}")

    return f"""
You are an expert in **SBVR (Semantics of Business Vocabulary and Business Rules)**, working for regulatory bodies, auditors, or process managers.

Your task is to classify {element_count} {element_type}(s) from a provided list into one or more **{rule_type} Rule subtypes**, explain each classification in detail, and assign a confidence score ranging from 0 to 1.

# Approach:
Use the **{rule_type} Rule subtype definitions**, associated templates, and guidelines to analyze each statement thoroughly, ensuring accurate classification.

---

# Steps

1. **Classify the Statement**:
   - For each `statement_text`, determine its rule subtype according to the provided **{rule_type} Rule subtypes** and their corresponding templates.
   - Use the provided templates, definitions, and examples to match the statement to the correct subtype.
   - If the statement does not align with a high-level type, analyze the sublevels.
   - The subtype to be used starts with "subtype: <subtype name>".
   - You should assign a subtype and a template ID, make your best guess, justify your choice, and lower the confidence level if necessary.
   - Templates and examples help identify subtypes.

2. **Assign Confidence Level**:
   - Assign a confidence score between **0 and 1** for each classification:
     - **1** indicates a strong and clear match.
     - Lower scores reflect weaker matches due to ambiguities or partial alignment.
   - Consider both template alignment and the clarity of the statement's intent when assigning scores.
   - Assign confidence scores to each class for the given statement, ensuring that no two classes receive the same score. If one class is assigned a score (e.g., 0.6), the others must have distinct values that are either higher or lower. The scores should reflect the likelihood of each class while avoiding ties.

3. **Provide an Explanation**:
   - Provide a concise yet detailed explanation for the assigned classification.
   - Justify the classification by referencing:
     - Template structure.
     - Terminology used in the statement.
     - Specific conditions or context highlighted by the statement.
   - Explicitly map the elements of the statement (e.g., terms, qualifying clauses, verb phrases, conditional clauses, etc.) to template components.

---

# {rule_type} Rule subtypes

{subclassification_text}

---

# Definitions
- **attribute term**: A term that signifies a non-Boolean property of an entity class (or object class).
- **role term**: A term that signifies the role played by one of the participating parties or objects in a relationship: for example, employer and employee are role terms (with respect to the relationship whereby an organization employs a person), whereas organization and person are not role terms.
- **category attribute term**: A term is usually admin-defined, with some external inputs. They have unique labels (e.g., 'Cash') and may use internal codes. Boolean attributes indicate "Yes" or "No" responses, shown as checkboxes or "Y/N" fields.
- **quantitative attribute**: An attribute on which some arithmetic can be performed (e.g., addition, subtraction) and on which comparisons other than "=" and "<>" can be performed.
- **qualifying clause**: refines a rule's scope or specificity by limiting the subject or other terms to particular subsets or conditions (e.g., “for a return journey” or “that is current”).

# Output Format:

Each analyzed statement must be provided in JSON format. The structure for each statement is as follows:

```json
{{
    "doc_id": "The Document ID from the input",
    "statement_id": "The original statement ID",
    "statement_title": "The original statement_title",
    "statement_text": "The original statement_text",
    "statement_sources": "The original statement_sources",
    "classification": [
        {{
            "subtype": "Assigned rule subtype, use the title of the section/subsection (e.g., Activity time limit rules)",
            "templates_ids": ["Template ID that matched the statement."],
            "confidence": Confidence Score (0-1),
            "explanation": "Detailed explanation of why this classification was assigned."
        }}
    ]
}}
```

---

## Example Output:

```
[
    {{
        "doc_id": "some doc id",
        "statement_id": "some id",
        "statement_title": "some title",
        "statement_text": "some text",
        "statement_sources": ["some source"],
        "classification": [
            {{
                "subtype": "Some Subtype Title",
                "templates_ids": ["T123", "T456"],
                "confidence": 0.9,
                "explanation": "This statement ..."
            }},
            {{
                "subtype": "Another Subtype Title",
                "templates_ids": ["T789"],
                "confidence": 0.4,
                "explanation": "There are elements ..."
            }}
        ]
    }},
    {{
        "doc_id": "another doc id",
        "statement_id": "another id",
        "statement_title": "another title",
        "statement_text": "another text",
        "statement_sources": ["another source"],
        "classification": [
            {{
                "subtype": "Subtype Title",
                "templates_ids": ["T123"],
                "confidence": 0.7,
                "explanation": "The clause dictates ..."
            }}
        ]
    }},
    ...
]
```

---

# Additional Notes:
- **Multiple Classifications**:
   - A statement can have multiple classifications if it aligns with different subtypes. Justify each with appropriate confidence levels.
- **Cross-References**:
   - When a statement refers to another section (e.g., "(a)(1)"), incorporate the referenced section if it is provided or available. If unavailable, indicate this in the explanation and lower the confidence score.

---

"""


RULE_TEMPLATE_FORMULATION = """
# How to interpret the templates and subtemplates

Each formulation is expressed using a template, in which the various symbols have the following meanings:

1. Each item enclosed in "angle brackets" ("<" and ">") is a placeholder, in place of which any suitable text may be substituted. For example, any of the following may be substituted in place of <operative rule statement subject> (subtemplate):
    a. a term: for example, "flight booking request",
    b. a term followed by a qualifying clause: for example, "flight booking request for a one-way journey",
    c. a reference to a combination of items: for example, "combination of enrollment date and graduation date", with or without a qualifying clause,
    d. a reference to a set of items: for example, "set of passengers", with or without a qualifying clause.
2. Each pair of braces ("{" and "}") encloses a set of options (separated from each other by the bar symbol: "|"), one of which is included in the rule statement. For example,
3. If a pair of braces includes a bar symbol immediately before the closing brace, the null option is allowed: that is, you can, if necessary, include none of the options at that point in the rule statement.
4. Sets of options may be nested. For example, in each of the templates above
    a. a conditional clause may be included or omitted,
    b. if included, the conditional clause should be preceded by either "if" or "unless".
5. A further notation, introduced later in this section, uses square brackets to indicate that a syntactic element may be repeated indefinitely.
6. Any text not enclosed in either "angle brackets" or braces (i.e., "must", "not", "may", and "only") is included in every rule statement conforming to the relevant template.
"""


def get_system_prompt_transform(element_name, rule_template_formulation, rule_templates_subtemplates):
    statement_name = "definition" if element_name in ["Term", "Name"] else "statement"
    return f"""
Transform each given {element_name} {statement_name} into a structured format by matching it to the specified templates and subtemplates.

# Steps

1. **Summarize {statement_name}**: Summarize the given {element_name} {statement_name} to understand its structure and content.

2. **Use Template**:
   - For given expression, use the templates and subtemplates ({"Fact Type Form" if element_name in ["Fact Type", "Fact"] else "Rule Form"}) provided for transformation.
   - Determine the appropriate template or subtemplate based on the structure of the expression.

3. **Replace Placeholders**:
   - Substitute placeholders, such as `<term>`, `<verb phrase>`, `<conditional clause>`, etc., with suitable values as per the expression.
   - For terms and names, the statement_id is the term defined by the statement.

4. **Include Qualifying Details**:
   - Where placeholders, such as `<qualifying clause>`, require additional details (e.g., attributes or qualifiers to distinguish meaning), ensure that these are included appropriately as per the respective subtemplate.

5. **Transform into Structured Format**:
   - Once the transformation is complete, ensure it's in the correct template format.

6. **Output as Structured JSON**:
   - For every transformed expression generate a JSON object as per the specified output format.

7. **Review and Validate**:
   - Ensure accuracy in grammar and compliance with logical constructs when performing substitutions.
   - Ensure the generated JSON is in the correct template format.

8. **Assess the Transformation**:
   - Record the confidence level and reason for the confidence score in the JSON object.

{rule_template_formulation}

# Provided templates and subtemplates for transformation

{rule_templates_subtemplates}

# Output Format

[
    {{
      "doc_id": <doc_id>,
      "statement_id": <statement_id or signifier>,
      "statement_title": <statement_title>,
      "sources": [<source>],
      "statement": <statement or definition>,
      "templates_ids": [<templates_id>],
      "transformed": <transformed_statement>,
      "confidence": <confidence_level>,
      "reason": <reason_for_confidence>
    }},
    ...
]

- **`doc_id`**: A original identifier of the given document.
- **`statement_id or signifier`**: The original identifier of the given {statement_name}. e.g., '1', 'Person'".
- **`statement_title`**: The title of the given {statement_name}.
- **`sources`**: The original sources of the given {statement_name}.
- **`statement or definition`**: The original text of the given {statement_name}.
- **`templates_ids`**: The template(s) used for the transformation (e.g., T1, T2, etc.)
- **`transformed`**: The transformed statement according to template.
- **`confidence`**: The confidence level of the transformation range from 0 to 1.
- **`reason`**: The reason for the confidence score.

# Notes
- Use only the provided templates and subtemplates for transformation.
- If a placeholder within an expression is not applicable or optional, consider whether it should be omitted or replaced by a suitable value.
- Each expression may involve nested levels of substitution as indicated by the subtemplate hierarchy (e.g., a qualifying clause that contains sub-elements).
- Ensure accuracy in grammar and compliance with logical constructs when performing substitutions.
"""


def get_user_prompt_transform(element_name, rule):

    return f"""
# Here's the {element_name} {"definition" if element_name in ["Term", "Name"] else "statement"} you need to transform using template {rule.get("templates_ids")}.

{json.dumps(rule, indent=2)}
"""


def get_system_prompt_judge_sentence_similarity(template):
    return f"""
   # Task

   You're an expert in judging sentence similarity and transformation using a template.

   These criteria should support the evaluation process by verifying classification accuracy, template application, and transformation fidelity.

   Check the criteria and evaluate the output:

   1. **Similarity Score**
      - Given the statement or definition and transformed sentence (transformed), how similar are they from 0 to 1? And how confident are you about your estimation from 0 to 1?

   2. **Transformation Accuracy**
      - From 0 to 1, how does the transformed sentence (transformed) reflect the original sentence (statement or definition) with the structure and phrasing provided by the template and subtemplates?

   3. **Grammar and Syntax Accuracy**
      - How is the transformed sentence (transformed) grammatically correct and syntactically accurate from 0 to 1?

   # Output Format

   Record your evaluation in JSON format as follows:

   ```json
   {{
      "doc_id": "<Document ID>",
      "statement_id": "<Statement ID>",
      "sources": ["<source>"],
      "similarity_score": <Similarity score>,
      "similarity_score_confidence": <Confidence score>,
      "transformation_accuracy": <Transformation score>,
      "grammar_syntax_accuracy": <Grammar score>,
      "findings": ["<Things found during the evaluation and worth to be mentioned>",
                  "<other things to mention>"
                  ],
      "semscore": <original semscore>
   }}
   ```

   # Input example

   ```json
   {{
      "doc_id": <Document ID>,
      "statement_id": <Statement ID>,
      "statement or definition": <original sentence>,
      "sources": [<source>],
      "terms": [
         {{"term": <signifier>, "classification": <Proper or Common Noun>}},
         ...
      ],
      "verb_symbols": <verbs or phrasal verbs>,
      "element_name": <name of element: Name, Term, Fact, Fact Type, Operative Rule>,
      "transformed": <transformed sentence>,
      "type": <type of element: Definitional, Activity, Party, Data>,
      "subtype": <subtype of element>,
      "templates_ids": ["T8"],
      "semscore": <semscore>
   }}
   ```

   # Templates and Subtemplates

   {template}
   """


def get_user_prompt_judge_sentence_similarity(element_name, rule):
    return f"""
# rule data for an element: {element_name}

{json.dumps(rule, indent=2)}
    """


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def extract_unique_terms(document: ElementsDocumentModel) -> List[str]:
    unique_terms: Set[str] = set()
    for element in document.elements:
        for term_info in element.terms:
            unique_terms.add(term_info.term)
    return list(unique_terms)


def batch(iterable, max_batch_size):
    iterator = iter(iterable)
    while True:
        batch_list = list(islice(iterator, max_batch_size))
        if not batch_list:
            break
        yield batch_list


def llm(system_prompt, user_prompt, document_model):
    from llm_query.main import query_instruct_llm

    response, completion, elapse_time = query_instruct_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        document_model=document_model,
        llm_model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response, completion, elapse_time


def get_embedding(text, model=EMBEDDING_MODEL):
    from openai import OpenAI

    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def compare_sentences(sentence1, sentence2):
    import numpy as np

    e1 = np.array(get_embedding(sentence1))
    e2 = np.array(get_embedding(sentence2))
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def grounding_score(fragment: str, source: str) -> float:
    """
    Best fuzzy alignment of `fragment` against a sliding window of `source`.
    1.0 means the fragment appears (near-)verbatim in the source text.
    Reference-free: requires only the source section, no gold standard.
    """
    frag = normalize_text(fragment)
    src = normalize_text(source)
    if not frag:
        return 1.0
    if frag in src:
        return 1.0
    frag_tokens = frag.split()
    src_tokens = src.split()
    n = len(frag_tokens)
    if n == 0 or len(src_tokens) == 0:
        return 0.0
    best = 0.0
    step = max(1, n // 4)
    for start in range(0, max(1, len(src_tokens) - n + 1), step):
        window = " ".join(src_tokens[start:start + n + max(2, n // 4)])
        ratio = difflib.SequenceMatcher(None, frag, window).ratio()
        if ratio > best:
            best = ratio
        if best > 0.995:
            break
    return round(best, 4)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def usage_from_completions(completions):
    tokens_in = sum(c.get("usage", {}).get("prompt_tokens", 0) for c in completions)
    tokens_out = sum(c.get("usage", {}).get("completion_tokens", 0) for c in completions)
    return tokens_in, tokens_out


# --------------------------------------------------------------------------
# Pipeline stages (mirroring the notebooks, single pass, no correction)
# --------------------------------------------------------------------------

def checkpoint_roundtrip(manager):
    """Persist and restore so pydantic content becomes plain dicts,
    exactly as happens between notebooks in the original flow."""
    save_checkpoint(filename=str(CHECKPOINT_FILE), manager=manager)
    return restore_checkpoint(filename=str(CHECKPOINT_FILE))


def stage_extraction(manager, section_text, usage):
    if manager.retrieve_document(f"{SECTION_ID}_P2", "llm_response"):
        print("[skip] extraction already in checkpoint")
        return manager

    manager.add_document(Document(id=SECTION_ID, type="section", content=section_text))

    user_prompt = f"""
# Document

{section_text}
    """
    print("[run] extraction P1 ...")
    response_part_1, completion_1, _ = llm(SYSTEM_PROMPT_EXTRACT_P1, user_prompt, ElementsDocumentModel)
    usage.append(completion_1.dict())
    manager.add_document(Document(
        id=f"{SECTION_ID}_P1", type="llm_response", content=response_part_1,
        completions=[completion_1.dict()],
    ))

    terms_list_part_1 = extract_unique_terms(response_part_1)
    user_prompt = f"""
# Terms list

{terms_list_part_1}

# Document
{section_text}
    """
    print("[run] extraction P2 ...")
    response_part_2, completion_2, _ = llm(SYSTEM_PROMPT_EXTRACT_P2, user_prompt, TermsDocumentModel)
    usage.append(completion_2.dict())
    manager.add_document(Document(
        id=f"{SECTION_ID}_P2", type="llm_response", content=response_part_2,
        completions=[completion_2.dict()],
    ))

    return checkpoint_roundtrip(manager)


def stage_classification(manager, usage):
    # ---- P1: operative rules top-level classification
    processor = DocumentProcessor(manager)
    rules_to_classify_p1 = [
        {
            "doc_id": item["doc_id"],
            "statement_id": item["statement_id"],
            "statement_title": item["statement_title"],
            "statement_sources": item["sources"],
            "statement_text": item["statement"],
        }
        for item in processor.get_rules()
    ]
    print(f"[run] classification P1 for {len(rules_to_classify_p1)} operative rules ...")
    if manager.retrieve_document("classify_P1", "llm_response_classification"):
        print("[skip] classify_P1 already in checkpoint")
    elif rules_to_classify_p1:
        response_classify_p1, completion_1, _ = llm(
            get_system_prompt_classify_p1(),
            get_user_prompt_classify(rules_to_classify_p1),
            StatementClassifications,
        )
        usage.append(completion_1.dict())
        manager.add_document(Document(
            id="classify_P1", type="llm_response_classification",
            content=response_classify_p1.StatementClassifications,
            completions=[completion_1.dict()],
        ))
    manager = checkpoint_roundtrip(manager)

    # ---- P2: subtype classification + templates
    def classify_p2(element_type, rule_type, items, part, element_name):
        doc_id = f"classify_{part}_{element_name.replace(' ', '_')}"
        if manager.retrieve_document(doc_id, "llm_response_classification"):
            print(f"[skip] {doc_id} already in checkpoint")
            return
        user_prompts, system_prompts = [], []
        for batch_num, batch_rules in enumerate(batch(items, 5)):
            grouped_data = defaultdict(list)
            for item in batch_rules:
                grouped_data[item["statement_type"]].append(item)
            for statement_type, grouped_items in grouped_data.items():
                system_prompts.append(get_system_prompt_classify_p2(
                    len(batch_rules), element_type, rule_type, statement_type))
                user_prompts.append(get_user_prompt_classify(grouped_items))

        all_responses = []
        completion = None
        for index, (up, sp) in enumerate(zip(user_prompts, system_prompts), start=1):
            print(f"[run] classification {part} {element_name} prompt {index}/{len(user_prompts)} ...")
            resp, completion, _ = llm(sp, up, StatementSubClassifications)
            usage.append(completion.dict())
            all_responses.extend(resp.StatementSubClassifications)
        if all_responses:
            manager.add_document(Document(
                id=f"classify_{part}_{element_name.replace(' ', '_')}",
                type="llm_response_classification",
                content=all_responses,
                completions=[completion.dict()],
            ))

    # Operative rules P2 (statement_type = P1 top-level type)
    processor = DocumentProcessor(manager)
    rules_to_classify_p2 = [
        {
            "doc_id": item["doc_id"],
            "statement_id": item["statement_id"],
            "statement_title": item["statement_title"],
            "statement_sources": item["sources"],
            "statement_text": item["statement"],
            "statement_type": item["type"],
        }
        for item in processor.get_rules()
        if item.get("type")
    ]
    classify_p2("operative rule", "Operative", rules_to_classify_p2, "P2_Operative", "rules")
    manager = checkpoint_roundtrip(manager)

    # Terms P2
    processor = DocumentProcessor(manager, merge=True)
    terms_to_classify_p2 = [
        {
            "doc_id": item["doc_id"],
            "statement_id": item["statement_id"],
            "statement_sources": item["sources"],
            "statement_text": item["definition"],
            "statement_type": "Definitional rules",
        }
        for item in processor.get_terms(definition_filter="non_null")
    ]
    classify_p2("term", "Definitional", terms_to_classify_p2, "P2_Definitional", "terms")
    manager = checkpoint_roundtrip(manager)

    # Names P2
    processor = DocumentProcessor(manager, merge=True)
    names_to_classify_p2 = [
        {
            "doc_id": item["doc_id"],
            "statement_id": item["statement_id"],
            "statement_sources": item["sources"],
            "statement_text": item["definition"],
            "statement_type": "Definitional rules",
        }
        for item in processor.get_names(definition_filter="non_null")
    ]
    classify_p2("name", "Definitional", names_to_classify_p2, "P2_Definitional", "names")
    manager = checkpoint_roundtrip(manager)

    # Facts P2
    processor = DocumentProcessor(manager)
    facts_to_classify_p2 = [
        {
            "doc_id": item["doc_id"],
            "statement_id": item["statement_id"],
            "statement_title": item["statement_title"],
            "statement_sources": item["sources"],
            "statement_text": item["statement"],
            "statement_type": "Definitional rules",
        }
        for item in processor.get_facts()
    ]
    classify_p2("fact type", "Definitional", facts_to_classify_p2, "P2_Definitional", "facts")
    return checkpoint_roundtrip(manager)


def get_prompts_for_rule(rules, rule_template_formulation, data_dir):
    rule_template_provider = RulesTemplateProvider(data_dir)

    system_prompts = []
    user_prompts = []
    element_name = None
    skipped = 0

    for rule in rules:
        element_name = rule.get("element_name")

        if not rule.get("templates_ids"):
            skipped += 1
            continue

        if element_name == ["Term", "Name"]:
            statement_key = "definition"
            statement_id_key = "signifier"
        else:
            statement_key = "statement"
            statement_id_key = "statement_id"

        input_rule = {
            "doc_id": rule["doc_id"],
            f"{statement_id_key}": rule["statement_id"],
            "statement_title": rule.get("statement_title", rule.get("statement_id")),
            "sources": rule["sources"],
            f"{statement_key}": rule.get("statement", rule.get("definition")),
            "templates_ids": rule["templates_ids"],
        }
        user_prompts.append(get_user_prompt_transform(element_name, input_rule))
        rule_templates_subtemplates = rule_template_provider.get_rules_template(rule["templates_ids"])
        system_prompts.append(get_system_prompt_transform(
            element_name, rule_template_formulation, rule_templates_subtemplates))

    if skipped:
        print(f"[warn] {skipped} {element_name} element(s) without templates_ids skipped (unclassified)")

    return system_prompts, user_prompts, element_name


def transform_statement(element_name, user_prompts, system_prompts, manager, usage):
    if not user_prompts or not system_prompts:
        print(f"[info] no prompts for {element_name}s")
        return []
    all_responses = []
    completion = None
    for index, (up, sp) in enumerate(zip(user_prompts, system_prompts), start=1):
        print(f"[run] transform {element_name} {index}/{len(user_prompts)} ...")
        resp, completion, _ = llm(sp, up, TransformedStatements)
        usage.append(completion.dict())
        all_responses.extend(resp.TransformedStatements)
    manager.add_document(Document(
        id=f"transform_{element_name.replace(' ', '_')}s",
        type="llm_response_transform",
        content=all_responses,
        completions=[completion.dict()],
    ))
    return all_responses


def stage_transformation(manager, usage):
    processor = DocumentProcessor(manager, merge=True)
    pred_operative_rules = processor.get_rules()
    pred_facts = processor.get_facts()
    pred_terms = processor.get_terms(definition_filter="non_null")
    pred_names = processor.get_names(definition_filter="non_null")

    for rules in [pred_operative_rules, pred_facts, pred_terms, pred_names]:
        sp, up, element_name = get_prompts_for_rule(rules, RULE_TEMPLATE_FORMULATION, str(DATA_DIR))
        if element_name and manager.retrieve_document(
                f"transform_{element_name.replace(' ', '_')}s", "llm_response_transform"):
            print(f"[skip] transform_{element_name.replace(' ', '_')}s already in checkpoint")
            continue
        transform_statement(element_name, up, sp, manager, usage)
        manager = checkpoint_roundtrip(manager)

    return manager


def collect_transformed_elements(manager):
    """Merge processor elements with their transformations (as the validation
    notebooks do via process_transformed_elements)."""
    processor = DocumentProcessor(manager, merge=True)
    groups = {
        "Operative Rules": processor.get_rules(),
        "Fact Types": processor.get_facts(),
        "Terms": processor.get_terms(definition_filter="non_null"),
        "Names": processor.get_names(definition_filter="non_null"),
    }
    # keep only elements that have a transformation
    for key in groups:
        groups[key] = [e for e in groups[key] if e.get("transformed")]
    return groups


def stage_semscore(manager, results_path):
    if results_path.exists():
        print("[skip] semscore already computed")
        return json.load(open(results_path, encoding="utf-8"))

    groups = collect_transformed_elements(manager)
    semscores = {}
    for group_name, items in groups.items():
        for item in items:
            original_sentence = f'{item.get("statement_id")}: {item.get("statement", item.get("definition"))}'
            transformed_sentence = item.get("transformed")
            key = f'{group_name}|{item.get("doc_id")}|{item.get("statement_id")}'
            print(f"[run] semscore {key} ...")
            semscores[key] = compare_sentences(original_sentence, transformed_sentence)
    save_json(results_path, semscores)
    return semscores


def stage_judge(manager, semscores, usage, results_path):
    if results_path.exists():
        print("[skip] judge already computed")
        return json.load(open(results_path, encoding="utf-8"))

    rule_template_provider = RulesTemplateProvider(str(DATA_DIR))
    groups = collect_transformed_elements(manager)
    judged = []
    for group_name, items in groups.items():
        for item in items:
            rule = dict(item)
            for key in ["explanation", "confidence", "subtype_confidence", "subtype_explanation",
                        "type_confidence", "type_explanation"]:
                rule.pop(key, None)
            skey = f'{group_name}|{item.get("doc_id")}|{item.get("statement_id")}'
            rule["semscore"] = semscores.get(skey, 0.0)
            templates_ids = rule.get("templates_ids") or []
            rule_templates_subtemplates = rule_template_provider.get_rules_template(templates_ids)
            system_prompt = get_system_prompt_judge_sentence_similarity(rule_templates_subtemplates)
            user_prompt = get_user_prompt_judge_sentence_similarity(rule.get("element_name"), rule)
            print(f"[run] judge {skey} ...")
            resp, completion, _ = llm(system_prompt, user_prompt, JudgeStatements)
            usage.append(completion.dict())
            for js in resp.JudgeStatements:
                record = js.dict()
                record["group"] = group_name
                record["semscore_input"] = rule["semscore"]
                record["transformed"] = rule.get("transformed")
                judged.append(record)
    save_json(results_path, judged)
    return judged


def stage_grounding(manager, section_text, results_path):
    if results_path.exists():
        print("[skip] grounding already computed")
        return json.load(open(results_path, encoding="utf-8"))

    processor = DocumentProcessor(manager, merge=True)
    records = []
    for group_name, items, text_key in [
        ("Operative Rules", processor.get_rules(), "statement"),
        ("Fact Types", processor.get_facts(), "statement"),
        ("Terms", processor.get_terms(definition_filter="non_null"), "definition"),
        ("Names", processor.get_names(definition_filter="non_null"), "definition"),
    ]:
        for item in items:
            fragment = item.get(text_key) or ""
            records.append({
                "group": group_name,
                "doc_id": item.get("doc_id"),
                "statement_id": item.get("statement_id"),
                "text_key": text_key,
                "fragment": fragment,
                "grounding_score": grounding_score(fragment, section_text),
            })
    save_json(results_path, records)
    return records


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        f"share_ge_{ACCEPTANCE_THRESHOLD}": round(
            sum(1 for v in values if v >= ACCEPTANCE_THRESHOLD) / len(values), 4),
    }


def write_report(manager, grounding_records, semscores, judged, usage):
    processor = DocumentProcessor(manager, merge=True)
    counts = {
        "operative_rules": len(processor.get_rules()),
        "fact_types": len(processor.get_facts()),
        "terms_with_definition": len(processor.get_terms(definition_filter="non_null")),
        "names_with_definition": len(processor.get_names(definition_filter="non_null")),
    }

    grounding_by_group = defaultdict(list)
    for r in grounding_records:
        grounding_by_group[r["group"]].append(r["grounding_score"])

    judged_by_group = defaultdict(list)
    for r in judged:
        judged_by_group[r["group"]].append(r)

    tokens_in, tokens_out = usage_from_completions(usage)

    indicators = {
        "section": SECTION_ID,
        "design": "frozen pipeline, single run, no gold-standard correction at any checkpoint",
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "counts": counts,
        "grounding_check": {g: stats(v) for g, v in grounding_by_group.items()},
        "grounding_check_overall": stats([r["grounding_score"] for r in grounding_records]),
        "semscore_overall": stats(list(semscores.values())),
        "llm_judge": {
            g: {
                "similarity_score": stats([r["similarity_score"] for r in v]),
                "transformation_accuracy": stats([r["transformation_accuracy"] for r in v]),
                "grammar_syntax_accuracy": stats([r["grammar_syntax_accuracy"] for r in v]),
            }
            for g, v in judged_by_group.items()
        },
        "llm_judge_overall": {
            "similarity_score": stats([r["similarity_score"] for r in judged]),
            "transformation_accuracy": stats([r["transformation_accuracy"] for r in judged]),
            "grammar_syntax_accuracy": stats([r["grammar_syntax_accuracy"] for r in judged]),
        },
        "flagged_below_threshold": [
            {
                "group": r["group"],
                "doc_id": r["doc_id"],
                "statement_id": r["statement_id"],
                "similarity_score": r["similarity_score"],
                "findings": r["findings"],
            }
            for r in judged if r["similarity_score"] < ACCEPTANCE_THRESHOLD
        ],
        "tokens": {"input": tokens_in, "output": tokens_out},
    }
    save_json(OUT_DIR / "indicators.json", indicators)

    lines = []
    lines.append("# Held-out generalization probe: 17 CFR " + SECTION_ID)
    lines.append("")
    lines.append("Design: frozen pipeline (prompts, gpt-4o, temperature 0), executed once, "
                 "with no gold-standard correction at any checkpoint. All indicators are "
                 "reference-free: they require only the source section, not an annotated reference.")
    lines.append("")
    lines.append("## Extracted elements (uncorrected)")
    lines.append("")
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Indicator 1: grounding (hallucination) check")
    lines.append("")
    lines.append("Best fuzzy alignment of each extracted statement/definition against the source text "
                 "(1.0 = present near-verbatim).")
    lines.append("")
    lines.append("| Group | n | mean | median | min | share >= 0.8 |")
    lines.append("|---|---|---|---|---|---|")
    for g, v in grounding_by_group.items():
        s = stats(v)
        lines.append(f"| {g} | {s['n']} | {s['mean']} | {s['median']} | {s['min']} | {s[f'share_ge_{ACCEPTANCE_THRESHOLD}']} |")
    so = indicators["grounding_check_overall"]
    lines.append(f"| **Overall** | {so['n']} | {so['mean']} | {so['median']} | {so['min']} | {so[f'share_ge_{ACCEPTANCE_THRESHOLD}']} |")
    lines.append("")
    lines.append("## Indicator 2: SemScore (original vs. transformed, embedding cosine)")
    lines.append("")
    s = indicators["semscore_overall"]
    if s["n"]:
        lines.append(f"n = {s['n']}, mean = {s['mean']}, median = {s['median']}, min = {s['min']}, "
                     f"share >= {ACCEPTANCE_THRESHOLD} = {s[f'share_ge_{ACCEPTANCE_THRESHOLD}']}")
    lines.append("")
    lines.append("## Indicator 3: LLM-as-a-Judge (against source statement and templates)")
    lines.append("")
    lines.append("| Group | n | similarity (mean) | transformation acc. (mean) | grammar (mean) |")
    lines.append("|---|---|---|---|---|")
    for g, v in judged_by_group.items():
        sim = stats([r["similarity_score"] for r in v])
        acc = stats([r["transformation_accuracy"] for r in v])
        gra = stats([r["grammar_syntax_accuracy"] for r in v])
        lines.append(f"| {g} | {sim['n']} | {sim['mean']} | {acc['mean']} | {gra['mean']} |")
    ov = indicators["llm_judge_overall"]
    if ov["similarity_score"].get("n"):
        lines.append(f"| **Overall** | {ov['similarity_score']['n']} | {ov['similarity_score']['mean']} | "
                     f"{ov['transformation_accuracy']['mean']} | {ov['grammar_syntax_accuracy']['mean']} |")
    lines.append("")
    nflag = len(indicators["flagged_below_threshold"])
    lines.append(f"Statements below the acceptance threshold ({ACCEPTANCE_THRESHOLD}), flagged for SME review: {nflag} "
                 f"(details in indicators.json).")
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Input tokens: {tokens_in}; output tokens: {tokens_out} (gpt-4o).")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Single run (n=1): no dispersion estimate, unlike the ten-run protocol of the main experiment.")
    lines.append("- LLM-as-a-Judge and SemScore measure semantic preservation and template conformity, "
                 "not extraction recall or classification correctness; those require an annotated reference "
                 "and remain future work.")
    lines.append("- The judge shares the underlying model family with the pipeline; self-preference bias "
                 "cannot be excluded. Read the probe as indicative, not as validation.")

    report_path = OUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(json.dumps(indicators["llm_judge_overall"], indent=2))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def selftest():
    print("Python:", sys.version)
    print("Repo src:", SRC_DIR, "exists:", SRC_DIR.exists())
    print("Data dir:", DATA_DIR, "exists:", DATA_DIR.exists())
    for f in ["witt_templates.yaml", "witt_examples.yaml", "classify_subtypes.yaml"]:
        print(f"  data/{f}:", (DATA_DIR / f).exists())
    print("Section file:", SECTION_FILE.exists())
    section_text = SECTION_FILE.read_text(encoding="utf-8")
    print("Section length (words):", len(section_text.split()))
    # prompt construction
    p = get_system_prompt_classify_p2(3, "term", "Definitional", "Definitional rules")
    print("classify_p2 prompt OK, length:", len(p))
    rtp = RulesTemplateProvider(str(DATA_DIR))
    t = rtp.get_rules_template(["T1"])
    print("templates provider OK, T1 length:", len(str(t)))
    g = grounding_score("family office means a company", section_text)
    print("grounding check OK, sample score:", g)
    print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))
    print("Selftest finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="check setup without API calls")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    # Load OPENAI_API_KEY from the repo-root .env if not already set
    if not os.environ.get("OPENAI_API_KEY"):
        env_file = REPO_CODE_DIR.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Aborting before any stage runs.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    section_text = SECTION_FILE.read_text(encoding="utf-8")
    usage = []

    manager = restore_checkpoint(filename=str(CHECKPOINT_FILE))
    manager = stage_extraction(manager, section_text, usage)
    manager = stage_classification(manager, usage)
    manager = stage_transformation(manager, usage)

    grounding_records = stage_grounding(manager, section_text, OUT_DIR / "grounding.json")
    semscores = stage_semscore(manager, OUT_DIR / "semscore.json")
    judged = stage_judge(manager, semscores, usage, OUT_DIR / "judge.json")

    save_json(OUT_DIR / "usage.json", usage)
    write_report(manager, grounding_records, semscores, judged, usage)


if __name__ == "__main__":
    main()
