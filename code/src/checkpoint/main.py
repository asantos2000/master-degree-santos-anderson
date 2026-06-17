
from typing import List, Dict, Optional, Any, Tuple, Set
from pydantic import BaseModel, Field
import logging
import json
from json import JSONDecodeError
from pathlib import Path
import re
import unicodedata

# Set up basic logging configuration for the checkpoint module
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or another level as needed
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_set_to_list(data: Any) -> Any:
    """
    Recursively converts sets to lists in the data structure.

    Args:
        data (Any): The data structure to process, which can be a dict, list, set, or other types.

    Returns:
        Any: The data structure with all sets converted to lists.
    """
    if isinstance(data, dict):
        return {key: convert_set_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_set_to_list(item) for item in data]
    elif isinstance(data, set):
        return list(data)
    else:
        return data


def normalize_str(s: str) -> str:
    """
    Normalize a string using Unicode normalization to ensure consistent representation.

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    return unicodedata.normalize("NFKD", s).strip()


# Define a model for the Document
class Document(BaseModel):
    id: str
    type: str  # New field to represent the type of the document
    content: Any  # Content can be any data type: list, dict, string, etc.
    elapsed_times: Optional[list[float]] = None  # Optional field for elapsed time
    completions: Optional[list[Dict]] = None  # Optional field for completion status

# Define the DocumentManager class
class DocumentManager(BaseModel):
    documents: Dict[Tuple[str, str], Document] = Field(
        default_factory=dict
    )  # Keys are tuples (id, type)

    def add_document(self, doc: Document) -> None:
        """
        Adds a document to the manager.

        Args:
            doc (Document): The document to add.
        """
        # Normalize the document ID and type to avoid inconsistencies
        normalized_id = normalize_str(doc.id)
        normalized_type = normalize_str(doc.type)

        key = (normalized_id, normalized_type)

        # Debug logging for added document keys
        logger.debug(f"Adding document with normalized key: {key}")

        self.documents[key] = doc

    def retrieve_document(self, doc_id: str, doc_type: str) -> Optional[Document]:
        """
        Retrieves a document by its id and type.

        Args:
            doc_id (str): The ID of the document.
            doc_type (str): The type of the document.

        Returns:
            Optional[Document]: The retrieved document, or None if not found.
        """
        # Normalize the identifiers before using them to access the dictionary
        normalized_id = normalize_str(doc_id)
        normalized_type = normalize_str(doc_type)

        key = (normalized_id, normalized_type)

        # Debug logging for retrieval attempt
        logger.debug(f"Retrieving document with key: {key}")

        return self.documents.get(key)

    def list_document_ids(self, doc_type: Optional[str] = None) -> List[str]:
        """
        Lists all document ids, optionally filtered by type.

        Args:
            doc_type (Optional[str], optional): The type of documents to list. Defaults to None.

        Returns:
            List[str]: A list of document ids.
        """
        if doc_type:
            normalized_type = normalize_str(doc_type)
            logger.debug(f"Listing documents filtered by type: {normalized_type}")
            return [
                doc_id
                for (doc_id, d_type) in self.documents.keys()
                if d_type == normalized_type
            ]
        else:
            logger.debug("Listing all documents without type filter.")
            return [doc_id for (doc_id, _) in self.documents.keys()]

    def exclude_document(self, doc_id: str, doc_type: str) -> None:
        """
        Excludes a document by its id and type.

        Args:
            doc_id (str): The ID of the document to exclude.
            doc_type (str): The type of the document.
        """
        # Normalize identifiers before attempting exclusion
        normalized_id = normalize_str(doc_id)
        normalized_type = normalize_str(doc_type)

        key = (normalized_id, normalized_type)

        if key in self.documents:
            logger.debug(f"Excluding document with key: {key}")
            del self.documents[key]

    def persist_to_file(self, filename: str) -> None:
        """
        Persists the current state to a file, converting tuple keys to strings and sets to lists.

        Args:
            filename (str): The filename to save the documents.
        """
        serializable_documents = {
            f"{doc_id}|{doc_type}": convert_set_to_list(doc.dict())
            for (doc_id, doc_type), doc in self.documents.items()
        }
        with open(filename, "w") as file:
            json.dump(serializable_documents, file, indent=4)
        logger.info(f"DocumentManager state persisted to file: {filename}")

    @classmethod
    def restore_from_file(cls, filename: str) -> "DocumentManager":
        """
        Restores the state from a file, converting string keys back to tuples.

        Args:
            filename (str): The filename to restore the documents from.

        Returns:
            DocumentManager: The restored DocumentManager instance.
        """
        with open(filename, "r") as file:
            data = json.load(file)
            documents = {
                (doc_id.split("|")[0], doc_id.split("|")[1]): Document(**doc_data)
                for doc_id, doc_data in data.items()
            }
            logger.info(f"DocumentManager restored from file: {filename}")
            return cls(documents=documents)


def restore_checkpoint(filename: Optional[str]) -> DocumentManager:
    """
    Restores the document manager from a checkpoint file.

    Args:
        filename (str, optional): The path to the checkpoint file. Defaults to DEFAULT_CHECKPOINT_FILE.

    Returns:
        DocumentManager: The restored DocumentManager instance.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.

    See Also:
        - Reset the values delete the documents.json file and run: manager = DocumentManager()
        - Restore the state from the documents.json file, run: DocumentManager.restore_from_file("documents.json")
        - Exclue a document: manager.exclude_document(doc_id="§ 275.0-2", doc_type="section")
        - List documents: manager.list_document_ids(doc_type="section")
        - Get a document: manager.retrieve_document(doc_id=doc, doc_type="section")
    """

    try:
        restored_docs = DocumentManager.restore_from_file(filename)
        logger.info(f"Checkpoint restored from {filename}.")
    except (FileNotFoundError, JSONDecodeError):
        restored_docs = DocumentManager()
        logger.error(
            f"Checkpoint file '{filename}' not found or is empty, initializing new checkpoint."
        )
    return restored_docs


def save_checkpoint(filename: Optional[str], manager: DocumentManager) -> None:
    """
    Saves the current state of the DocumentManager to a checkpoint file.

    Args:
        manager (DocumentManager): The DocumentManager instance to save.

    Raises:
        Exception: If there is an error saving the checkpoint.
    """
    try:
        manager.persist_to_file(filename=filename)
        logger.info("Checkpoint saved.")
    except FileNotFoundError:
        logger.error(
            "Error saving checkpoint. Check the directory path and permissions."
        )


def get_all_checkpoints(checkpoint_dir, prefix="documents", extension="json"):
    managers = []

    path = Path(checkpoint_dir)

    path.mkdir(parents=True, exist_ok=True)

    files = list(path.glob(f"{prefix}-*.{extension}"))
    file_info_list = []

    pattern = re.compile(rf"^{prefix}-(\d{{4}}-\d{{2}}-\d{{2}})-(\d+)\.{extension}$")
    for filepath in files:
        match = pattern.match(filepath.name)
        if match:
            date_str = match.group(1)
            number = int(match.group(2))
            file_info_list.append(
                {"filename": filepath.name, "date": date_str, "number": number}
            )

            logger.debug(f"filepath: {filepath}")
            managers.append(DocumentManager.restore_from_file(filepath))

    return managers, file_info_list


def get_elements_from_checkpoints(checkpoint_dir, merge=True, filter="non_null"):
    managers, file_info_list = get_all_checkpoints(checkpoint_dir)

    logger.debug(file_info_list)

    pred_operative_rules = []
    pred_facts = []
    pred_terms = []
    pred_names = []
    pred_files = []

    for manager, file_info in zip(managers, file_info_list):
        # Process documents
        processor = DocumentProcessor(manager, merge=merge)

        # Access processed data
        # unique_terms = processor.get_unique_terms()
        # unique_names = processor.get_unique_names()
        pred_operative_rules += processor.get_rules()
        pred_facts += processor.get_facts()
        pred_terms += processor.get_terms(definition_filter=filter)
        pred_names += processor.get_names(definition_filter=filter)
        pred_files.append(file_info)

    logger.debug(f"Rules: {pred_operative_rules}")
    logger.debug(f"Facts: {pred_facts}")
    logger.debug(f"Terms: {pred_terms}")
    logger.debug(f"Names: {pred_names}")
    logger.info(f"Rules to evaluate: {len(pred_operative_rules)}")
    logger.info(f"Facts to evaluate: {len(pred_facts)}")
    logger.info(f"Terms to evaluate: {len(pred_terms)}")
    logger.info(f"Names to evaluate: {len(pred_names)}")

    return pred_operative_rules, pred_facts, pred_terms, pred_names, pred_files


def get_true_table_keys():
    return [
        "classify_P1|true_table",
        "classify_P2_Definitional_facts|true_table",
        "classify_P2_Definitional_names|true_table",
        "classify_P2_Definitional_terms|true_table",
        "classify_P2_Operative_rules|true_table",
    ]


def get_elements_from_true_tables(data_dir):
    true_table_file = f"{data_dir}/documents_true_table.json"
    true_table_keys = get_true_table_keys()

    true_operative_rules_p1 = []
    true_facts_p2 = []
    true_names_p2 = []
    true_terms_p2 = []
    true_operative_rules_p2 = []

    manager_true_elements = restore_checkpoint(true_table_file)

    for key in true_table_keys:
        match key:
            case "classify_P1|true_table":
                true_operative_rules_p1 = manager_true_elements.retrieve_document(
                    "classify_P1", "true_table"
                ).content
                logger.debug(f"P1: True Operative Rules: {true_operative_rules_p1}")
                logger.info(
                    f"P1: Operative Rules to evaluate: {len(true_operative_rules_p1)}"
                )
            case "classify_P2_Definitional_facts|true_table":
                true_facts_p2 = manager_true_elements.retrieve_document(
                    "classify_P2_Definitional_facts", "true_table"
                ).content
                logger.debug(f"P2: True Facts: {true_facts_p2}")
                logger.info(f"P2: Facts to evaluate: {len(true_facts_p2)}")
            case "classify_P2_Definitional_names|true_table":
                true_names_p2 = manager_true_elements.retrieve_document(
                    "classify_P2_Definitional_names", "true_table"
                ).content
                logger.debug(f"P2: True Names: {true_names_p2}")
                logger.info(f"P2: Names to evaluate: {len(true_names_p2)}")
            case "classify_P2_Definitional_terms|true_table":
                true_terms_p2 = manager_true_elements.retrieve_document(
                    "classify_P2_Definitional_terms", "true_table"
                ).content
                logger.debug(f"P2: True Terms: {true_terms_p2}")
                logger.info(f"P2: Terms to evaluate: {len(true_terms_p2)}")
            case "classify_P2_Operative_rules|true_table":
                true_operative_rules_p2 = manager_true_elements.retrieve_document(
                    "classify_P2_Operative_rules", "true_table"
                ).content
                logger.debug(f"P2: True Operative Rules: {true_operative_rules_p2}")
                logger.info(
                    f"P2: Operative Rules to evaluate: {len(true_operative_rules_p2)}"
                )

    return (
        true_operative_rules_p1,
        true_facts_p2,
        true_names_p2,
        true_terms_p2,
        true_operative_rules_p2,
    )


class DocumentProcessor:
    """
    DocumentProcessor is responsible for processing documents and categorizing elements such as terms, names, facts, and rules.

    Attributes:
        manager: Object used to manage document retrieval.
        elements_terms_set (set): Set of unique terms found in the documents.
        elements_names_set (set): Set of unique names found in the documents.
        elements_terms (list): List of detailed information about terms.
        elements_names (list): List of detailed information about names.
        elements_facts (list): List of facts extracted from documents.
        elements_rules (list): List of rules extracted from documents.
        elements_terms_definition (dict): Dictionary to store terms definitions by document ID.
    """

    def __init__(self, manager: DocumentManager, merge: bool = False):
        self.manager = manager
        self.elements_terms_set = set()
        self.elements_names_set = set()
        self.elements_terms = []
        self.elements_names = []
        self.elements_facts = []
        self.elements_rules = []
        self.elements_terms_definition = {}
        self.operative_rules_classifications = (
            []
        )  # To store classifications with type and subtype for operative rules
        self.facts_classifications = (
            []
        )  # To store classifications with type and subtype for facts
        self.terms_classifications = (
            []
        )  # To store classifications with type, subtype, and confidence for definitional terms
        self.names_classifications = (
            []
        )  # To store classifications with type, subtype, and confidence for definitional names

        # Automatically process definitions, classifications, and elements when instantiated
        try:
            self.process_definitions()
        except Exception as e:
            logger.error(f"Error processing definitions: {e}")

        try:
            self.process_operative_rules_classifications()
        except Exception as e:
            logger.info(
                f"Document did not have operative rules classifications to process: {e}"
            )

        try:
            self.process_facts_classifications()
        except Exception as e:
            logger.info(f"Document did not have facts classifications to process: {e}")

        try:
            self.process_terms_classifications()
        except Exception as e:
            logger.info(f"Document did not have terms classifications to process: {e}")

        try:
            self.process_names_classifications()
        except Exception as e:
            logger.info(f"Document did not have names classifications to process: {e}")

        try:
            self.process_elements()
        except Exception as e:
            logger.error(f"Error processing elements: {e}")
            raise e

        # Conditionally merge elements if `merge` is True
        if merge:
            try:
                self.merge_terms()
                self.merge_names()
            except Exception as e:
                logger.error(f"Error merging elements: {e}")
                raise e
            
        try:
            self.process_transformed_elements()
        except Exception as e:
            logger.error(f"Error processing transformed elements: {e}")
            raise e

        try:
            self.process_validations()
        except Exception as e:
            logger.error(f"Error processing validations: {e}")
            raise e

    def process_validations(self):
        """
        Processes validation documents and updates elements with validation scores and findings.
        """
        validation_docs = {
            "validation_judge_Operative_Rules": self.elements_rules,
            "validation_judge_Fact_Types": self.elements_facts,
            "validation_judge_Terms": self.elements_terms,
            "validation_judge_Names": self.elements_names,
        }

        for doc_name, elements_list in validation_docs.items():
            try:
                # Retrieve the validation document
                validation_doc = self.manager.retrieve_document(doc_name, "llm_validation")
                if not validation_doc or not validation_doc.content:
                    logger.warning(f"Validation document '{doc_name}' not found or empty.")
                    continue

                # Iterate over items in the validation document
                for item in validation_doc.content:
                    doc_id = normalize_str(item.get("doc_id"))
                    statement_id = normalize_str(str(item.get("statement_id")))
                    sources = item.get("sources")

                    # Fields to add
                    semscore = item.get("semscore")
                    similarity_score = item.get("similarity_score")
                    similarity_score_confidence = item.get("similarity_score_confidence")
                    transformation_accuracy = item.get("transformation_accuracy")
                    grammar_syntax_accuracy = item.get("grammar_syntax_accuracy")
                    findings = item.get("findings")

                    # Find matching element in elements_list
                    for element in elements_list:
                        if (
                            normalize_str(element.get("doc_id")) == doc_id
                            and normalize_str(str(element.get("statement_id"))) == statement_id
                            and set(element["sources"]) == set(sources) 
                        ):
                            # Update element with new fields
                            element.update({
                                "semscore": semscore,
                                "similarity_score": similarity_score,
                                "similarity_score_confidence": similarity_score_confidence,
                                "transformation_accuracy": transformation_accuracy,
                                "grammar_syntax_accuracy": grammar_syntax_accuracy,
                                "findings": findings,
                            })
                            break  # Exit the loop after finding the matching element
            except Exception as e:
                logger.error(f"Error processing validation document '{doc_name}': {e}")

    def _update_element(self, elements_list, doc_id, statement_id, sources, transformed):
        """
        Updates the transformed attribute for the matching element in the given list.
        """
        for element in elements_list:
            logger.debug(f'{doc_id}, {statement_id}, {sources} == {normalize_str(element["doc_id"])}, {str(element["statement_id"])}, {element["sources"]}')
            if (
                normalize_str(element["doc_id"]) == doc_id
                and normalize_str(str(element["statement_id"])) == statement_id
                and set(element["sources"]) == set(sources)  # Updated comparison for list
            ):  
                logger.debug('MATCH')
                element["transformed"] = transformed
                logger.debug(f"Updated element: {element}")
                break

    def process_transformed_elements(self):
        """
        Processes the transformed elements from the llm_response_transform document types and updates the transformed attribute in each relevant list.
        """
        transform_docs = {
            "transform_Fact_Types": self.elements_facts,
            "transform_Terms": self.elements_terms,
            "transform_Names": self.elements_names,
            "transform_Operative_Rules": self.elements_rules,
        }

        for transform_doc_id, elements_list in transform_docs.items():
            transform_doc = self.manager.retrieve_document(transform_doc_id, "llm_response_transform")

            if transform_doc is None or not transform_doc.content:
                logger.warning(f"Document '{transform_doc_id}' of type 'llm_response_transform' not found or empty.")
                continue  # Skip this document and move to the next iteration

            transform_doc_content = transform_doc.content

            for item in transform_doc_content:
                logger.debug(f"{item=}")
                doc_id = normalize_str(item.get("doc_id"))
                statement_id = normalize_str(str(item.get("statement_id")))
                sources = item.get("statement_sources")
                transformed = item.get("transformed")

                logger.debug(f"doc_id: {doc_id} - statement_id: {statement_id} - transformed: {transformed}")
                self._update_element(elements_list, doc_id, statement_id, sources, transformed)


    def process_names_classifications(self):
        """
        Processes classification information specifically for names from 'classify_P2_Definitional_names'
        document and stores the type, subtype, subtype confidence, and subtype explanation.
        The type is always set to 'Definitional'.
        """
        # Document identifier we are interested in
        doc_classification = "classify_P2_Definitional_names"

        # Retrieve document content
        doc = self.manager.retrieve_document(
            doc_classification, "llm_response_classification"
        )
        if not doc or not doc.content:
            logger.warning(
                f"Document '{doc_classification}' not found or has empty content."
            )
            return

        doc_content = doc.content

        # Iterate over each item in the document content
        for item in doc_content:
            doc_id = normalize_str(item.get("doc_id"))
            statement_id = normalize_str(str(item.get("statement_id")))
            classifications = item.get("classification", [])

            # Iterate over each classification to extract the highest confidence one for names
            for classification in classifications:
                confidence = classification.get("confidence", 0)

                # Check if this classification already exists in names_classifications
                existing_name = next(
                    (
                        name
                        for name in self.names_classifications
                        if name["doc_id"] == doc_id
                        and name["statement_id"] == statement_id
                    ),
                    None,
                )

                # Initialize if not found
                if existing_name is None:
                    existing_name = {
                        "doc_id": doc_id,
                        "statement_id": statement_id,
                        "type": "Definitional",  # Set type as "Definitional"
                        "subtype": None,
                        "confidence": -1,
                        "explanation": "",
                        "templates_ids": [],
                    }
                    self.names_classifications.append(existing_name)

                # Update subtype information if the confidence is higher than the existing one
                if confidence > existing_name["confidence"]:
                    existing_name["subtype"] = classification.get("subtype")
                    existing_name["confidence"] = confidence
                    existing_name["explanation"] = classification.get("explanation", "")
                    existing_name["templates_ids"] = classification.get(
                        "templates_ids", []
                    )

        # Log the final classification for debugging purposes
        logger.debug(f"{self.names_classifications=}")

    def process_facts_classifications(self):
        """
        Processes classification information specifically for facts from 'classify_P2_Definitional_facts'
        document and stores the type, subtype, subtype confidence, and subtype explanation.
        The type is always set to 'Definitional'.
        """
        # Document identifier we are interested in
        doc_classification = "classify_P2_Definitional_facts"

        # Retrieve document content
        doc_content = self.manager.retrieve_document(
            doc_classification, "llm_response_classification"
        ).content

        # Iterate over each item in the document content
        for item in doc_content:
            doc_id = normalize_str(item.get("doc_id"))
            statement_id = normalize_str(str(item.get("statement_id")))
            classifications = item.get("classification", [])

            # Iterate over each classification to extract the highest confidence one for facts
            for classification in classifications:
                confidence = classification.get("confidence", 0)

                # Check if this classification already exists in facts_classifications
                existing_fact = next(
                    (
                        fact
                        for fact in self.facts_classifications
                        if fact["doc_id"] == doc_id
                        and fact["statement_id"] == statement_id
                    ),
                    None,
                )

                # Initialize if not found
                if existing_fact is None:
                    existing_fact = {
                        "doc_id": doc_id,
                        "statement_id": statement_id,
                        "type": "Definitional",  # Set type as "Definitional"
                        "subtype": None,
                        "subtype_confidence": -1,
                        "subtype_explanation": "",
                        "templates_ids": [],
                    }
                    self.facts_classifications.append(existing_fact)

                # Update subtype information if the confidence is higher than the existing one
                if confidence > existing_fact["subtype_confidence"]:
                    existing_fact["subtype"] = classification.get("subtype")
                    existing_fact["subtype_confidence"] = confidence
                    existing_fact["subtype_explanation"] = classification.get(
                        "explanation", ""
                    )
                    existing_fact["templates_ids"] = classification.get(
                        "templates_ids", []
                    )

        # Log the final classification for debugging purposes
        logger.debug(f"{self.facts_classifications=}")

    def process_terms_classifications(self):
        """
        Processes classification information specifically for terms from 'classify_P2_Definitional_terms'
        document and stores the type, subtype, subtype confidence, and subtype explanation.
        The type is always set to 'Definitional'.
        """
        # Document identifier we are interested in
        doc_classification = "classify_P2_Definitional_terms"

        # Retrieve document content
        doc = self.manager.retrieve_document(
            doc_classification, "llm_response_classification"
        )
        if not doc or not doc.content:
            logger.warning(
                f"Document '{doc_classification}' not found or has empty content."
            )
            return

        doc_content = doc.content

        # Iterate over each item in the document content
        for item in doc_content:
            doc_id = normalize_str(item.get("doc_id"))
            statement_id = normalize_str(str(item.get("statement_id")))
            classifications = item.get("classification", [])

            # Iterate over each classification to extract the highest confidence one for terms
            for classification in classifications:
                confidence = classification.get("confidence", 0)

                # Check if this classification already exists in terms_classifications
                existing_term = next(
                    (
                        term
                        for term in self.terms_classifications
                        if term["doc_id"] == doc_id
                        and term["statement_id"] == statement_id
                    ),
                    None,
                )

                # Initialize if not found
                if existing_term is None:
                    existing_term = {
                        "doc_id": doc_id,
                        "statement_id": statement_id,
                        "type": "Definitional",  # Set type as "Definitional"
                        "subtype": None,
                        "confidence": -1,
                        "explanation": "",
                        "templates_ids": [],
                    }
                    self.terms_classifications.append(existing_term)

                # Update subtype information if the confidence is higher than the existing one
                if confidence > existing_term["confidence"]:
                    existing_term["subtype"] = classification.get("subtype")
                    existing_term["confidence"] = confidence
                    existing_term["explanation"] = classification.get("explanation", "")
                    existing_term["templates_ids"] = classification.get(
                        "templates_ids", []
                    )

        # Log the final classification for debugging purposes
        logger.debug(f"{self.terms_classifications=}")

    def add_definition(self, doc_id, term, definition, isLocalScope):
        """
        Adds a term definition and isLocalScope to the elements_terms_definition dictionary.

        Args:
            doc_id (str): Identifier of the document.
            term (str): The term to be defined.
            definition (str): The definition of the term.
            isLocalScope (bool): The isLocalScope value.
        """
        self.elements_terms_definition.setdefault(doc_id, {})[term] = {
            'definition': definition,
            'isLocalScope': isLocalScope
        }

    def process_definitions(self):
        """
        Processes document terms definitions and stores them in elements_terms_definition.
        """
        docs_p2 = [
            s
            for s in self.manager.list_document_ids(doc_type="llm_response")
            if s.endswith("_P2")
        ]

        for doc in docs_p2:
            doc_id = doc.replace("_P2", "")
            doc_content = self.manager.retrieve_document(
                doc, doc_type="llm_response"
            ).content
            doc_terms = doc_content.get("terms", [])
            for term in doc_terms:
                self.add_definition(
                    doc_id,
                    term.get("term"),
                    term.get("definition"),
                    term.get("isLocalScope")
                )

    def process_elements(self):
        """
        Processes elements from documents and categorizes them into terms, names, facts, and rules.
        """
        # Get the list of documents that end with '_P1'
        docs_p1 = [
            s
            for s in self.manager.list_document_ids(doc_type="llm_response")
            if s.endswith("_P1")
        ]

        for doc in docs_p1:
            doc_content = self.manager.retrieve_document(
                doc, doc_type="llm_response"
            ).content
            doc_id = doc_content.get("section")
            doc_elements = doc_content.get("elements", [])
            for element in doc_elements:
                element_classification = element.get("classification")
                element_id = element.get("id")
                verb_symbols = element.get("verb_symbols") or element.get("verb_symbol")
                if isinstance(verb_symbols, str):
                    verb_symbols = [verb_symbols]
                elif verb_symbols is None:
                    verb_symbols = []
                element_dict = {
                    "doc_id": doc_id,
                    "statement_id": element_id,
                    "statement_title": element.get("title"),
                    "statement": element.get("statement"),
                    "sources": element.get("sources"),
                    "terms": element.get("terms", []),
                    "verb_symbols": verb_symbols,
                    "element_name": element_classification,
                }
                logger.debug(f"{element_dict=}")

                match element_classification:
                    case "Fact" | "Fact Type":
                        self.elements_facts.append(element_dict)
                    case "Operative Rule":
                        self.elements_rules.append(element_dict)

                element_terms = element.get("terms", [])
                if element_terms:
                    for term in element_terms:
                        signifier = term.get("term")
                        # Retrieve definition and isLocalScope
                        term_info = self.elements_terms_definition.get(doc_id, {}).get(signifier, {})
                        definition = term_info.get('definition')
                        isLocalScope = term_info.get('isLocalScope')

                        term_dict = {
                            "doc_id": doc_id,
                            "statement_id": signifier,  # Using signifier as statement_id
                            "definition": definition,
                            "isLocalScope": isLocalScope,
                            "sources": element.get("sources"),
                            "element_name": "Term" if term.get("classification") == "Common Noun" else "Name",
                        }

                        # Append new term
                        if term_dict["element_name"] == "Term":
                            self.elements_terms.append(term_dict)
                            self.elements_terms_set.add(signifier)
                        else:
                            self.elements_names.append(term_dict)
                            self.elements_names_set.add(signifier)

    def process_operative_rules_classifications(self):
        """
        Processes classification information specifically for operative rules from
        'classify_P1' and 'classify_P2_Operative_rules' documents, and stores the type, subtype,
        confidence, and explanation.
        """
        # Get only the specific documents we are interested in
        docs_classification = [
            s
            for s in self.manager.list_document_ids(
                doc_type="llm_response_classification"
            )
            if s in ["classify_P1", "classify_P2_Operative_rules"]
        ]

        # A temporary dictionary to group classifications by (doc_id, statement_id)
        classification_dict = {}

        # Iterate over each document
        for doc in docs_classification:
            # Retrieve document content for each classification document
            doc_content = self.manager.retrieve_document(
                doc, "llm_response_classification"
            ).content

            # Iterate over each item in the document content
            for item in doc_content:
                doc_id = normalize_str(item.get("doc_id"))
                statement_id = normalize_str(str(item.get("statement_id")))
                classifications = item.get("classification", [])

                # Create a key for grouping classifications
                key = (doc_id, statement_id)

                # Initialize the classification entry if it doesn't exist
                if key not in classification_dict:
                    classification_dict[key] = {
                        "doc_id": doc_id,
                        "statement_id": statement_id,
                        "type": None,
                        "type_confidence": -1,  # Initialize with a negative value to ensure first confidence is updated
                        "type_explanation": "",
                        "subtype": None,
                        "subtype_confidence": -1,  # Initialize with a negative value to ensure first confidence is updated
                        "subtype_explanation": "",
                        "templates_ids": [],
                    }

                # Iterate over each classification to extract the highest confidence one
                for classification in classifications:
                    # Extract confidence and other classification details
                    confidence = classification.get("confidence", 0)
                    current_classification = classification_dict[key]

                    # Update based on document type and ensure we retain both type and subtype
                    if doc == "classify_P1":
                        # Update type if this document is from classify_P1 and has higher confidence
                        if confidence > current_classification["type_confidence"]:
                            current_classification["type"] = classification.get("type")
                            current_classification["type_confidence"] = confidence
                            current_classification["type_explanation"] = (
                                classification.get("explanation", "")
                            )

                    elif doc == "classify_P2_Operative_rules":
                        # Update subtype if this document is from classify_P2_Operative_rules and has higher confidence
                        if confidence > current_classification["subtype_confidence"]:
                            current_classification["subtype"] = classification.get(
                                "subtype"
                            )
                            current_classification["subtype_confidence"] = confidence
                            current_classification["subtype_explanation"] = (
                                classification.get("explanation", "")
                            )
                            current_classification["templates_ids"] = (
                                classification.get("templates_ids", [])
                            )

        # Convert the classification_dict to a list and assign to operative_rules_classifications
        self.operative_rules_classifications = list(classification_dict.values())
        logger.debug(f"{self.operative_rules_classifications=}")

    def merge_terms(self):
        """
        Merges term elements with the same doc_id and statement_id by combining sources lists and 
        retaining fields from the element with the highest semscore and similarity_score.
        """
        # Build a dictionary to group terms by (doc_id, statement_id)
        grouped_terms = {}
        for term in self.elements_terms:
            key = (term['doc_id'], term['statement_id'])
            grouped_terms.setdefault(key, []).append(term)
        
        # Now process each group to merge elements
        merged_terms = []
        for key, terms in grouped_terms.items():
            # Collect all 'sources' lists into a combined list and remove duplicates
            sources = []
            for term in terms:
                sources.extend(term.get('sources', []))
            sources = list(set(sources))  # Remove duplicates
            
            # Find the term with the highest 'semscore' and 'similarity_score'
            best_term = max(terms, key=lambda t: (t.get('similarity_score', 0), t.get('semscore', 0)))
            
            # Create a new term with merged data
            merged_term = best_term.copy()
            merged_term['sources'] = sources
            
            # Add the merged term to the list
            merged_terms.append(merged_term)
        
        # Replace self.elements_terms with the merged_terms
        self.elements_terms = merged_terms

    def merge_names(self):
        """
        Merges name elements with the same doc_id and statement_id by combining sources lists and 
        retaining fields from the element with the highest semscore and similarity_score.
        """
        # Build a dictionary to group names by (doc_id, statement_id)
        grouped_names = {}
        for name in self.elements_names:
            key = (name['doc_id'], name['statement_id'])
            grouped_names.setdefault(key, []).append(name)
        
        # Now process each group to merge elements
        merged_names = []
        for key, names in grouped_names.items():
            # Collect all 'sources' lists into a combined list and remove duplicates
            sources = []
            for name in names:
                sources.extend(name.get('sources', []))
            sources = list(set(sources))  # Remove duplicates
            
            # Find the name with the highest 'semscore' and 'similarity_score'
            best_name = max(names, key=lambda n: (n.get('similarity_score', 0), n.get('semscore', 0)))
            
            # Create a new name with merged data
            merged_name = best_name.copy()
            merged_name['sources'] = sources
            
            # Add the merged name to the list
            merged_names.append(merged_name)
        
        # Replace self.elements_names with the merged_names
        self.elements_names = merged_names


    def get_unique_terms(self, doc_id=None):
        """
        Returns the set of unique terms found in the documents. If doc_id is provided,
        returns only the unique terms for that specific document.

        Args:
            doc_id (str, optional): Identifier of the document. Defaults to None.

        Returns:
            set: Set of unique terms.
        """
        if doc_id:
            return {
                term["signifier"]
                for term in self.elements_terms
                if term["doc_id"] == doc_id
            }
        return self.elements_terms_set

    def get_unique_names(self, doc_id=None):
        """
        Returns the set of unique names found in the documents. If doc_id is provided,
        returns only the unique names for that specific document.

        Args:
            doc_id (str, optional): Identifier of the document. Defaults to None.

        Returns:
            set: Set of unique names.
        """
        if doc_id:
            return {
                name["signifier"]
                for name in self.elements_names
                if name["doc_id"] == doc_id
            }
        return self.elements_names_set

    def get_terms(self, doc_id=None, term_id=None, definition_filter="all"):
        """
        Returns the list of terms extracted from documents, enriched with type, subtype, confidence, and explanation.
        If doc_id and term_id are provided, returns a specific term.
        If only doc_id is provided, returns all terms for that specific document.

        Args:
            doc_id (str, optional): Document identifier to filter a specific term or all terms in the document.
            term_id (str, optional): Term identifier to filter a specific term.
            definition_filter (str): Filter for terms based on definition presence.
                                    "non_null" returns terms with definitions,
                                    "null" returns terms without definitions,
                                    "all" returns all terms regardless of definition.

        Returns:
            list or dict: List of enriched terms, or a dictionary with a specific term if both doc_id and term_id are provided.
        """
        # Create a lookup dictionary for terms classifications for efficient access
        term_classification_lookup = {
            (
                normalize_str(classification["doc_id"]),
                normalize_str(str(classification["statement_id"])),
            ): classification
            for classification in self.terms_classifications
        }

        enriched_terms = []

        # Enrich terms with classification information
        for term in self.elements_terms:
            term_key = (
                normalize_str(term["doc_id"]),
                normalize_str(term.get("statement_id", "")),
            )
            classification = term_classification_lookup.get(term_key)

            if classification:
                term.update(
                    {
                        "type": classification.get("type"),
                        "subtype": classification.get("subtype"),
                        "confidence": classification.get("confidence"),
                        "explanation": classification.get("explanation"),
                        "templates_ids": classification.get("templates_ids", []),
                    }
                )

            enriched_terms.append(term)

        # Apply filtering based on definition presence
        if definition_filter == "non_null":
            enriched_terms = [term for term in enriched_terms if term.get("definition")]
        elif definition_filter == "null":
            enriched_terms = [
                term for term in enriched_terms if not term.get("definition")
            ]
        else:
            logger.info("Not filtering terms based on definition presence.")

        # If both doc_id and term_id are provided, return the specific term
        if doc_id and term_id:
            for term in enriched_terms:
                if normalize_str(term["doc_id"]) == normalize_str(
                    doc_id
                ) and normalize_str(term.get("statement_id", "")) == normalize_str(
                    term_id
                ):
                    return term

            # Return None if no matching term is found
            logger.debug(f"No term found for doc_id='{doc_id}', term_id='{term_id}'")
            return None

        # If only doc_id is provided, return all terms that match the given doc_id
        if doc_id:
            filtered_terms = [
                term
                for term in enriched_terms
                if normalize_str(term["doc_id"]) == normalize_str(doc_id)
            ]

            if not filtered_terms:
                logger.debug(f"No terms found for doc_id='{doc_id}'")
                return []

            return filtered_terms

        # If neither doc_id nor term_id is provided, return all enriched terms
        return enriched_terms

    def get_names(self, doc_id=None, name_id=None, definition_filter="all"):
        """
        Returns the list of names extracted from documents, enriched with type, subtype, confidence, and explanation.
        If doc_id and name_id are provided, returns a specific name.
        If only doc_id is provided, returns all names for that specific document.

        Args:
            doc_id (str, optional): Document identifier to filter a specific name or all names in the document.
            name_id (str, optional): Name identifier to filter a specific name.
            definition_filter (str, optional): Filter for names based on definition presence.
                                            "non_null" returns names with definitions,
                                            "null" returns names without definitions,
                                            "all" returns all names regardless of definition.

        Returns:
            list or dict: List of enriched names, or a dictionary with a specific name if both doc_id and name_id are provided.
        """
        # Create a lookup dictionary for names classifications for efficient access
        name_classification_lookup = {
            (
                normalize_str(classification["doc_id"]),
                normalize_str(str(classification["statement_id"])),
            ): classification
            for classification in self.names_classifications
        }

        enriched_names = []

        # Enrich names with classification information
        for name in self.elements_names:
            name_key = (
                normalize_str(name["doc_id"]),
                normalize_str(name.get("statement_id", "")),
            )
            classification = name_classification_lookup.get(name_key)

            if classification:
                name.update(
                    {
                        "type": classification.get("type"),
                        "subtype": classification.get("subtype"),
                        "confidence": classification.get("confidence"),
                        "explanation": classification.get("explanation"),
                        "templates_ids": classification.get("templates_ids", []),
                    }
                )

            enriched_names.append(name)

        # Apply filtering based on definition presence
        if definition_filter == "non_null":
            enriched_names = [name for name in enriched_names if name.get("definition")]
        elif definition_filter == "null":
            enriched_names = [
                name for name in enriched_names if not name.get("definition")
            ]
        else:
            logger.info("Not filtering names based on definition presence.")

        # If both doc_id and name_id are provided, return the specific name
        if doc_id and name_id:
            for name in enriched_names:
                if normalize_str(name["doc_id"]) == normalize_str(
                    doc_id
                ) and normalize_str(name.get("statement_id", "")) == normalize_str(
                    name_id
                ):
                    return name

            # Return None if no matching name is found
            logger.debug(f"No name found for doc_id='{doc_id}', name_id='{name_id}'")
            return None

        # If only doc_id is provided, return all names that match the given doc_id
        if doc_id:
            filtered_names = [
                name
                for name in enriched_names
                if normalize_str(name["doc_id"]) == normalize_str(doc_id)
            ]

            if not filtered_names:
                logger.debug(f"No names found for doc_id='{doc_id}'")
                return []

            return filtered_names

        # If neither doc_id nor name_id is provided, return all enriched names
        return enriched_names

    def get_facts(self, doc_id=None, statement_id=None):
        """
        Returns the list of facts extracted from documents, enriched with type, subtype, confidence, and explanation.
        If doc_id and statement_id are provided, returns a specific fact.
        If only doc_id is provided, returns all facts for that specific document.

        Args:
            doc_id (str, optional): Document identifier to filter a specific fact or all facts in the document.
            statement_id (str, optional): Statement identifier to filter a specific fact.

        Returns:
            list or dict: List of enriched facts, or a dictionary with a specific fact if both doc_id and statement_id are provided.
        """
        # Create a lookup dictionary for facts classifications for efficient access
        fact_classification_lookup = {
            (
                normalize_str(classification["doc_id"]),
                normalize_str(str(classification["statement_id"])),
            ): classification
            for classification in self.facts_classifications
        }

        enriched_facts = []

        # Enrich facts with classification information
        for fact in self.elements_facts:
            fact_key = (
                normalize_str(fact["doc_id"]),
                normalize_str(str(fact["statement_id"])),
            )
            classification = fact_classification_lookup.get(fact_key)

            if classification:
                fact.update(
                    {
                        "type": classification.get("type"),
                        "subtype": classification.get("subtype"),
                        "subtype_confidence": classification.get("subtype_confidence"),
                        "subtype_explanation": classification.get(
                            "subtype_explanation"
                        ),
                        "templates_ids": classification.get("templates_ids", []),
                    }
                )

            enriched_facts.append(fact)

        # If both doc_id and statement_id are provided, return the specific fact
        if doc_id and statement_id:
            for fact in enriched_facts:
                if normalize_str(fact["doc_id"]) == normalize_str(
                    doc_id
                ) and normalize_str(str(fact["statement_id"])) == normalize_str(
                    str(statement_id)
                ):
                    return fact

            # Return None if no matching fact is found
            logger.debug(
                f"No fact found for doc_id='{doc_id}', statement_id='{statement_id}'"
            )
            return None

        # If only doc_id is provided, return all facts that match the given doc_id
        if doc_id:
            filtered_facts = [
                fact
                for fact in enriched_facts
                if normalize_str(fact["doc_id"]) == normalize_str(doc_id)
            ]

            if not filtered_facts:
                logger.debug(f"No facts found for doc_id='{doc_id}'")
                return []

            return filtered_facts

        # If neither doc_id nor statement_id is provided, return all enriched facts
        return enriched_facts

    def get_rules(self, doc_id=None, statement_id=None):
        """
        Returns the list of rules extracted from documents, enriched with type, subtype, confidence, and explanation.
        If doc_id and statement_id are provided, returns a specific rule.
        If only doc_id is provided, returns all rules for that specific document.

        Args:
            doc_id (str, optional): Document identifier to filter a specific rule or all rules in the document.
            statement_id (str, optional): Statement identifier to filter a specific rule.

        Returns:
            list or dict: List of enriched rules, or a dictionary with a specific rule if both doc_id and statement_id are provided.
        """
        # Create a lookup dictionary for rules classifications for efficient access
        rule_classification_lookup = {
            (
                normalize_str(classification["doc_id"]),
                normalize_str(str(classification["statement_id"])),
            ): classification
            for classification in self.operative_rules_classifications
        }

        enriched_rules = []

        # Enrich rules with classification information
        for rule in self.elements_rules:
            rule_key = (
                normalize_str(rule["doc_id"]),
                normalize_str(str(rule["statement_id"])),
            )
            classification = rule_classification_lookup.get(rule_key)

            if classification:
                rule.update(
                    {
                        "type": classification.get("type"),
                        "type_confidence": classification.get("type_confidence"),
                        "type_explanation": classification.get("type_explanation"),
                        "subtype": classification.get("subtype"),
                        "subtype_confidence": classification.get("subtype_confidence"),
                        "subtype_explanation": classification.get(
                            "subtype_explanation"
                        ),
                        "templates_ids": classification.get("templates_ids"),
                    }
                )

            enriched_rules.append(rule)

        # If both doc_id and statement_id are provided, return the specific rule
        if doc_id and statement_id:
            for rule in enriched_rules:
                if normalize_str(rule["doc_id"]) == normalize_str(
                    doc_id
                ) and normalize_str(str(rule["statement_id"])) == normalize_str(
                    str(statement_id)
                ):
                    return rule

            # Return None if no matching rule is found
            logger.debug(
                f"No rule found for doc_id='{doc_id}', statement_id='{statement_id}'"
            )
            return None

        # If only doc_id is provided, return all rules that match the given doc_id
        if doc_id:
            filtered_rules = [
                rule
                for rule in enriched_rules
                if normalize_str(rule["doc_id"]) == normalize_str(doc_id)
            ]

            if not filtered_rules:
                logger.debug(f"No rules found for doc_id='{doc_id}'")
                return []

            return filtered_rules

        # If neither doc_id nor statement_id is provided, return all enriched rules
        return enriched_rules
