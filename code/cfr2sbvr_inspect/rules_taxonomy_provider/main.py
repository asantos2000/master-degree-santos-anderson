

import os
from pathlib import Path
import yaml

class RuleInformationProvider:
    """
    A class to provide information about rule classifications and templates based on YAML data.

    This class loads and processes rule classification data, template data, and example data from specified YAML files.
    It is used to generate markdown documentation for a given rule type, including details such as templates and examples.

    Attributes:
    -----------
    data_path : str
        Path to the directory containing the YAML files.
    template_dict : dict
        Dictionary containing template information loaded from the templates YAML file.
    examples_dict : dict
        Dictionary containing example information loaded from the examples YAML file.
    """
    
    def __init__(self, data_path):
        """
        Initializes the RuleInformationProvider with the specified data path.

        Parameters:
        -----------
        data_path : str
            Path to the directory containing the YAML files with rules, templates, and examples.
        """
        self.data_path = Path(data_path)
        self.template_dict = self._load_yaml(self.data_path / 'witt_templates.yaml', 'template_list')
        self.examples_dict = self._load_yaml(self.data_path / 'witt_examples.yaml', 'example_list')

    def _load_yaml(self, file_path, list_key=None):
        """
        Loads data from a YAML file.

        Parameters:
        -----------
        file_path : Path
            Path to the YAML file to be loaded.
        list_key : str, optional
            Key used to extract a specific list from the YAML data. If provided, returns a dictionary indexed by 'id'.

        Returns:
        --------
        dict
            If list_key is provided, returns a dictionary with items indexed by 'id'.
        Any type
            If list_key is not provided, returns the entire data structure from the YAML file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            if list_key and list_key in data:
                return {item['id']: item for item in data[list_key]}
            return data

    def get_classification_and_templates(self, section_title, return_forms="all"):
        """
        Retrieves classification information and templates for a specified rule section.

        Parameters:
        -----------
        section_title : str
            Title of the section for which to retrieve information.

        Returns:
        --------
        str
            A markdown formatted string containing the classification details, templates, and examples for the given section.
        """
        data = self._load_yaml(self.data_path / 'classify_subtypes.yaml')
        filtered_data = self._filter_sections_by_title(data, section_title)
        return self._convert_to_markdown(filtered_data, return_forms)

    def _filter_sections_by_title(self, data, title):
        """
        Filters sections based on the given title.

        Parameters:
        -----------
        data : list
            List of sections to filter from.
        title : str
            Title to filter sections by.

        Returns:
        --------
        list
            A list of sections that match the given title.
        """
        #return [section for section in data if section.get('section_title') == title]
        # Improve the filtering to include subsections
        results = []
        for section in data:
            if section.get('section_title') == title:
                results.append(section)
            if 'subsections' in section:
                results.extend(self._filter_sections_by_title(section['subsections'], title))
        return results

    def _convert_to_markdown(self, filtered_data, return_forms):
        """
        Converts filtered rule classification data to markdown format.

        Parameters:
        -----------
        filtered_data : list
            List of filtered sections to convert into markdown.

        Returns:
        --------
        str
            A markdown formatted string representing the filtered sections.
        """
        def process_section(section, level=2):
            """
            Processes a section recursively and converts it to markdown format.

            Parameters:
            -----------
            section : dict
                The section to process.
            level : int, optional
                The heading level for the section title in markdown (default is 2).

            Returns:
            --------
            str
                A markdown formatted string for the section and its subsections.
            """
            markdown = ""
            if 'class' in section:
                markdown += f"{'#' * level} Subtype: {section['section_title']}\n\n"
            else:
                markdown = f"{'#' * level} {section['section_title']}\n\n"
            
            markdown += f"**ID**: {section['section_id']}\n\n"
            markdown += f"**Definition**: {section['section_definition']}\n\n"

            if 'templates' in section:
                markdown += self._process_templates(section['templates'], return_forms)

            if 'examples' in section:
                #print(section, return_forms)
                #print(section['examples'])
                markdown += self._process_examples(section['examples'], return_forms)

            if 'subsections' in section:
                for subsection in section['subsections']:
                    markdown += process_section(subsection, level + 1)

            return markdown

        markdown = ""
        for section in filtered_data:
            markdown += process_section(section)
        return markdown

    def _process_templates(self, templates, return_forms):
        """
        Processes templates and formats them to markdown.
        """
        markdown = ""
        for template_id in templates:
            template = self.template_dict.get(template_id, None)
            if template:
                markdown += f"**Template ID**: {template_id}\n\n"
                if 'form' in template:
                    markdown += f"**Form**:\n```form\n{template['form']}\n```\n\n"
                markdown += f"**Template Explanation**: {template['explanation']}\n\n"
                if return_forms in ["rule", "all"] and 'rule_form' in template:
                    markdown += f"**Rule Form**:\n```rule_form\n{template['rule_form']}\n```\n\n"
                if return_forms in ["fact_type", "all"] and 'fact_type_form' in template:
                    markdown += f"**Fact Type Form**:\n```fact_type_form\n{template['fact_type_form']}\n```\n\n"
            else:
                markdown += f"**Template ID**: {template_id} - No details found.\n\n"
        return markdown

    def _process_examples(self, examples, return_forms="all"):
        """
        Processes examples and formats them to markdown.
        """
        markdown = ""
        for example_id in examples:
            example = self.examples_dict.get(example_id, None)
            if example:
                if return_forms == "rule" and not example_id.startswith("R"):
                    continue
                if return_forms == "fact_type" and not example_id.startswith("F"):
                    continue
                markdown += f"**Example ID**: {example_id}\n\n"
                markdown += f"**Example Text**:\n\n```example\n{example['text']}\n```\n\n"
            else:
                markdown += f"**Example ID**: {example_id} - No details found.\n\n"
        return markdown


class RulesTemplateProvider:
    """
    A class to provide information about rules templates and their relationships from YAML data.

    This class loads and processes template data, subtemplate data, and their relationships from specified YAML files.
    It is used to extract information about templates and format them into readable output.
    """
    
    def __init__(self, data_directory):
        """
        Initializes the RulesTemplateProvider with the specified data directory.

        Parameters:
        -----------
        data_directory : str or Path
            Path to the directory containing the YAML files with templates, subtemplates, and relationships.
        """
        self.data_directory = Path(data_directory)
        self.data_dicts = self._load_data()

    def _load_yaml(self, file_path):
        """
        Loads data from a YAML file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def _load_data(self):
        """
        Loads data from multiple YAML files required for template processing.
        """
        return {
            'witt_template_relationship_data': self._load_yaml(self.data_directory / 'witt_template_subtemplate_relationship.yaml').get('template_subtemplate_relationship', {}),
            'witt_templates_data': self._load_yaml(self.data_directory / 'witt_templates.yaml').get('template_list', []),
            'witt_subtemplates_data': self._load_yaml(self.data_directory / 'witt_subtemplates.yaml').get('subtemplate_list', [])
        }

    def get_rules_template(self, template_keys, return_forms="all"):
        """
        Retrieves the formatted rules templates for the specified list of template keys, avoiding duplicate subtemplates.

        Parameters:
        -----------
        template_keys : str or list of str
            The key(s) of the template(s) to be retrieved.
        return_forms : str
            Indicates which forms to return: 'rule_form', 'fact_type_form', or 'both'.
        
        Returns:
        --------
        str
            A formatted string representation of each template and its associated subtemplates, without duplicates.
        """
        if isinstance(template_keys, str):
            template_keys = [template_keys]

        output = ""
        processed_subtemplates = set()

        for template_key in template_keys:
            output += self._process_template(template_key, processed_keys=processed_subtemplates, return_forms=return_forms)
            output += "\n\n"

        return output

    def _process_template(self, template_key, processed_keys=None, return_forms="all"):
        if processed_keys is None:
            processed_keys = set()

        if template_key in processed_keys:
            return ''
        processed_keys.add(template_key)

        template_data = self._get_template_data(template_key)
        if not template_data:
            return f"# {template_key}\n\nTemplate data not found.\n\n"

        output = self._format_template_output(template_key, template_data, return_forms)

        if 'usesSubtemplate' in template_data:
            subtemplate_keys = template_data['usesSubtemplate']
            subtemplate_keys = [subtemplate_keys] if isinstance(subtemplate_keys, str) else subtemplate_keys
            for sub_key in subtemplate_keys:
                sub_key = sub_key.strip()
                output += self._process_template(sub_key, processed_keys, return_forms)

        return output

    def _get_template_data(self, template_key):
        if template_key.startswith('T'):
            template_data = self._find_data(template_key, self.data_dicts['witt_templates_data'])
            uses_subtemplate = self.data_dicts['witt_template_relationship_data'].get(template_key, [])
            if uses_subtemplate:
                template_data['usesSubtemplate'] = uses_subtemplate
        elif template_key.startswith('S'):
            template_data = self._find_data(template_key, self.data_dicts['witt_subtemplates_data'])
            uses_subtemplate = self.data_dicts['witt_template_relationship_data'].get(template_key, [])
            if uses_subtemplate:
                template_data['usesSubtemplate'] = uses_subtemplate
        else:
            template_data = None
        return template_data

    def _find_data(self, template_key, data_list):
        for item in data_list:
            if item.get('id', '') == template_key:
                return item
        return None

    def _format_template_output(self, template_key, template_data, return_forms):
        title = template_data.get('title', '')
        output = f"## {template_key}: {title}\n\n" if title else f"## {template_key}\n\n"

        if not template_data:
            output += "Template data not found.\n\n"
            return output

        if 'usesSubtemplate' in template_data:
            output += "### Subtemplate(s) in use\n"
            subtemplate_list = []
            for sub_key in template_data['usesSubtemplate']:
                sub_data = self._find_data(sub_key, self.data_dicts['witt_subtemplates_data'])
                sub_title = sub_data.get('title', '') if sub_data else "Unknown"
                subtemplate_list.append(f"- {sub_key}: {sub_title}")
            output += "\n".join(subtemplate_list) + "\n\n"
        
        if return_forms in ["rule", "all"] and 'rule_form' in template_data:
            output += f"### Rule Form\n\n{template_data['rule_form']}\n\n"
        if return_forms in ["fact_type", "all"] and 'fact_type_form' in template_data:
            output += f"### Fact Type Form\n\n{template_data['fact_type_form']}\n\n"
        if 'form' in template_data:
            output += f"### Form\n\n{template_data['form']}\n\n"
        if 'explanation' in template_data:
            output += f"### Explanation\n\n{template_data['explanation']}\n\n"
        return output
