# CFR2SBVR

The primary objective of this project is to create a Knowledge Graph in RDF format, consisting of three named graphs:

- **FIBO**: This graph contains the [Financial Industry Business Ontology (FIBO)](https://github.com/edmcouncil/fibo). Specifically, it includes the [FIBO Production Quickstart](https://spec.edmcouncil.org/fibo/ontology/master/2024Q2/prod.fibo-quickstart.ttl) from the [EDM Council](https://edmcouncil.org/). Additional resources are accessible via the [Product Downloads](https://edmconnect.edmcouncil.org/okgspecialinterestgroup/resources-sig-link/resources-sig-link-fibo-products-download) page.

- **CFR**: This graph includes the Code of Federal Regulations (CFR) obtained from [eCFR](https://www.ecfr.gov/), specifically [Chapter 17, Part 275](https://www.ecfr.gov/current/title-17/chapter-II/part-275). It is encoded in RDF/OWL format using the [Financial Regulation Ontology (FRO)](https://finregont.com/), provided by [Jayzed Data Models Inc.](https://jayzed.com/). The corresponding file, [FRO_CFR_Title_17_Part_275.ttl](https://finregont.com/fro/cfr/FRO_CFR_Title_17_Part_275.ttl), is distributed under the [FIB-DM Open-Source, Core License](https://jayzed.com/terms-of-use/), which follows the GNU General Public License (GPL-3.0).

- **SBVR**: This graph integrates definitional rules from FIBO and CFR, along with behavioral rules extracted from CFR. These rules are extracted, transformed, and linked to their original sources using the cfr2sbvr tool. Additional details about this tool will be shared in an upcoming publication.

## Project Structure

- **src**: Contains the Python source code. Refer to `src/README.md` for setup instructions and dependencies.
- **data**: Holds all data files.
- **README.md**: The main documentation file providing an overview of the project.

## Dependencies

- Python 3.11 or later
  - [agraph-python](https://github.com/edmcouncil/agraph-python)
  - [SPARQLWrapper](https://github.com/RDFLib/sparqlwrapper)
  - Check `src/requirements.txt`
- [AllegroGraph 8.2.1 or later](https://franz.com/agraph/support/documentation/8.2.1/agraph-quick-start.html)
- Linux/MacOS or Windows WSL2 running Ubuntu 20.04 LTS or higher
- Miniconda 24.4.0 or later

## Python Environment Setup

To ensure a clean environment, it is recommended to create a virtual environment using [conda](https://docs.conda.io/en/latest/). 

After installing Conda, use the following commands to set up and activate the environment:

```bash
conda create -n ipt-cfr2sbvr python=3.11
conda activate ipt-cfr2sbvr
pip install -r requirements.txt
# Optional alternative:
# conda install -c conda-forge -c franzinc -n cfr2sbvr --file requirements.txt
```

## Running the Project

Before executing the scripts, set up the required environment variables (e, g., OPENAI_API_KEY) in the `.env` file and export them to the shell:

```bash
set -a
source .env
set +a
```

> Notebooks can be executed using Google Colab or JupyterLab.

### Notebook Execution Sequence

1. **Create the Python Module**: Execute `src/chap_6_cfr2sbvr_modules.ipynb` to generate the required Python module.
2. **Semantic Annotation Elements Extraction**: Run `src/chap_6_semantic_annotation_elements_extraction.ipynb` to extract semantic elements.
3. **Rules Classification**: Execute `src/chap_6_semantic_annotation_rules_classification.ipynb` to classify the extracted rules.
4. **NLP to SBVR Transformation**: Run `src/chap_6_nlp2sbvr_transform.ipynb` to transform the rules into SBVR.
5. **Prepare the Knowledge Graph (KG)**: Use `src/chap_6_create_kg.ipynb` to set up the KG structure.
6. **Populate the Knowledge Graph**: Execute `src/chap_6_nlp2sbvr_elements_association_creation.ipynb` to populate the CFR-SBVR graph.
7. **Validation - Extraction**: Run `src/chap_7_validation_elements_extraction.ipynb` to validate the extracted elements.
8. **Validation - Classification**: Execute `src/chap_7_validation_rules_classification.ipynb` to validate the rule classifications.
9. **Validation - Transformation**: Run `src/chap_7_validation_rules_transformation.ipynb` to validate the transformation process.

## Contributing

For guidelines on contributing to the project, please consult [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). 
