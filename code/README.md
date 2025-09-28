# CFR2SBVR

## TL;DR

The project creates a Knowledge Graph with three components: FIBO (Financial Industry Business Ontology), CFR (Code of Federal Regulations), and SBVR (Semantic Business Vocabulary and Rules) extracted from the first two.

## The project

The primary objective of this project is to create a Knowledge Graph in RDF format, consisting of three named graphs:

- **FIBO**: This graph contains the [Financial Industry Business Ontology (FIBO)](https://github.com/edmcouncil/fibo). Specifically, it includes the [FIBO Production Quickstart](https://spec.edmcouncil.org/fibo/ontology/master/2024Q2/prod.fibo-quickstart.ttl) from the [EDM Council](https://edmcouncil.org/). Additional resources are accessible via the [Product Downloads](https://edmconnect.edmcouncil.org/okgspecialinterestgroup/resources-sig-link/resources-sig-link-fibo-products-download) page.

- **CFR**: This graph includes the Code of Federal Regulations (CFR) obtained from [eCFR](https://www.ecfr.gov/), specifically [Chapter 17, Part 275](https://www.ecfr.gov/current/title-17/chapter-II/part-275). It is encoded in RDF/OWL format using the [Financial Regulation Ontology (FRO)](https://finregont.com/), provided by [Jayzed Data Models Inc.](https://jayzed.com/). The corresponding file, [FRO_CFR_Title_17_Part_275.ttl](https://finregont.com/fro/cfr/FRO_CFR_Title_17_Part_275.ttl), is distributed under the [FIB-DM Open-Source, Core License](https://jayzed.com/terms-of-use/), which follows the GNU General Public License (GPL-3.0).

- **SBVR**: This graph integrates definitional rules from FIBO and CFR, along with behavioral rules extracted from CFR. These rules are extracted, transformed, and linked to their original sources using the cfr2sbvr tool. Additional details about this tool will be shared in an upcoming publication.

## The project main structure

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

## Project Folder Structure

Root Level (/master-degree-santos-anderson/)

- code/ - Main project code (core implementation)
- docs/ - Documentation files
- grc-suppliers/ - GRC suppliers related files
- ontologies/ - Ontology files (FIBO, CFR, etc.)
- rsl/ - RSL (Research Support Library) files

Code Directory (/code/)

- src/ - Python source code (Jupyter notebooks & modules)
- data/ - Data files (input/output datasets)
- labs/ - Laboratory/experimental code
- outputs/ - Generated outputs from processing
- scripts/ - Utility scripts
- media/ - Media files (images, diagrams)
- cfr2sbvr_inspect/ - Code inspection tools
- Configuration files:
  - config.yaml - Local configuration
  - config.colab.yaml - Google Colab configuration
  - requirements.txt - Python dependencies

Source Directory (/code/src/)

Core Implementation:
- Jupyter Notebooks (execution sequence):
  a. chap_6_cfr2sbvr_modules.ipynb - Python module creation
  b. chap_6_semantic_annotation_elements_extraction.ipynb - Element extraction
  c. chap_6_semantic_annotation_rules_classification.ipynb - Rules classification
  d. chap_6_nlp2sbvr_transform.ipynb - NLP to SBVR transformation
  e. chap_6_create_kg.ipynb - Knowledge Graph setup
  f. chap_6_nlp2sbvr_elements_association_creation.ipynb - KG population
  g. chap_7_validation_*.ipynb - Validation notebooks

Support Modules:
- configuration/ - Configuration management
- checkpoint/ - Checkpoint/state management
- llm_query/ - LLM query handling
- logging_setup/ - Logging configuration
- rules_taxonomy_provider/ - Rules taxonomy management
- token_estimator/ - Token estimation utilities

## Notebook Execution Sequence

1. **Create the Python Module**: Execute `src/chap_6_cfr2sbvr_modules.ipynb` to generate the required Python module.
2. **Semantic Annotation Elements Extraction**: Run `src/chap_6_semantic_annotation_elements_extraction.ipynb` to extract semantic elements.
3. **Rules Classification**: Execute `src/chap_6_semantic_annotation_rules_classification.ipynb` to classify the extracted rules.
4. **NLP to SBVR Transformation**: Run `src/chap_6_nlp2sbvr_transform.ipynb` to transform the rules into SBVR.
5. **Prepare the Knowledge Graph (KG)**: Use `src/chap_6_create_kg.ipynb` to set up the KG structure.
6. **Populate the Knowledge Graph**: Execute `src/chap_6_nlp2sbvr_elements_association_creation.ipynb` to populate the CFR-SBVR graph.
7. **Validation - Extraction**: Run `src/chap_7_validation_elements_extraction.ipynb` to validate the extracted elements.
8. **Validation - Classification**: Execute `src/chap_7_validation_rules_classification.ipynb` to validate the rule classifications.
9. **Validation - Transformation**: Run `src/chap_7_validation_rules_transformation.ipynb` to validate the transformation process.

## Google Colab integration

Google Colab integration code is found across multiple Jupyter notebooks. Here's what handles Google Colab integration:

### Key Integration Pattern

The main Google Colab detection and setup code appears in multiple notebooks with this consistent pattern:

Files with Colab integration: /mnt/d/Projects/master-degree-santos-anderson/code/src/chap_6_*.ipynb

Core integration code (found in multiple notebooks):

```python
import sys
import os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
  from google.colab import drive
  drive.mount('/content/drive')
  !git clone https://github.com/asantos2000/master-degree-santos-anderson.git cfr2sbvr
  !cp -r cfr2sbvr/code/src/configuration .
  !cp -r cfr2sbvr/code/src/checkpoint .
  !cp -r cfr2sbvr/code/config.colab.yaml config.yaml
  DEFAULT_CONFIG_FILE="config.yaml"
else:
  # Local setup code
```

Colab-specific configuration: /mnt/d/Projects/master-degree-santos-anderson/code/config.colab.yaml:34-36
- Sets up Google Drive paths for data storage
- Configures output directories in /content/drive/MyDrive/cfr2sbvr/

Colab integration features:
1. Environment detection: Checks if running in Google Colab via 'google.colab' in sys.modules
2. Drive mounting: Mounts Google Drive for persistent storage
3. Repository cloning: Clones the GitHub repo into Colab environment
4. Configuration setup: Copies Colab-specific config files
5. Badge integration: Each notebook has a "Open in Colab" badge linking to the GitHub repo

## Contributing

For guidelines on contributing to the project, please consult [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). 
