# CFR2SBVR Agent Instructions

## Environment Setup
```bash
conda activate ipt-cfr2sbvr  # Python 3.11+
pip install -r code/requirements.txt
source .env  # Load API keys (OPENAI_API_KEY, etc.)
```

## Key Commands
- **Run notebooks**: Use JupyterLab or execute cells sequentially in `code/src/chap_*.ipynb`
- **Lint**: No linter configured; follow existing code patterns
- **Tests**: No test suite; validate outputs in `code/outputs/`

## Code Style
- **Imports**: Group by standard lib, third-party (pandas, openai, rdflib), then local modules
- **Type hints**: Use typing (List, Dict, Optional, Any, Tuple) for function signatures
- **Functions**: Descriptive names with docstrings, helper functions prefixed with underscore
- **Error handling**: Try-except blocks for API calls and file I/O operations
- **File paths**: Use pathlib.Path for cross-platform compatibility
- **Config**: Load from yaml files (config.yaml, config.colab.yaml) via configuration module
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants