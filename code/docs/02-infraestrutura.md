# 02 — Infraestrutura

Este documento cataloga tudo que é necessário para **executar** o CFR2SBVR: linguagem, dependências,
serviços externos, ambientes de execução, configuração e segredos. É a base para decidir a plataforma
alvo da modernização.

## 1. Plataforma e runtime

| Item | Valor observado no código |
|------|---------------------------|
| Linguagem | Python **3.11+** (`README.md`, devcontainer usa `python:1-3.11-bullseye`) |
| Gerenciador de ambiente recomendado | **conda/Miniconda** (env `ipt-cfr2sbvr`) — alternativa `pip` |
| SO alvo | Linux/macOS ou **Windows via WSL2 (Ubuntu 20.04+)** |
| Execução do pipeline | **Jupyter/JupyterLab** ou **Google Colab** |
| Execução do app | **Streamlit** (porta 8501) |
| Dev container | `.devcontainer/devcontainer.json` — imagem devcontainers Python 3.11, instala `requirements.txt` + `streamlit`, sobe o app no `postAttachCommand` |

> **Nota:** o repositório está clonado em Windows (`D:\Projects\...`), mas o código assume caminhos
> POSIX (`../data`, `/content/drive/...`, `/home/adsantos/agraph-8.3.1`). A execução real presume
> WSL2/Linux/Colab. Isso é relevante para a estratégia de containerização na modernização.

## 2. Dependências de software

### 2.1 Pipeline — `code/requirements.txt`

Sem versões fixadas (exceto pins de segurança do Snyk). Agrupadas por função:

- **Núcleo/dados:** `pandas`, `numpy`, `pydantic`, `openpyxl`, `statsmodels`
- **IA/LLM/NLP:** `openai`, `instructor`, `tiktoken`, `anthropic`, `google-generativeai`,
  `spacy`, `scikit-learn`, `sentencepiece`, `humanize`
- **Grafo/RDF/DB:** `agraph-python` (Franz AllegroGraph), `rdflib`, `defusedxml`, `duckdb`
- **Visualização/relatórios:** `matplotlib`, `plotly`, `seaborn`, `XlsxWriter`, `missingno`
- **Utilitários:** `inflect`, `pyyaml`, `jsonref`, `jellyfish`, `python-dotenv`, `natsort`
- **Ambiente:** `jupyterlab`, `streamlit`
- **Pins de segurança (Snyk):** `tornado>=6.5.7`, `pillow>=10.0.1`, `setuptools>=78.1.1`,
  `fonttools>=4.61.0`, `nbconvert>=7.17.0`, `protobuf>=6.33.5`, `urllib3>=2.5.0`, `zipp>=3.19.1`,
  `mistune>=3.2.1`

> Embora `anthropic` e `google-generativeai` estejam listados, o código de execução usa **apenas
> OpenAI** (`llm_query.query_instruct_llm` → `instructor.from_openai(OpenAI())`). Os outros parecem
> resquícios de experimentação (`labs/`).

### 2.2 App de inspeção — `code/cfr2sbvr_inspect/requirements.txt`

**Este sim tem versões fixadas** (é o artefato mais "produtizável" do repositório):

```
duckdb==1.3.2      deepdiff==8.0.1    numpy==1.26.4     pandas==2.2.2
rdflib==7.0.0      networkx==3.2.1    matplotlib==3.9.2 jellyfish==1.1.0
python-dotenv==1.0.1  PyYAML==6.0.2   natsort==8.4.0    openai==1.52.2
streamlit==1.41.1  requests==2.31.0
```

### 2.3 Dependências de sistema (fora do pip)

- **AllegroGraph 8.2.1+** (Franz) — triplestore RDF. Instalado localmente em
  `/home/adsantos/agraph-8.3.1` (config local) — usado via `bin/agtool` e `bin/agraph-control`.
- **stunnel** — túnel TLS para conectar à AllegroGraph Cloud (o cliente `agraph-python` fala com
  `127.0.0.1:8443` → `...allegrograph.cloud:443`).
- **graphviz/graphviz-dev** — apenas se `pygraphviz` for reativado (comentado no requirements).

## 3. Serviços externos (dependências de rede)

```mermaid
flowchart LR
    subgraph Local["Máquina / Colab"]
        NB["Notebooks + módulos"]
        APP["Streamlit app"]
    end
    OpenAI["OpenAI API<br/>gpt-4o + text-embedding-3-small"]
    AG_local["AllegroGraph local<br/>localhost:10035"]
    AG_cloud["AllegroGraph Cloud<br/>*.allegrograph.cloud:443 (via stunnel)"]
    MD["MotherDuck<br/>md:cfr2sbvr_db"]
    GH["GitHub<br/>clone em Colab"]
    Drive["Google Drive<br/>persistência em Colab"]

    NB -->|chat + embeddings| OpenAI
    NB -->|agraph-python / agtool| AG_local
    NB -->|stunnel| AG_cloud
    APP -->|duckdb read_only| MD
    APP -->|chatbot| OpenAI
    NB -.Colab.-> GH
    NB -.Colab.-> Drive
```

| Serviço | Uso | Autenticação | Onde configurado |
|---------|-----|--------------|------------------|
| **OpenAI** | LLM (`gpt-4o`) para extração/classificação/transformação/validação; embeddings (`text-embedding-3-small`) para indexação vetorial no AG; chatbot do app | `OPENAI_API_KEY` (env var / `.env` / secrets) | `config.yaml` LLM, `.env` |
| **AllegroGraph (local)** | Triplestore do Knowledge Graph; índice vetorial via `agtool llm index` | usuário/senha em `config.yaml` | `ALLEGROGRAPH_LOCAL` |
| **AllegroGraph Cloud** | Alternativa hospedada; acesso via **stunnel** na porta 8443 | usuário/senha em `config.yaml` | `ALLEGROGRAPH_CLOUD` + `agraph_stunnel.conf` |
| **MotherDuck** | DuckDB serverless na nuvem para o app | `MOTHER_DUCK_TOKEN` | `.env` / `st.secrets` |
| **GitHub** | Clone do repo dentro do Colab | pública | `config.colab.yaml` |
| **Google Drive** | Persistência de dados/checkpoints/outputs no Colab | OAuth interativo (`drive.mount`) | `config.colab.yaml` (paths `/content/drive/...`) |

## 4. Ambientes de execução

O código suporta **três topologias**, selecionadas por detecção/configuração:

### 4.1 Local (WSL2 / Linux / macOS)
- Config: `config.yaml` (`ALLEGROGRAPH_HOSTING: ALLEGROGRAPH_LOCAL`).
- Dados em `../data` relativo aos notebooks (`src/`).
- AllegroGraph rodando em `localhost:10035`.

### 4.2 Google Colab
- Detecção: `IN_COLAB = 'google.colab' in sys.modules`.
- Ações automáticas: `drive.mount('/content/drive')`, `git clone` do repo, cópia dos módulos
  `configuration`/`checkpoint` e de `config.colab.yaml` → `config.yaml`.
- Dados em `/content/drive/MyDrive/cfr2sbvr/...`.
- Cada notebook tem um badge *"Open in Colab"*.

### 4.3 AllegroGraph Cloud (via stunnel)
- Config: `ALLEGROGRAPH_HOSTING: ALLEGROGRAPH_CLOUD`.
- O notebook `chap_6_create_kg` escreve `agraph_stunnel.conf`, pede a senha sudo interativamente
  (`getpass`), sobe o `stunnel` e reaponta `HOST=localhost`, `PORT=8443`.

### 4.4 App Streamlit — local vs. cloud
- `DATABASE` define a fonte: `database_v5.db`/`database_v4.db` (arquivo local em `DEFAULT_DATA_DIR`) ou
  `md:cfr2sbvr_db` (MotherDuck). A conexão é sempre **`read_only=True`**.
- Deploy sugerido: Streamlit Community Cloud / devcontainer / Codespaces (o `devcontainer.json` já
  sobe o app com `--server.enableCORS false --server.enableXsrfProtection false`).

> ⚠️ Desabilitar CORS **e** XSRF protection é aceitável em preview de Codespaces, mas **não** deve ir
> para um deploy exposto. Anotado para a modernização.

## 5. Configuração

### 5.1 Arquivos de configuração

| Arquivo | Papel |
|---------|-------|
| `config.yaml` | **Config ativa** do pipeline (⚠️ contém segredos reais) |
| `config.example.yaml` | Modelo para novos ambientes |
| `config.colab.yaml` | Overrides para Colab (paths do Drive, `MAX_TOKENS: 4095`, `SIMILARITY_THRESHOLD: 0.99`) |
| `.env` / `.env.example` | `OPENAI_API_KEY`, `MOTHER_DUCK_TOKEN`, `DATABASE` (consumidos pelo app e pelos notebooks) |
| `.streamlit/config.toml` | Preferências do Streamlit (avisos) |

Carregamento (`configuration.load_config`):
1. Lê o YAML (`yaml.safe_load`).
2. Valida presença de chaves obrigatórias (`LLM`, `DEFAULT_CHECKPOINT_DIR`).
3. Sobrescreve `LLM.OPENAI_API_KEY` com a env var `OPENAI_API_KEY` (env tem prioridade).
4. Gera dinamicamente `DEFAULT_CHECKPOINT_FILE` e `DEFAULT_EXTRACTION_REPORT_FILE` com nomes
   versionados por data.

O app usa uma cascata própria: **`st.secrets` → variável de ambiente → default**
(`streamlit_app.get_config`, `app_modules.get_secret`).

### 5.2 Parâmetros relevantes (`config.yaml`)

| Chave | Valor | Significado |
|-------|-------|-------------|
| `LLM.MODEL` | `gpt-4o` | Modelo de chat |
| `LLM.EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings para busca vetorial no AG |
| `LLM.TEMPERATURE` | `0` | Determinismo |
| `LLM.MAX_TOKENS` | `8192` (Colab: `4095`) | Limite de saída |
| `SIMILARITY_THRESHOLD` | `0.85` (Colab: `0.99`) | Limiar de match exato na associação FIBO/CFR |
| `ALLEGROGRAPH_FORCE_RUN` | `True` | Executa mesmo com risco de perda de dados |
| `ALLEGROGRAPH_CLEAN_BEFORE_RUN` | `True` | Limpa o repositório antes de popular |
| `FIBO_GRAPH_VECTOR_STORE` | `fibo-glossary-3m-vec` | Nome do índice vetorial FIBO |
| `CFR_SBVR_GRAPH_VECTOR_STORE` | `cfr-sbvr-3m-vec` | Nome do índice vetorial CFR-SBVR |
| `QUALITY_THRESHOLD` (app) | `0.8` | Abaixo disso, escores aparecem em vermelho na UI |

## 6. 🔴 Segredos e segurança (ação necessária)

Levantamento factual do que está exposto no repositório:

| Local | Segredo exposto | Gravidade |
|-------|-----------------|-----------|
| `config.yaml` (`ALLEGROGRAPH_LOCAL.PASSWORD`) | `2002` | Média (local, mas versionada) |
| `config.yaml` (`ALLEGROGRAPH_CLOUD.PASSWORD`) | `GYL6N1KPB4R4xdWKIZqrYw` + host real | **Alta** — credencial de cloud em texto claro |
| `config.colab.yaml` (`PASSWORD`) | `2002` | Baixa (placeholder repetido) |
| `scripts/export-...sh`, `import-...sh` | `http://super:2002@localhost:10035` | Média (senha embutida na URL) |
| Notebook `create_kg` (cell 45) | monta `repo_spec` com `USER:PASSWORD@HOST` a partir do config | Média |
| `.env` presente no diretório de trabalho | pode conter chave OpenAI real | **Alta** se contiver chave válida |

**Recomendações imediatas (independentes da modernização):**
1. **Rotacionar** a senha da AllegroGraph Cloud e a chave OpenAI **agora** — devem ser consideradas
   comprometidas por estarem versionadas.
2. Substituir valores reais em `config.yaml` por `null`/placeholders e passar a lê-los **somente** de
   variáveis de ambiente / secret manager.
3. Purgar segredos do **histórico git** (BFG / `git filter-repo`), não apenas do HEAD.
4. Confirmar que `.env` está no `.gitignore` (o `.gitignore` existe; validar cobertura).
5. Nos `.def` de indexação, a chave é injetada em tempo de execução e removida depois
   (`update_llm_spec_file`), o que é um paliativo — preferir passagem por variável de ambiente.

## 7. Persistência e artefatos

| Artefato | Formato | Local | Papel |
|----------|---------|-------|-------|
| Checkpoints | JSON (`documents-YYYY-MM-DD-N.json`) | `data/checkpoints*/`, `data/run_v4/`, `data/run_v5/` | Estado intermediário entre estágios |
| Gabarito | JSON | `data/documents_true_table.json` | Golden dataset de validação |
| Bancos do app | DuckDB (`.db`) | `cfr2sbvr_inspect/data/` (`cfr2sbvr_v4.db`, `cfr2sbvr_v5.db`, e `database_v4/v5.db` em `data/`) | Consumo pelo Streamlit |
| Views | SQL | `cfr2sbvr_inspect/data/db_objects_v4|v5/` (numeradas 10..180) | Reprojeção relacional dos checkpoints |
| Ontologias | Turtle/RDF (`.ttl`, `.rdf`) | `data/` | FIBO, CFR/FRO, SBVR, subtipos |
| Métricas | XLSX/HTML | `outputs/` | Resultados da validação (cap. 7) |
| Logs | texto rotativo diário | `../logs/application.log` (config `DEFAULT_LOG_DIR`), `cfr2sbvr_inspect/streamlit_app.log` | Observabilidade |
| Índices vetoriais | interno AllegroGraph | repositório `cfr2sbvr` | Busca semântica FIBO/CFR |

## 8. Observabilidade

- **Logging** centralizado em `logging_setup.setting_logging` — `TimedRotatingFileHandler` (rotação à
  meia-noite, `backupCount=0` = mantém todos) + `StreamHandler` (console). Nível controlado por
  `LOG_LEVEL` no config.
- **Vários módulos chamam `logging.basicConfig` por conta própria** (`checkpoint`, `llm_query`,
  `app_modules`), o que pode conflitar com a configuração central — ponto de limpeza.
- **Custo/tempo do LLM:** `llm_query.measure_time` registra o tempo por chamada; `completion.usage`
  loga tokens. Não há métricas agregadas/telemetria formal.

Continua em [03-funcional.md](03-funcional.md).
