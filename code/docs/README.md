# Documentação Técnica — CFR2SBVR

> Documentação de engenharia do código legado localizado em `code/`, produzida como base para o
> projeto de **rearquitetura e modernização**. O objetivo é registrar de forma minuciosa **o que o
> sistema faz**, **como está construído** e **onde estão os pontos frágeis**, para que decisões de
> modernização sejam tomadas com conhecimento completo do legado.

## O que é o CFR2SBVR

CFR2SBVR é a ferramenta de pesquisa desenvolvida na dissertação de mestrado de Anderson Santos. Ela
automatiza a transformação de **regulação financeira em linguagem natural** (Code of Federal
Regulations — CFR, Título 17, Parte 275, que regula *Investment Advisers*) em **regras de negócio
formais** no padrão **SBVR** (Semantic Business Vocabulary and Rules, da OMG), usando **LLMs**
(GPT-4o) orquestrados por um pipeline de quatro estágios, e materializa o resultado em um **Knowledge
Graph RDF** que integra três vocabulários: **FIBO**, **CFR (via FRO)** e o **SBVR** extraído.

Em uma frase: **texto regulatório → (LLM) → elementos semânticos → classificação → regras SBVR →
grafo de conhecimento**, com um aplicativo Streamlit para inspeção e validação humana dos resultados.

## Como ler esta documentação

Os documentos foram escritos para serem lidos em ordem, mas cada um é autocontido:

| # | Documento | Para quem / quando |
|---|-----------|--------------------|
| 1 | [01-arquitetura.md](01-arquitetura.md) | Visão de componentes, módulos, fluxos de controle e de dados. Comece aqui. |
| 2 | [02-infraestrutura.md](02-infraestrutura.md) | Dependências, ambientes de execução (local, Colab, cloud), AllegroGraph, DuckDB/MotherDuck, segredos. |
| 3 | [03-funcional.md](03-funcional.md) | O pipeline funcional em detalhe: extração, classificação, transformação, KG, validação. |
| 4 | [04-modelo-de-dados.md](04-modelo-de-dados.md) | Estruturas de dados: checkpoints JSON, `Document`/`DocumentManager`, views DuckDB, ontologia SBVR/RDF. |
| 5 | [05-glossario-e-dominio.md](05-glossario-e-dominio.md) | Glossário de domínio (SBVR, FIBO, CFR, taxonomia de Witt) e siglas. |
| 6 | [06-modernizacao.md](06-modernizacao.md) | Diagnóstico do legado (dívida técnica, riscos) e recomendações de rearquitetura. |

## Mapa rápido do repositório `code/`

```
code/
├── README.md                     # Visão do autor + sequência de execução dos notebooks
├── CONTRIBUTTING.md              # (sic) guia de contribuição
├── requirements.txt              # Dependências do pipeline (notebooks/módulos)
├── config.yaml                   # Configuração ativa (contém SEGREDOS — ver infra)
├── config.example.yaml           # Modelo de configuração
├── config.colab.yaml             # Configuração para Google Colab
│
├── src/                          # NÚCLEO do pipeline
│   ├── chap_6_*.ipynb            # Notebooks de execução (extração→classificação→transformação→KG)
│   ├── chap_7_*.ipynb            # Notebooks de validação/avaliação
│   ├── configuration/           # Módulo: carga de config + nomeação de arquivos versionados
│   ├── checkpoint/              # Módulo: modelo de dados Document + persistência + agregação
│   ├── llm_query/               # Módulo: chamada ao LLM (OpenAI via instructor)
│   ├── token_estimator/         # Módulo: contagem de tokens (tiktoken)
│   ├── logging_setup/           # Módulo: logging rotativo
│   ├── rules_taxonomy_provider/ # Módulo: taxonomia/templates de regras (Witt 2012)
│   ├── sbvr_xsd_to_rdf.py        # Utilitário CLI: XSD do SBVR → RDF
│   └── queries.sparqlbook        # Consultas SPARQL de apoio
│
├── cfr2sbvr_inspect/             # APLICATIVO Streamlit de inspeção/validação
│   ├── streamlit_app.py          # UI principal
│   ├── app_modules.py            # Lógica: highlight, DuckDB, geração de RDF, chatbot
│   ├── rules_taxonomy_provider/  # Cópia do provider de taxonomia
│   └── data/                     # Bancos DuckDB (v4/v5), views SQL, YAMLs de taxonomia
│
├── data/                         # Dados de entrada, ontologias (.ttl/.rdf) e checkpoints
│   ├── checkpoints*/             # Estados intermediários por estágio (JSON)
│   ├── run_v4/ run_v5/           # Execuções completas versionadas
│   ├── *.ttl / *.rdf             # FIBO, CFR/FRO, SBVR, tabela de subtipos
│   └── documents_true_table.json # "Golden dataset" (gabarito) de validação
│
├── scripts/                      # Shell + specs de indexação vetorial AllegroGraph (.def)
├── labs/                         # Protótipos/experimentos (não-produção)
├── outputs/                      # Planilhas/HTML de métricas geradas pela validação
└── media/                        # Imagens/diagramas
```

## Aviso importante para a modernização

Este repositório é **código de pesquisa acadêmica**, não um produto de engenharia. As implicações
estão detalhadas em [06-modernizacao.md](06-modernizacao.md), mas os pontos que o leitor deve ter em
mente desde já:

- **A lógica de negócio principal vive dentro de notebooks Jupyter** (`src/chap_6_*.ipynb`,
  `chap_7_*.ipynb`), não em módulos testáveis. Os módulos Python (`src/*/main.py`) são bibliotecas de
  apoio.
- **Há segredos reais versionados** em `config.yaml` (senha do AllegroGraph, credenciais de cloud).
  Devem ser rotacionados e removidos do histórico.
- **Não há testes automatizados** nem CI de qualidade de código (apenas correções de vulnerabilidade
  Snyk visíveis no histórico de commits).
- **O estado é passado entre estágios via arquivos JSON versionados por data** (checkpoints), e depois
  reprojetado em DuckDB por meio de views SQL — dois modelos de dados paralelos para os mesmos dados.
- **Duas "versões" de processo coexistem** (v4 e v5) com semânticas diferentes de encadeamento entre
  estágios (ver [03-funcional.md](03-funcional.md#versões-de-processo-v4-vs-v5)).

---
*Documentação gerada a partir da inspeção do código em `code/` na branch `main`.*
