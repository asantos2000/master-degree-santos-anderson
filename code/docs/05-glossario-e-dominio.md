# 05 — Glossário e Domínio

Referência de termos de domínio, padrões e siglas usados no código e nesta documentação. O sistema
cruza quatro domínios densos (regulação, ontologias financeiras, regras de negócio SBVR e LLMs); este
glossário existe para que a equipe de modernização não precise reconstruir esse contexto.

## 1. Domínios e padrões

### SBVR — Semantic Business Vocabulary and Rules
Padrão da **OMG** para especificar vocabulário de negócio e regras em forma controlada, próxima da
linguagem natural (CNL). É o **formato de saída** do projeto. Conceitos usados no código:

| Conceito SBVR | No código | Significado |
|---------------|-----------|-------------|
| **Term** | `elements_terms`, `sbvr:GeneralConcept` | Substantivo comum → conceito geral (ex.: "investment adviser") |
| **Name** | `elements_names`, `sbvr:IndividualNounConcept` | Substantivo próprio → conceito individual (ex.: "Commission") |
| **Fact Type** | `elements_facts` | Template abstrato de relação entre termos |
| **Fact** | `elements_facts` | Instância concreta de um fact type |
| **Operative Rule** | `elements_rules`, `sbvr:BehavioralBusinessRule` | Regra comportamental (obrigação/proibição/permissão) |
| **Definitional Rule** | `sbvr:DefinitionalRule` | Regra que define/constrange um construto |
| **Verb Symbol** | `verb_symbols`, `sbvr:VerbSymbol` | Símbolo verbal que liga termos num fact type |
| **Signifier** | `sbvr:signifier` | A designação (texto) de um conceito |
| **Designation is in namespace** | `sbvr:designationIsInNamespace` | Vocabulário ao qual a designação pertence |

### CFR — Code of Federal Regulations
Corpo da regulação federal dos EUA. O projeto foca **Título 17, Parte 275** (regras da SEC para
*Investment Advisers*, sob o Investment Advisers Act de 1940). Unidade de referência: **seção**
(`§ 275.0-2`) e **parágrafo** (`(a)`, `(b)(1)`, `(a)(1)`). Fonte pública: ecfr.gov.

### FRO — Financial Regulation Ontology
Ontologia (da Jayzed Data Models) que codifica o CFR em RDF/OWL. Arquivo:
`FRO_CFR_Title_17_Part_275.ttl`. Licença FIB-DM Open-Source Core (GPL-3.0). Fornece o vocabulário
`fro-cfr:` usado nas triplas.

### FIBO — Financial Industry Business Ontology
Ontologia de referência da indústria financeira (EDM Council). Usa-se o **Quickstart**
(`prod-fibo-quickstart-2024Q3.ttl`). Serve de alvo para **associação semântica**: termos SBVR
extraídos são ligados a conceitos FIBO via `skos:exactMatch`/`closeMatch` por busca vetorial.

### Taxonomia de Witt (2012)
De *Graham Witt, "Writing Effective Business Rules" (Elsevier, 2012)*. Classifica regras de negócio em
uma árvore (Definitional vs Operative rules, com subtipos como *Formal intensional definitions*,
*Categorization scheme enumerations*, etc.) e fornece **templates** de redação (CNL). É a base da
**classificação** (estágio 2) e da **transformação** (estágio 3). Materializada em
`classify_subtypes.yaml`, `witt_templates.yaml`, `witt_subtemplates.yaml`, `witt_examples.yaml`.

### CNL — Controlled Natural Language
Linguagem natural restrita a padrões gramaticais fixos (os templates de Witt). O produto da
transformação é CNL SBVR, não texto livre.

## 2. Padrões técnicos e de IA

| Termo | Significado no projeto |
|-------|------------------------|
| **LLM** | Large Language Model — `gpt-4o` da OpenAI, usado em todos os estágios cognitivos |
| **instructor** | Biblioteca que força a saída do LLM a um schema Pydantic (`response_model`) |
| **Structured output** | Resposta do LLM validada como objeto tipado (não texto solto) |
| **Embedding / vector store** | `text-embedding-3-small`; índices `fibo-glossary-3m-vec`, `cfr-sbvr-3m-vec` no AllegroGraph para busca semântica |
| **SEMSCORE** | Métrica de similaridade semântica (Aynetdinov & Akbik, 2024) entre statement transformado e original |
| **LLM-as-a-judge** | Um LLM avalia a qualidade da transformação (similarity, accuracy, grammar, findings) |
| **Levenshtein distance** | Distância de edição (via `jellyfish`) usada no app para comparar redações |
| **Checkpoint** | Arquivo JSON com o estado do `DocumentManager` ao fim de um estágio |
| **True table / golden dataset** | Gabarito curado manualmente (`documents_true_table.json`) |
| **P1 / P2** | Passagem 1 (elementos) e Passagem 2 (definições de termos) da extração |
| **Knowledge Graph (KG)** | Grafo RDF no AllegroGraph integrando FIBO + CFR + SBVR |

## 3. Componentes e serviços

| Termo | Significado |
|-------|-------------|
| **AllegroGraph (AG)** | Triplestore RDF da Franz; hospeda o KG e os índices vetoriais; CLI `agtool` |
| **agraph-python / franz** | Cliente Python para o AG (`franz.openrdf`) |
| **stunnel** | Túnel TLS para acessar o AG Cloud na porta 443 via `localhost:8443` |
| **DuckDB** | Banco analítico embarcado; alimenta o app com as views |
| **MotherDuck** | DuckDB serverless na nuvem (`md:cfr2sbvr_db`) |
| **Streamlit** | Framework do app de inspeção (`cfr2sbvr_inspect`) |
| **SME** | Subject Matter Expert — especialista que revisa/valida via app |
| **Instructor `create_with_completion`** | Chamada que retorna objeto validado + metadados de completion |

## 4. Convenções de identificadores

| Padrão | Exemplo | Onde |
|--------|---------|------|
| Seção CFR | `§ 275.0-2` | `doc_id` |
| Documento de extração | `§ 275.0-2_P1`, `§ 275.0-2_P2` | `id` de `llm_response` |
| Documento de classificação | `classify_P1`, `classify_P2_Operative_rules`, `classify_P2_Definitional_terms` | `id` de `llm_response_classification` |
| Documento de transformação | `transform_Operative_Rules`, `transform_Terms` | `id` de `llm_response_transform` |
| Documento de validação | `validation_judge_Terms` | `id` de `llm_validation` |
| Template de Witt | `T1`, `T7` | `witt_templates.yaml` |
| Subtemplate | `S1` | `witt_subtemplates.yaml` |
| Exemplo de regra / fact type | `R70`, `F188` | `witt_examples.yaml` |
| Checkpoint | `documents-2024-12-08-3.json` | `data/checkpoints*/` |
| Chave serializada de documento | `§ 275.0-2_P1|true_table` | JSON checkpoint |

## 5. Siglas rápidas

CFR (Code of Federal Regulations) · SBVR (Semantic Business Vocabulary and Rules) · FIBO (Financial
Industry Business Ontology) · FRO (Financial Regulation Ontology) · KG (Knowledge Graph) · CNL
(Controlled Natural Language) · LLM (Large Language Model) · SME (Subject Matter Expert) · AG
(AllegroGraph) · OMG (Object Management Group) · SEC (Securities and Exchange Commission) · P1/P2
(passagens de extração) · VW (view SQL).

Continua em [06-modernizacao.md](06-modernizacao.md).
