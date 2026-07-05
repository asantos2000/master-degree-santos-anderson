# Held-out generalization probe: 17 CFR § 275.202(a)(11)(G)-1 (Family offices)

Isolated experiment that runs the FROZEN CFR2SBVR pipeline a single time, with no
gold-standard correction at any checkpoint, on a section not used anywhere in the
original experiment, and reports reference-free indicators as an indicative probe
of generalization. It responds to Reviewer 2's circularity concern (R2.6) within
the resource limits stated in Section 6.2 of the article: no new expert-annotated
reference is required.

## Design

Frozen artifacts, copied verbatim from the original notebooks (no re-tuning):

| Artifact | Source |
|---|---|
| Extraction prompts P1/P2 + response models | `src/chap_6_semantic_annotation_elements_extraction.ipynb` (`system_prompt_v4_1`, `system_prompt_v4_2`) |
| Classification prompts P1/P2 + response models | `src/chap_6_semantic_annotation_rules_classification.ipynb` |
| Transformation prompts + template formulation | `src/chap_6_nlp2sbvr_transform.ipynb` |
| Judge prompts + SemScore procedure | `src/chap_7_validation_rules_transformation.ipynb` |
| Taxonomy/templates data | `data/classify_subtypes.yaml`, `data/witt_templates.yaml`, `data/witt_examples.yaml` |
| LLM parameters | gpt-4o, temperature 0, max_tokens 8192 (config.yaml) |

Differences from the main experiment, by design:

1. Single run (the main experiment ran each transformation ten times).
2. No gold-standard reset of stage inputs: each stage consumes the previous
   stage's uncorrected output (end-to-end, no simulated SME intervention).
3. Only reference-free indicators are computed. Precision/recall/F1 and
   gold-referenced SemScore are NOT computed (no gold standard exists for this
   section). One necessary deviation: elements that the classification stage
   fails to assign `templates_ids` are skipped at transformation and logged,
   since the transformation prompt requires a template.

## Reported indicators (all reference-free)

1. **Grounding (hallucination) check** — best fuzzy alignment of each extracted
   statement and definition against the source section text (1.0 = present
   near-verbatim). Mirrors the hallucination check of Section 5.3 of the article.
2. **SemScore** — embedding cosine similarity (text-embedding-3-large) between
   each original statement/definition and its SBVR-SE transformation, exactly as
   in the transformation-validation notebook (original vs. transformed, not gold).
3. **LLM-as-a-Judge** — similarity, transformation accuracy, and grammar/syntax
   scores of each transformation against the source statement and the writing
   templates, using the frozen judge prompts. Items below the 0.8 acceptance
   threshold are flagged, as in the article's checkpoint design.

## How to run

From the environment used for the original notebooks, or a fresh one with the
pinned dependencies (Python 3.10+):

```bash
cd code/experiments/heldout_275_202a11G_1
pip install -r requirements.txt
python run_probe.py --selftest   # sanity check, no API calls
python run_probe.py              # full probe
```

`OPENAI_API_KEY` is read from the environment or from the repo-root `.env`.
The script is resumable: completed stages are skipped on re-run (checkpoint in
`outputs/probe_checkpoint.json`). Delete `outputs/` for a clean run. Expected
runtime is roughly 10-25 minutes; expected cost is a few USD (single pass,
one section of 1,285 words).

## Outputs

- `outputs/probe_checkpoint.json` — full pipeline state (all intermediate artifacts, auditable)
- `outputs/grounding.json`, `outputs/semscore.json`, `outputs/judge.json` — per-item indicator records
- `outputs/indicators.json` — aggregated indicators
- `outputs/report.md` — human-readable report
- `outputs/usage.json` — token usage per call

## Interpretation caveats (also printed in report.md)

Single run, no dispersion estimate; the indicators measure source grounding,
semantic preservation, and template conformity, not extraction recall or
classification correctness; the judge shares the model family with the pipeline,
so self-preference bias cannot be excluded. The probe is indicative, not a
substitute for held-out validation with an independent expert-annotated reference.
