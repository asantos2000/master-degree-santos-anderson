# Held-out generalization probe: 17 CFR § 275.202(a)(11)(G)-1

Design: frozen pipeline (prompts, gpt-4o, temperature 0), executed once, with no gold-standard correction at any checkpoint. All indicators are reference-free: they require only the source section, not an annotated reference.

## Extracted elements (uncorrected)

- operative_rules: 5
- fact_types: 1
- terms_with_definition: 15
- names_with_definition: 0

## Indicator 1: grounding (hallucination) check

Best fuzzy alignment of each extracted statement/definition against the source text (1.0 = present near-verbatim).

| Group | n | mean | median | min | share >= 0.8 |
|---|---|---|---|---|---|
| Operative Rules | 5 | 0.8531 | 0.8149 | 0.7831 | 0.6 |
| Fact Types | 1 | 0.4245 | 0.4245 | 0.4245 | 0.0 |
| Terms | 15 | 0.765 | 0.832 | 0.3813 | 0.5333 |
| **Overall** | 21 | 0.7697 | 0.8149 | 0.3813 | 0.5238 |

## Indicator 2: SemScore (original vs. transformed, embedding cosine)

n = 21, mean = 0.8938, median = 0.9041, min = 0.797, share >= 0.8 = 0.9524

## Indicator 3: LLM-as-a-Judge (against source statement and templates)

| Group | n | similarity (mean) | transformation acc. (mean) | grammar (mean) |
|---|---|---|---|---|
| Operative Rules | 5 | 0.88 | 0.85 | 0.97 |
| Fact Types | 1 | 0.95 | 0.95 | 1.0 |
| Terms | 15 | 0.9133 | 0.86 | 0.9467 |
| **Overall** | 21 | 0.9071 | 0.8619 | 0.9548 |

Statements below the acceptance threshold (0.8), flagged for SME review: 1 (details in indicators.json).

## Cost

Input tokens: 208962; output tokens: 17557 (gpt-4o).

## Caveats

- Single run (n=1): no dispersion estimate, unlike the ten-run protocol of the main experiment.
- LLM-as-a-Judge and SemScore measure semantic preservation and template conformity, not extraction recall or classification correctness; those require an annotated reference and remain future work.
- The judge shares the underlying model family with the pipeline; self-preference bias cannot be excluded. Read the probe as indicative, not as validation.