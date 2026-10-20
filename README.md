# FGS religious dialogue filtering and refinement

Research and demonstration source for the third manuscript revision: topic classifiers,
bounded generation/refinement pipelines, ablation evaluation, statistics, and local
anonymous-code human rating. The application is a learning-assistant demo, not an
authenticated religious authority.

## Seven-configuration local demo — no API key or dependencies

Python 3.10 or newer is required. In a terminal in this folder:

```bash
python3 tools/make_seven_rating_demo.py
python3 -m rating_app.comparison --study studies/demo7/auth --bank studies/demo7/bank.json --responses studies/demo7/responses.jsonl --out studies/demo7/comparison --port 8771
```

Open http://127.0.0.1:8771/ (researcher page: /admin). Local codes are generated in
`studies/demo7/auth/PRIVATE_participant_codes.csv` and `PRIVATE_admin_code.txt`.
Use a different participant code for every real person. Demo content is entirely
invented and prominently labeled. It has four questions and seven answers per
question; it is not any of the 308 research question sets or empirical results.
Save each answer rating, log out to change participants, and use the researcher
export to inspect the demo results. The demo uses existing local fixture answers;
it cannot generate new answers and requires no model or network service.

Do not upload the generated studies folder. The 8771 demo port is separate from
the author's private 8770 research workstation.

## Classifiers and research analyses

```bash
python3 -m pip install -e .
# Optional contextual encoder and figures:
python3 -m pip install -e ".[transformer,plots]"
```

`fgs/` implements train-only features, validation checkpoints, TF-IDF/Word2Vec/
Transformer classifiers, stage switches, final-answer checks and abstention.
`tools/supplement.py` and `supplement_finish.py` describe grouped sensitivity,
learning curves, negative controls and cached ablations. `tools/summarize_final308.py`
documents the frozen-rating audit and paired exploratory analysis.
The historical research tools retain their original experiment paths and may
require adaptation, additional dependencies, model weights and protocol files.
The released data support independent inspection and reconstruction of the
reported descriptive ratings and pipeline decisions; see the offline verification
command below. Full classifier retraining and hosted-model generation are
separate operations and were not rerun for this release.

The frozen private 308-question service (`rating_app.final_rating`) is disclosed
for transparency. Its private operational layout, secrets and credentials are
not distributed. The public research database is an analysis export, not a
drop-in operational study database. Run the synthetic demo above.
The private rating-service UI now uses direct question entry without a visible question-bank selector. It retrieves stored answers, does not generate arbitrary new answers, and restores an existing consent record without creating another consent event. Figures 3–4 show this updated interface; the synthetic 8771 demo remains a separately labeled demonstration.
Historical API adapters remain source documentation; no live requests are needed
for the demos or existing-cache analysis.

## Research datasets from Table 1

All three datasets are included in `research_data/`:

| Dataset | Released contents |
|---|---|
| Classifier development | 2,400 QA pairs (1,200 related / 1,200 unrelated), original pair partition membership, deduplicated question/answer random partitions and supplementary grouped partitions |
| Balanced pipeline evaluation | 240 questions (120 related / 120 unrelated), references, 21 configurations and 5,040 recorded outputs with acceptance statuses |
| Short-answer human assessment | 308 questions, seven configurations, 2,156 recorded outputs and 2,156 three-axis rating vectors (6,468 numeric scores) |

Read the data dictionary and limitations in `research_data/README.md`.
The following read-only verification needs only Python's standard library:

```bash
python3 tools/verify_public_data.py
```

It checks database integrity, partition counts, JSONL/CSV consistency and the
published descriptive values for accepted outputs, all outputs, common accepted
questions and balanced pipeline conformity. It does not call an API or train models.

## Revised study and limitations

The redesigned corpus contains 2,400 partly synthetic Chinese QA pairs. The
balanced pipeline comparison uses 240 questions; later short-answer assessment
uses 308 questions and seven configurations. Five Fo Guang Shan monastics shared
one login in that assessment. Individual rater assignments were not retained;
the study must not be described as five independent ratings per output.
Topic acceptance is not doctrinal correctness or safety. The new human-rating
results do not establish a quality advantage of the full Word2Vec pipeline.
Source overlap, post-hoc analyses, ceiling scores, unassessed risk fields and
shared cached draws are documented limitations.

## Package scope

Automated tests are maintained in the private development workspace and are not
required to run this release. Historical six-system fixture generation, bulk
prefetch and budget-estimation helpers are omitted. The retained `tools/` files
provide the runnable seven-configuration demo and manuscript analysis protocols;
they are not imported by the rating server.

## Contents and licensing

This release includes allowlisted source, example configuration, documentation,
dependency declarations, and the three research datasets described in manuscript
Table 1. The `research_data/` directory contains a newly constructed public
SQLite database and matching JSONL/CSV exports. Operational rating databases,
participant identifiers, access codes, authentication tokens, consent and audit
records, API keys, model weights, provider logs, manuscript files and previous
Git history are excluded.
`PUBLIC_MANIFEST.json` lists the exact source files and SHA-256 hashes.

No new software license grant is inferred: see LICENSE_NOTICE.md. Source can be
published with rights reserved; select a license explicitly before describing it
as an open-source release. Third-party models and dependencies retain their terms.
This package has not been uploaded to GitHub by its builder.
