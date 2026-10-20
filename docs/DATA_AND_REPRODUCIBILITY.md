# Data contract and reproducibility boundaries

## Two separate rating workstations

The six-core-system contract and balanced allocation below describe `rating_app.server`.
The public seven-system demo uses `rating_app.comparison`, with a separate study folder
and one extra Word2Vec B2 configuration. See the README for its commands. The 308-question
research assessment used a shared account and must not be described using the balanced
24-person default allocation. Never infer a participant count from output-row counts.

## Rating input

UTF-8 JSONL; one record for each question/system. Required keys: id, question, answer, category (related/unrelated), system, status. replicate must be zero or omitted. Accepted and bounded-refusal answers are retained. Provider errors and duplicate system keys are rejected.

Core systems: generic_b1, domain_b1, domain_refinement_only, word2vec__full, tfidf__full, transformer__full. Every item requires all six systems, the same question text and category. The display contains neither system names, source ids nor references. Identical text across systems is not silently deduplicated.

Use only texts you are permitted to process. The synthetic demo contains no original research answers. prepare creates a new private directory and refuses to overwrite it. Selection and assignments freeze before human scores are collected.

## Human ratings

Three 1–5 axes: appropriateness (primary), clarity, helpfulness. unable and skipped remain missing, not zero. Risk observations are unassessed/yes/no/uncertain; general raters do not replace expert adjudication. Complete submissions lock further editing. Saves retain a versioned audit trail. Exports include the mapping hash, study id, original response hash and submission status.

The default 24-person allocation uses 120 class-balanced questions, 10 per person, six systems per question; each selected question appears to two people if everyone completes. This is a workload allocation, not a power justification. Dependent responses must not be treated as independent participants. Confirm design and recruitment before activation. Ethics/consent text must satisfy the research institution; this software makes no ethics-approval claim.

## Research data release and reproducibility

The three manuscript Table 1 datasets are now included under `research_data/`.
See its README for schemas, record units and known dependence. A new public SQLite
analysis database mirrors the JSONL/CSV exports. Private operational databases,
secrets, consent/audit records, model weights and provider logs remain excluded.

`python3 tools/verify_public_data.py` checks export/database consistency and
recalculates descriptive Tables 4, 5 and 7 values without scientific dependencies.
Classifier retraining needs the declared packages; original historical scripts
retain experiment paths and may need adaptation. Hosted-model answers are released
as recorded outputs, not a guarantee that rerunning an API reproduces identical text.
Do not infer assessor-level data or independent generation replicates from output rows.
