# Research datasets corresponding to manuscript Table 1

This directory contains research records, not invented demo fixtures. The author
requested release of all three Table 1 datasets. No participant access codes,
account identifiers, authentication tokens, ratings timestamps, free-text rater
notes, consent records, audit trail, API ledger or provider request identifiers
are distributed. The SQLite database was constructed anew from allowlisted fields;
it is not a copy of the operational rating database.

## Files and units

- `classifier_pairs_2400.jsonl`: original QA-pair IDs, Chinese question/answer text,
  label (1=related, 0=unrelated), primary partition, available origin/source/group
  metadata, question/answer labels, synthetic flags and researcher review status.
  Nullable synthetic flags mean unspecified, not verified non-synthetic.
- `classifier_text_partitions.jsonl`: actual classifier input rows keyed by protocol,
  task, partition and row ID. Random partitions preserve original within-partition
  deduplication; grouped partitions are separate supplementary inputs. Labels have
  the same 0/1 meaning. Group and pair IDs are data linkage identifiers, not people.
- `balanced_questions_240.jsonl`: questions, reference answers, related/unrelated
  category, expected action (`answer`/`abstain`) and available provenance metadata.
- `balanced_outputs_5040.jsonl`: 240 questions x 21 configurations, final displayed
  answers, status, switches and filter-stage trace. Repeated answer strings are
  intentionally preserved because configurations share generation calls.
- `short_questions_308.jsonl`: only the 308 completed study questions, not the
  remaining unused questions in the larger bank. Their intended scope is related;
  this is not an independently adjudicated topic label.
- `short_outputs_2156.jsonl`: 308 questions x seven configurations, answers,
  statuses, module switches, traces and recorded answer-length information.
- `short_ratings_2156.csv`: one final three-axis rating vector per question/system.
  Appropriateness, clarity and helpfulness are integers from 1 to 5. This file does
  not contain five independent ratings per output. Individual assessor assignments
  cannot be reconstructed. No participant IDs or notes are included.
- `table1_research_data.sqlite3`: the same data in seven relational tables, suitable
  for offline SQL inspection; open read-only. It contains no users/login/consent tables.
- `DATASET_COUNTS.json`: expected counts; `DATA_MANIFEST.json`: export file hashes.

## Primary and supplementary partitions

The primary 2,400 pairs were stratified 80/10/10 by label with seed 42 (1,920/240/240
pairs). Question and answer texts were then deduplicated separately within splits:
question rows 1,828/240/240; answer rows 1,414/226/228. These totals do not imply
cross-split independence. Supplementary approximately 60/20/20 grouped partitions
contain question rows 1,352/457/445 and answer rows 1,016/332/336. They account for
known source/template links, normalized duplicates and character-trigram similarity
>=0.9. Related tasks and partition protocols are not pooled as independent samples.

## Output and rating interpretation

`accepted` means released by the configured pipeline. `query_rejected`,
`generation_exhausted` and `refinement_exhausted` denote fixed fallback outputs at
question, draft and revised-answer checks respectively; the latter names are
historical implementation statuses, not repeated generation attempts.
`transformer` in system IDs denotes frozen BGE; `word2vec` denotes Word2Vec-MLP.
The 21 balanced configurations and seven short-answer configurations are different
experiments. The balanced study evaluates acceptance/withholding decisions, not
independent doctrinal safety. The short-answer ratings are pooled contributions
from five monastics without recoverable individual assignments. The three risk
annotation fields were unassessed and are not treated as negative safety labels.
Fallback message ratings are retained, including HY073's 5/5/5.

The corpus is partly synthetic, includes source-grounded paraphrases and shared
source/template families, and is researcher-reviewed. Text is learning material;
AI paraphrases and generated answers must not be attributed as authenticated
quotations or official institutional statements. Source titles describe recorded
provenance and are not independent verification of every attribution. Data overlap,
ceiling scores, shared generations and post-hoc accepted-subset analyses limit
population-level and causal interpretations. See the manuscript for study context.
No new blanket license is granted over third-party source material; see LICENSE_NOTICE.md.
