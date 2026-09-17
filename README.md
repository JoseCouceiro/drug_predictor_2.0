# Drug & ATC Classification from Molecular Structure

A transfer-learning pipeline that goes from a large-scale **Lipinski "drug-likeness" model**
to two downstream classifiers — a **drug vs. non-drug predictor** and an **ATC
(Anatomical Therapeutic Chemical) class predictor** — built entirely from molecular
fingerprints/descriptors (no external protein or assay data).

Built with [Kedro](https://kedro.org/) (pipelines/nodes/catalog), RDKit (featurization),
and TensorFlow/Keras (Conv1D transfer-learning backbone).

---

## 1. Foundation model: Lipinski Rule-of-Five classifier

- **Source data:** bulk compound records from PubChem (`Compound_*.sdf.gz` partitions).
- **Training set:** **277,686 molecules**, featurized (RDKit fingerprints/descriptors)
  and balanced across Lipinski-compliant / non-compliant classes, processed in
  partitions to fit in memory.
- **Task:** binary classification — does a molecule satisfy Lipinski's Rule of Five
  (drug-likeness heuristic based on molecular weight, LogP, H-bond donors/acceptors)?
- **Architecture:** Conv1D-based network over the flattened fingerprint vector.
- **Result:** **~97% validation accuracy** (precision/recall/F1 ≈ 0.97 for both classes,
  26,393-molecule held-out validation split).

This model's Conv1D "fingerprint reader" backbone is frozen and reused as a transfer-learning
base for the two downstream tasks below — the idea being that a network trained to recognize
general structural/physicochemical patterns across hundreds of thousands of molecules should
transfer better than training a small classifier from scratch on a much smaller labelled set.

## 1b. Multi-task Lipinski pretraining: does a richer pretext task help?

The single-task backbone above is only ever taught one thing: a binary Rule-of-Five
label, itself just a threshold rule over 4 scalar properties (MW, LogP, HBD, HBA).
That's a narrow pretext task — the backbone's easiest path to low loss is to learn
to "read off" those 4 numbers from the fingerprint, not to build a broadly useful
structural representation. To test this, a second backbone was pretrained on the
same Conv1D trunk with **5 output heads at once**, sharing one representation:

| Head | Type | Result |
|---|---|---|
| RuleFive (Lipinski compliance) | binary | accuracy 0.97 (identical to single-task) |
| QED (drug-likeness score) | regression | MAE 0.050, **R² 0.915** |
| NumRotatableBonds | regression | MAE 0.97, **R² 0.888** |
| NumAromaticRings | regression | MAE 0.34, **R² 0.883** |
| FractionCSP3 (sp3 carbon fraction) | regression | MAE 0.054, **R² 0.907** |

The backbone learned all 4 auxiliary structural properties to a high standard
(R² 0.88–0.92) **without losing any RuleFive performance** — strong evidence it
built a genuinely richer shared representation, not just a wider model.

## 2. Downstream task 1: Drug vs. non-drug classifier

- **Dataset:** 19,793 molecules — 9,954 real drugs vs. 9,839 non-drug decoys.
- **Approach:** frozen Lipinski Conv1D backbone + a new trainable dense head, binary
  focal loss, decision threshold tuned on the validation set (optimal threshold ≈ 0.10).
- **Result:** macro-F1 ≈ 0.66 with the single-task backbone, **≈ 0.665 with the
  multi-task backbone** — a small but consistent improvement.

## 3. Downstream task 2: ATC class classifier

- **Dataset:** the same 9,954 drug molecules, labelled with one of 16 ATC top-level
  codes (a 17th, "V" = "Various", was excluded — it's a heterogeneous catch-all with
  no coherent chemical identity, so it only added label noise).
- **Approach:** same frozen backbone + a dedicated dense head; extensively tuned
  (loss function, class weighting, label smoothing, dropout, batch size) across many
  iterations before declaring a plateau.
- **Result (full 15-class taxonomy):** accuracy ≈ 0.74, macro-F1 ≈ 0.62–0.63 —
  noticeably weaker than the Lipinski model, and it didn't improve much further
  with more tuning.

### Why the ATC classifier underperforms

1. **Mixed taxonomy axis.** The ATC top-level code mixes two very different
   classification criteria: *anatomical/organ system* targeted (e.g. `C` = cardiovascular,
   `N` = nervous system) and *pharmacological action* (e.g. `J` = anti-infective,
   `H` = hormonal). Two structurally similar molecules can land in different ATC
   classes simply because they act on different organs, even if their mechanism is
   the same — and structure alone can't recover physiological targeting.
2. **Severe class imbalance.** Per-class support in the full taxonomy ranges from
   ~94 (`H`) to ~2,589 (`J`), a >27x imbalance.
3. **Small dataset relative to class count.** ~9,600 molecules spread over 15 classes
   leaves some classes with only a few dozen validation examples.
4. **GPU memory constraints** repeatedly forced smaller batch sizes/architectures during
   development (`ResourceExhaustedError` on the training GPU), limiting how large a head
   could be explored.

## 4. Experiment: does splitting the taxonomy help?

Given cause #1 above, we tested whether splitting the 15 ATC classes along their
natural taxonomy axis — **action-based** vs. **organ-based** — makes the label space
more learnable from structure alone, and whether the multi-task Lipinski backbone
(section 1b) compounds that improvement.

> **Note:** this dataset repurposes the letters 'I' and 'O' for its own categories
> (`I` = "Antiinflammatory", `O` = "Lipid regulation" — verified against
> `drug_raw['MATC_Code_Explanation']`), not the standard WHO ATC meanings. Both are
> mechanism/action categories, not anatomical ones. An earlier version of this split
> put them in the organ-based group (and misplaced `A`/`B`, two genuine anatomical
> categories, in the action-based group) purely because of a wrong assumption from
> their letters. The table below reflects the corrected split, based on each code's
> actual meaning in this dataset:
> - **Action-based** (mechanism): `H, I, J, L, O, P`
> - **Organ-based** (anatomical): `A, B, C, D, G, M, N, R, S`

| Configuration | Classes | Samples | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| Full taxonomy + single-task backbone | 15 | 9,649 | 0.7425 | 0.6213 | 0.7475 |
| Full taxonomy + multi-task backbone | 15 | 9,649 | 0.7399 | 0.6278 | 0.7426 |
| Organ-based + single-task backbone | 9 | 4,975 | 0.7266 | 0.6732 | 0.7370 |
| **Organ-based + multi-task backbone** | 9 | 4,975 | 0.7347 | 0.6728 | 0.7348 |
| Action-based + single-task backbone | 6 | 4,674 | 0.8727 | 0.7841 | 0.8815 |
| **Action-based + multi-task backbone** | 6 | 4,674 | **0.8695** | **0.7842** | **0.8797** |

### Conclusion

Both taxonomy subsets clearly outperform the mixed full taxonomy, confirming that the
organ/action mix was diluting the learnable signal. Between the two, **action-based
classification is substantially more learnable from molecular structure than
organ-based classification** (macro-F1 ~0.78 vs. ~0.67): a drug's *mechanism of
action* (anti-infective, analgesic, hormonal, anti-inflammatory, etc.) is far more
tied to its chemical structure than *which organ system* it's routed to, which
depends on pharmacokinetics/physiology that structure alone doesn't fully determine.

Unlike the full taxonomy (where the multi-task backbone gives a small but real bump,
0.6213 → 0.6278 macro-F1), the multi-task backbone makes essentially no difference
for either the action-based or organ-based split once the taxonomy itself is
correct (differences are within normal training-run variance). Overall:
**restructuring the label taxonomy to match what the input features can actually
predict was by far the bigger lever** — getting that split right (both in which
codes go where, and in the actual per-code meaning behind this dataset's ATC
letters) mattered far more than which pretrained backbone was used.

## 5. Sanity check: testing on molecules approved after training

As an out-of-distribution sanity check, 10 real-world drugs approved/in late-stage
approval in 2026 (`drugpredictor2/data/moleculas_aprobadas_2026_classified.csv`) were
run through the app (fetched live from PubChem by CID), each hand-labelled with an
expected action/organ code from its known mechanism/target tissue, and compared
against the model's predictions:

| Metric | Accuracy |
|---|---|
| Action-based, top-1 | 33.3% (3/9; 1 molecule has no clean fit in the 6-class action taxonomy) |
| Action-based, top-3 | 44.4% |
| Organ-based, top-1 | 20.0% (2/10) |
| Organ-based, top-3 | 50.0% |
| Mean recalibrated drug-probability | 38.6% (all 10 are real approved drugs) |

Substantially below the validation-set macro-F1 (~0.78 action, ~0.67 organ) — expected,
since these are brand-new 2026 approvals with mechanisms that postdate the training
data. Splitting the 10 by how chemically novel each one is (data source's own
"Derivative Scaffold/Next-in-class" vs. "Novel Scaffold/First-in-class" labels)
sharpens the picture:

| Scaffold type | N | Action top-1 | Organ top-1 |
|---|---|---|---|
| Derivative / known chemotype | 7–8 | **42.9%** | **25.0%** |
| Novel / first-in-class chemotype | 2 | **0%** | **0%** |

The two genuinely first-in-class molecules (Orforglipron, a novel oral non-peptide
GLP-1 agonist; Oveporexton, an orexin-2 receptor agonist) missed on every metric,
while derivative/"me-too" compounds (following known BTK/PDE4/JAK/SERD/KRAS-inhibitor
chemotypes) fared meaningfully better. This is consistent with the model failing where
it should be expected to fail — on chemistry with no precedent in its training
distribution — rather than failing randomly. The sample is small (n=10, only 2
"novel"), so this is a directional signal, not a statistically robust conclusion.

## How to run

The project is a set of [Kedro](https://kedro.org/) pipelines. Two config parameters
control which experiment you get, and both branches can also be produced in a single
run without touching either parameter:

- `atc_subset` (in `conf/base/parameters/process_drug_data.yml`) — which ATC codes
  the **default** drug/ATC pipeline branch trains on:
  - `null` → global/full 15-class taxonomy
  - `"H,I,J,L,O,P"` → action-based subset (6 classes)
  - `"A,B,C,D,G,M,N,R,S"` → organ-based subset (9 classes)

  > **Pitfall:** this parameter must be `null` for the default branch's numbers
  > to mean anything for the drug/no-drug classifier. Any ATC subset excludes
  > `ND` (non-drug) rows entirely, since `ND` isn't an ATC code — leaving
  > `y_drug_train` 100% positive with no negative examples to learn from. This
  > happened once during development (`atc_subset` left set to the action-based
  > codes from an earlier experiment) and silently collapsed the drug
  > classifier to predicting ~the same probability for every molecule. Always
  > confirm it's `null` before training/retraining `drug_classifier_model` or
  > the default (non-prefixed) `atc_classifier_model`.
- `backbone_source` (in `conf/base/parameters/multitask_model.yml`) — which pretrained
  Lipinski backbone the drug/ATC classifiers transfer from:
  - `"single"` → RuleFive-only pretraining
  - `"multitask"` → RuleFive + QED + 3 descriptors pretraining (section 1b)

### 1) Train the Lipinski backbone(s) (once)

```bash
kedro run --pipeline build_model
```
Trains both `lipinski_model` (single-task) and `lipinski_multitask_model`
(multi-task) in the same run.

### 2) Pick a taxonomy for the default branch, then train drug/ATC classifiers

**Global (full 15-class) taxonomy**, no file edits — via CLI:
```bash
kedro run --pipeline process_drug_data --params atc_subset:null
kedro run --pipeline multitask_model
```

**Action-based** subset — edit `atc_subset: "H,I,J,L,O,P"` in
`conf/base/parameters/process_drug_data.yml` (a comma-containing value can't be
passed via `--params`, since Kedro splits that flag on commas), then:
```bash
kedro run --pipeline process_drug_data
kedro run --pipeline multitask_model
```

**Organ-based** subset: same as above, with
`atc_subset: "A,B,C,D,G,M,N,R,S"` in the YAML file.

**Choosing the backbone** for whichever branch you just ran: edit
`backbone_source` in `conf/base/parameters/multitask_model.yml` to `"single"` or
`"multitask"` before the `multitask_model` run (no comma issue, so
`--params backbone_source:multitask` also works directly on the CLI).

### 3) Get action-based AND organ-based results in one go

You don't need to touch `atc_subset` at all for this — `process_drug_data` and
`multitask_model` always additionally run two fixed branches (hardcoded to the
action/organ code lists) alongside whatever the default `atc_subset`-driven branch
produces:

```bash
kedro run --pipeline process_drug_data
kedro run --pipeline multitask_model
```

This produces, in a single pair of runs:
- `data/08_reporting/atc_classifier_val_report.txt` — whatever `atc_subset` is set to
- `data/08_reporting/action_atc_classifier_val_report.txt` — always action-based
- `data/08_reporting/organ_atc_classifier_val_report.txt` — always organ-based

All three branches (default/action/organ) train against the same selected backbone,
so the comparison in the table above is a fair, like-for-like one.

## Demo app

A Streamlit app (`drugpredictor2/apps/drug_predictor/`) wraps the three trained
classifiers (`drug_classifier_model`, `action_atc_classifier_model`,
`organ_atc_classifier_model`) for interactive use — enter a PubChem CID or a
SMILES string and get the drug-likeness probability plus top-3 action-based and
organ-based ATC class predictions (top-3, not just the argmax, since a single
molecule can legitimately straddle multiple real-world ATC codes). It also
supports batch prediction from a CSV of CIDs/SMILES. Run it with:

```bash
cd drugpredictor2/apps/drug_predictor
streamlit run drug_predictor.py
```

First load takes about a minute (three Keras models loaded from disk); models
are cached in memory afterwards via `st.cache_resource`, so subsequent
interactions are instant.

## Tech stack

- **Kedro** — pipeline orchestration (`process_drug_data`, `build_model` /lipinski,
  `multitask_model`)
- **RDKit** — molecular featurization (fingerprints, descriptors)
- **TensorFlow / Keras** — Conv1D transfer-learning backbone + task-specific dense heads
- **Streamlit** — interactive demo app for the trained classifiers
