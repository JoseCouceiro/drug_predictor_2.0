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

| Configuration | Classes | Samples | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| Full taxonomy + single-task backbone | 15 | 9,649 | 0.7425 | 0.6213 | 0.7475 |
| Full taxonomy + multi-task backbone | 15 | 9,649 | 0.7399 | 0.6278 | 0.7426 |
| Organ-based + single-task backbone | 9 | 4,220 | 0.7595 | 0.7057 | 0.7612 |
| **Organ-based + multi-task backbone** | 9 | 4,220 | 0.7524 | 0.6961 | 0.7559 |
| Action-based + single-task backbone | 6 | 5,429 | 0.8287 | 0.7522 | 0.8336 |
| **Action-based + multi-task backbone** | 6 | 5,429 | **0.8370** | **0.7658** | **0.8412** |

### Conclusion

Both taxonomy subsets clearly outperform the mixed full taxonomy, confirming that the
organ/action mix was diluting the learnable signal. Between the two, **action-based
classification is more learnable from molecular structure than organ-based
classification**: a drug's *mechanism of action* (anti-infective, analgesic, hormonal,
etc.) is far more tied to its chemical structure than *which organ system* it's
routed to, which depends on pharmacokinetics/physiology that structure alone doesn't
fully determine.

The multi-task backbone (section 1b) then adds a further, smaller improvement on top
of the action-based split (macro-F1 0.7522 → 0.7658), and a similar small improvement
on the full taxonomy (0.6213 → 0.6278). It doesn't help the organ-based split, which
sits about the same either way. Overall: **restructuring the label taxonomy to match
what the input features can actually predict was by far the bigger lever**, and the
richer multi-task pretraining objective is a smaller, mostly-additive improvement on
top of it.

## How to run

The project is a set of [Kedro](https://kedro.org/) pipelines. Two config parameters
control which experiment you get, and both branches can also be produced in a single
run without touching either parameter:

- `atc_subset` (in `conf/base/parameters/process_drug_data.yml`) — which ATC codes
  the **default** drug/ATC pipeline branch trains on:
  - `null` → global/full 15-class taxonomy
  - `"J,L,B,A,P,H"` → action-based subset (6 classes)
  - `"N,C,R,D,G,S,M,O,I"` → organ-based subset (9 classes)
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

**Action-based** subset — edit `atc_subset: "J,L,B,A,P,H"` in
`conf/base/parameters/process_drug_data.yml` (a comma-containing value can't be
passed via `--params`, since Kedro splits that flag on commas), then:
```bash
kedro run --pipeline process_drug_data
kedro run --pipeline multitask_model
```

**Organ-based** subset: same as above, with
`atc_subset: "N,C,R,D,G,S,M,O,I"` in the YAML file.

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

## Tech stack

- **Kedro** — pipeline orchestration (`process_drug_data`, `build_model` /lipinski,
  `multitask_model`)
- **RDKit** — molecular featurization (fingerprints, descriptors)
- **TensorFlow / Keras** — Conv1D transfer-learning backbone + task-specific dense heads
