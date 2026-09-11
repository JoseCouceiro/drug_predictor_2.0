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

## 2. Downstream task 1: Drug vs. non-drug classifier

- **Dataset:** 19,793 molecules — 9,954 real drugs vs. 9,839 non-drug decoys.
- **Approach:** frozen Lipinski Conv1D backbone + a new trainable dense head, binary
  focal loss, decision threshold tuned on the validation set (optimal threshold ≈ 0.10).
- **Result:** macro-F1 ≈ 0.66 on the held-out validation split.

## 3. Downstream task 2: ATC class classifier

- **Dataset:** the same 9,954 drug molecules, labelled with one of 16 ATC top-level
  codes (a 17th, "V" = "Various", was excluded — it's a heterogeneous catch-all with
  no coherent chemical identity, so it only added label noise).
- **Approach:** same frozen backbone + a dedicated dense head; extensively tuned
  (loss function, class weighting, label smoothing, dropout, batch size) across many
  iterations before declaring a plateau.
- **Result (full 15-class taxonomy):** accuracy ≈ 0.74, macro-F1 ≈ 0.62 — noticeably
  weaker than the Lipinski model, and it didn't improve much further with more tuning.

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
4. **Structure-only features.** The model only sees molecular fingerprints/descriptors —
   no biological/target/pathway information — which caps how much of the label signal
   is recoverable at all.
5. **GPU memory constraints** repeatedly forced smaller batch sizes/architectures during
   development (`ResourceExhaustedError` on the training GPU), limiting how large a head
   could be explored.

## 4. Experiment: does splitting the taxonomy help?

Given cause #1 above, we tested whether splitting the 15 ATC classes along their
natural taxonomy axis — **action-based** vs. **organ-based** — makes the label space
more learnable from structure alone.

| Configuration | Classes | Samples | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| Full taxonomy (mixed) | 15 | 9,649 | 0.7425 | 0.6213 | 0.7475 |
| **Organ-based** (`C,D,G,I,M,N,O,R,S`) | 9 | 4,220 | 0.7595 | 0.7057 | 0.7612 |
| **Action-based** (`A,B,H,J,L,P`) | 6 | 5,429 | **0.8287** | **0.7522** | **0.8336** |

### Conclusion

Both subsets clearly outperform the mixed full taxonomy, confirming that the
organ/action mix was diluting the learnable signal. Between the two, **action-based
classification is more learnable from molecular structure than organ-based
classification**: a drug's *mechanism of action* (anti-infective, analgesic, hormonal,
etc.) is far more tied to its chemical structure than *which organ system* it's
routed to, which depends on pharmacokinetics/physiology that structure alone doesn't
fully determine. In short: restructuring the label taxonomy to match what the input
features can actually predict was a bigger lever than further architecture/loss tuning.

## Tech stack

- **Kedro** — pipeline orchestration (`process_drug_data`, `build_model` /lipinski,
  `multitask_model`)
- **RDKit** — molecular featurization (fingerprints, descriptors)
- **TensorFlow / Keras** — Conv1D transfer-learning backbone + task-specific dense heads
