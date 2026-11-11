# Labeling guide

You are judging **claims**, one row at a time, against the **retrieved sources only**. Your own
knowledge of the world does not count. The label column takes one letter.

| Label | Meaning |
|---|---|
| **S** supported | The sources state the claim or directly entail all of it: every number, unit, condition and qualifier. |
| **U** unsupported | Anything else: not mentioned, only partly mentioned, or adds detail, a cause, a comparison or a generalisation the sources do not give. |
| **C** contradicted | The sources state something incompatible with the claim. |
| **X** not a claim | The row is not a checkable factual claim (a bad split, a fragment, a greeting). Leave the reason in `notes`. |

For the headline precision and recall, **U and C both count as "unsupported"**. Keeping them apart
only matters for error analysis.

## Rules that decide the hard cases

1. **True in the real world but absent from the sources is U.** The system's job is to answer from
   the sources. Do not look anything up.
2. **Partly supported is U.** "The warranty lasts 10 years and covers labour" when the sources only
   give the 10 years: U.
3. **Check every number and unit.** 98 kg vs 89 kg is C. 10 years vs "about a decade" is S.
   A rounded or converted figure is S only if it is exact (6,000 = 6000), not if it is approximate.
4. **Paraphrase is fine.** Different words with the same meaning is S.
5. **Reasonable direct inference is S**, one step at most. "Installation in a bedroom is not
   permitted" supports "You should not put it in a bedroom". Anything needing outside knowledge or
   two or more inference steps is U.
6. **Evidence can come from any retrieved chunk**, not only the cited one. A claim supported by an
   uncited chunk is **S**. (Wrong citations are measured separately by the pipeline.)
7. **Evidence can span chunks.** If two retrieved passages together entail the claim, it is S.
8. **Vague quantifiers** ("many", "most", "quickly") are U unless the sources use the same word.
9. **Hedged claims** ("it may be possible to...") are S only if the sources hedge the same way.
10. **Refusals** ("The sources do not say ...") are not claims and are removed by the pipeline before
    labeling. If one slips through, mark it X.

## Procedure (so the numbers are defensible)

1. Build the sheet: `python scripts/build_label_sheet.py --run runs/run1.jsonl --out labels/annotator_a.csv`.
   It shows the question, the answer, the claim, the cited evidence and the other retrieved evidence,
   and hides all verifier output. **Do not look at verifier output before labeling.**
2. Label with the corpus open next to you. Work answer by answer. Fill `label`; add `notes` for hard calls.
3. A **second annotator** labels at least 25 answers (about a quarter of the set), chosen with
   `scripts/build_label_sheet.py --double 25 --seed 7`. Report Cohen's kappa. Aim for kappa ≥ 0.7 on the
   supported-vs-not distinction; if it is lower, tighten the rules above and relabel.
4. **Adjudicate** disagreements together, then use the adjudicated labels as `labels/annotator_a.csv`
   for the evaluation, and keep the raw second-annotator file for the kappa.
5. Freeze the labels before you tune anything. Dev answers (30% by default) are for choosing
   thresholds; the rest are the test set and are not looked at until the end.
6. If you evaluate `regenerate`, also label the new claims:
   `python scripts/build_label_sheet.py --run runs/run1.jsonl --stage final --skip-labeled labels/annotator_a.csv --out labels/final_claims.csv`
   then pass both files to the evaluation: `--labels labels/annotator_a.csv labels/final_claims.csv`.
