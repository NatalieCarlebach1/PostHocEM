# Cover Letter — Paper 71, SIPAIM 2026

**Post-hoc Entropy Minimization Improves Semi-Supervised Medical Image Segmentation**
Tal Shaharabany, Natalie Carlebach, Lior Wolf

*(Typeset version: `cover_letter.pdf`, from `cover_letter.tex`. This file mirrors it.)*

Dear SIPAIM 2026 Organizing Committee,

Thank you for the conditional acceptance, and for the additional time that allowed the experiments below to be completed. Our thanks also to Reviewer #4, whose two questions were narrowly aimed and, on investigation, both correct. Answering them required re-running the experimental protocol rather than rewording the paper. The revision changes several reported numbers, withdraws two claims, and narrows the comparison with the baselines. Because some choices from the submitted version are retained, we describe the revised evaluation as a *corrected retrospective reanalysis*, and say so in the paper.

## Reviewer #4, Comment 1 — cross-validation on four labeled volumes

> *Section V reports that the learning rates were selected by 5-fold cross-validation on the labeled folds but the LA 5% setting has 4 labeled volumes. How was 5-fold cross-validation performed on 4 labeled volumes?*

It was not. The reviewer is right that the arithmetic is impossible, and on checking our experiment records we found that no cross-validation was run in any setting. The learning rates and the confidence threshold were selected by test-set Dice, and the epoch budgets were set with test-set learning curves in view. The sentence was wrong, and it contradicted the abstract's claim of "no test-set tuning." We apologize for it.

**The replacement rule.** PEM optimizes an entropy objective on the *unlabeled* split alone, so the labeled training volumes (12 for Pancreas-CT, 4 and 8 for LA) are never seen by any post-hoc objective. They are used to select the learning rate and the mask threshold. They are held out from the post-hoc step but *not* from base training, since the BCP checkpoints were fitted on them, so we do not call them an independent validation set. Cross-validation is neither needed nor definable here, because no post-hoc method trains on labels.

**Two safeguards.** (1) Selection maximizes the Dice *change* these volumes report, never their absolute Dice, which is optimistic (92.68 against 87.32 on LA 5%). (2) We checked, retrospectively, that this change is informative. Over 33 configurations (the 30 of Table III and three learning-rate-ceiling runs), validation ΔDice ranks configurations as test ΔDice does, with Spearman ρ = 0.78 (Pancreas), 0.75 (LA 5%) and 0.99 (LA 10%), and 0.67 to 0.96 after removing every configuration that collapses outright, so the agreement is not merely collapse detection. This check necessarily uses test data and is labeled retrospective in the paper.

**Applied to every method.** PL-FT and SC go through the identical rule over the identical learning-rate grid, and all three methods report means over the same three seeds. Previously the baselines used hand-set rates and a single seed while PEM reported a three-seed mean.

**What remains.** The epoch budgets E=2 (Pancreas) and E=5 (LA) are retained from the submitted version rather than re-selected, and the bounds of the learning-rate grid were informed by the earlier sweeps. Learning rate and threshold are no longer chosen on test data, but the evaluation is not a pristine held-out one, and the paper states this in Sec. V and among its limitations.

## Reviewer #4, Comment 2 — oracle early stop, and τ=0.95 versus τ=0.99

> *Table III is reported as the peak Dice over the fine-tuning budget, that is, an oracle early stop, while Table II keeps a budget fixed a priori. The recommendation of τ=0.95 rests entirely on the oracle table. Under the fixed-budget protocol, rather than the oracle peak, does τ=0.95 still beat τ=0.99?*

The reviewer identified the inconsistency correctly, and the answer is **no**. Table III is now read at each dataset's fixed endpoint (E=5 LA, E=2 Pancreas-CT), with early stopping disabled. There, τ=0.95 beats τ=0.99 in three of six cells and loses in three, by margins as small as 0.03 Dice. The τ=0.95 recommendation is withdrawn.

The corrected table supports a different recommendation. Counting the cells, of six, in which each mask degrades a BCP checkpoint:

| Mask | Degraded cells |
|------|----------------|
| Full (no mask) | 3 of 6 |
| **τ_m = 0.90** | **0 of 6** |
| τ_m = 0.95 | 1 of 6 |
| τ_m = 0.99 | 2 of 6 |
| τ_m = 0.999 | 3 of 6 |

τ_m = 0.90 is the only threshold that never degrades a BCP checkpoint, and it also has the highest mean over the six cells (+0.72 against +0.46 for τ_m = 0.95, which has only the largest single cell). Fragility rises monotonically as the mask loosens toward the unmasked loss, with failures up to 30 Dice points, as the normalization argument of Sec. IV predicts, and the selection rule independently picks τ_m = 0.90 in all three settings. We also separated the notation: τ now denotes only the diagnostic threshold in ρ_f(τ), and τ_m the loss mask.

## Consequential changes to the results

**Main results.** With hyperparameters selected as above, Table II changes:

| Setting | Submitted | Revised |
|---------|-----------|---------|
| Pancreas-CT 20% | 84.01 (+1.12) | **83.91 (+1.02)** |
| LA 5% | 88.59 (+1.27) | **87.99 (+0.66)** |
| LA 10% | 90.02 (+0.62) | **90.02 (+0.62)** |

The submitted Pancreas-CT result also used the *unmasked* loss rather than the confident-shell loss the paper described; all three settings now use one configuration.

**Statistics.** A new Table IV reports per-case tests on the models Table II reports. Each case's metric is averaged over the three seeds before testing, the tests are two-sided paired Wilcoxon, HD95 is the primary family and Dice secondary, and Holm correction is applied across the three settings within each family; raw and adjusted p-values are both given.

- *Pancreas-CT:* PEM's HD95 improvement is significant against BCP (p_H < .001, better on 17 of 18 cases) and against both PL-FT and SC (p_H = .003); Dice improves over BCP more weakly (p_H = .042).
- *Left Atrium:* HD95 gains over BCP are nominal only (p = .030 and .033; p_H = .059); Dice improves significantly over BCP on LA 10% (p_H = .041) but not LA 5% (p_H = .090); and PEM is not separable from PL-FT on either metric. The mean HD95 advantage over PL-FT on LA 5% is carried by a few cases (PEM is better on 9 of 20, p = .70) and is not claimed.

The paper therefore no longer says the comparison "separates by metric": it separates by dataset, with a clear boundary-metric advantage on Pancreas-CT and nominal gains on Left Atrium.

**Withdrawn claims.** (1) *"Every gain significant at p < 0.025"* is false and removed. (2) *"PEM obtains the largest mean gain on the two less-saturated configurations"* is removed; PEM leads the other post-hoc methods reliably on Pancreas-CT only.

**Supervised negative control.** The submitted control, in which PEM degraded a supervised-only checkpoint (ρ_f = 0.55), had been run with the unmasked loss. Since the revision argues that the mask is insurance against collapse, we re-ran it at the selected configuration over three seeds: Dice falls from 75.66 to 17.29 ± 4.47. The configuration that improves all three BCP checkpoints destroys the one below the suitability threshold. The diagnostic claim is scoped accordingly: under the selected protocol, ρ_f separates these four checkpoints, and we do not claim more than four checkpoints can support.

**Other corrections.** The Pancreas-CT checkpoint, described in the submission as publicly released, is our reproduction of BCP with the official recipe (82.89 Dice against 82.91 reported), saved at its best-test-Dice evaluation; the paper now says so and notes that its base number is therefore optimistic, although every post-hoc method starts from it. The submission also stated that decoder-only updates give no gain; with the unmasked loss they gain +0.65 Dice against +1.12 for full fine-tuning, which still motivates updating all parameters. Statements generalizing from the BCP checkpoints to SSL methods at large have been removed, and the Table III grid is identified as single-seed.

**Figure 1** was regenerated from the reported checkpoints. Its correction maps are not uniformly favourable (on LA 10% PEM breaks part of the boundary rim even in a case it improves overall), and the text describes them as they are.

## Summary of revisions

1. Sec. V: the cross-validation claim replaced by the labeled-volume selection rule, its retrospective rank check, and an explicit statement of what was retained from the submission.
2. Table III: each dataset's fixed endpoint, extended to Pancreas-CT, with a degraded-cells column; τ=0.95 withdrawn in favour of τ_m=0.90.
3. Table II: all methods selected by the identical rule, three seeds each.
4. Table IV (new): seed-averaged per-case Wilcoxon tests, HD95 primary, Holm-corrected, raw and adjusted p.
5. Supervised control re-run at the selected configuration; the diagnostic claim scoped to the four checkpoints evaluated.
6. Abstract, introduction and conclusion aligned with the corrected claims; two claims withdrawn.
7. Notation separated (τ versus τ_m); a Sec. IV overclaim corrected (on instance-normalized Pancreas the unmasked loss is better than τ_m=0.90 by 0.12 and 0.08 Dice at the two learning rates; the mask is insurance against collapse, not a source of gain).
8. Pancreas-CT checkpoint correctly described as a reproduction; decoder-only claim and run-time statement corrected.
9. Figure 1 regenerated from the reported checkpoints.
10. Length: 4 pages of content with references on page 5, as submitted.

About 95 additional training runs were required. The revision reports smaller gains than the submitted version and narrower claims, and we believe it is the stronger paper for it. We are grateful to Reviewer #4 for the questions that produced it.

Sincerely,
The authors
