# Cover Letter — Paper 71, SIPAIM 2026

**Title:** Post-hoc Entropy Minimization Improves Semi-Supervised Medical Image Segmentation
**Authors:** Tal Shaharabany, Natalie Carlebach, Lior Wolf

Dear SIPAIM 2026 Organizing Committee,

Thank you for the conditional acceptance, and our thanks to Reviewer #4, whose two
questions were narrowly aimed and, on investigation, both correct. Answering them
required re-running the experimental protocol rather than rewording the paper, and
the revision changes several reported numbers. We set out below what we found, what
we changed, and which claims we have withdrawn.

---

## Reviewer #4, Comment 1 — cross-validation on four labeled volumes

> *Section V reports that the learning rates were selected by 5-fold cross-validation
> on the labeled folds but the LA 5% setting has 4 labeled volumes. How was 5-fold
> cross-validation performed on 4 labeled volumes?*

It was not. The reviewer is right that the arithmetic is impossible, and on checking
our experiment records we found the problem is worse than a misstatement: no
cross-validation was run in any setting. The learning rates, the confidence threshold
and the fine-tuning epoch were all selected by test-set Dice. The sentence was wrong,
and it contradicted the abstract's claim of "no test-set tuning." We apologize for it.

Rather than restate what was actually done, we have replaced it with a protocol the
method genuinely admits, and re-run every experiment under that protocol.

**The protocol.** PEM optimizes an entropy objective on the *unlabeled* split alone.
The labeled training volumes — 12 for Pancreas-CT, 4 and 8 for LA — are therefore
never seen by any post-hoc objective, and constitute a legitimate held-out set for
selecting hyperparameters. No cross-validation is needed or even definable here,
because no fold is ever trained on. This is now stated in Sec. V under "Selection
protocol."

**Two safeguards**, since the labeled volumes were used to fit the *base* SSL
checkpoint and their absolute Dice is consequently optimistic (92.68 against 87.32 on
LA 5%):

1. Selection maximizes the Dice **change** these volumes report, never their absolute
   Dice.
2. We verified that this change is informative rather than assumed it. Across the 33
   configurations of the ablation grid, validation ΔDice ranks configurations as test
   ΔDice does, with Spearman ρ = 0.78 (Pancreas), 0.75 (LA 5%) and 0.99 (LA 10%).
   Because a rank correlation could be driven purely by separating collapsed runs from
   healthy ones, we recomputed it after removing every configuration that collapses
   outright: ρ remains 0.67 to 0.96. The signal is not merely collapse detection.

**Applied to every method.** PL-FT and SC are now put through the identical selection
rule over the identical learning-rate grid, and all three methods report means over the
same three seeds. Previously the baselines used hand-set rates and a single seed while
PEM reported a three-seed mean, which was an asymmetry in PEM's favour.

No number in the revised paper is selected using the test split.

---

## Reviewer #4, Comment 2 — oracle early stop, and τ=0.95 versus τ=0.99

> *Table III is reported as the peak Dice over the fine-tuning budget, that is, an
> oracle early stop, while Table II keeps a budget fixed a priori. The recommendation
> of τ=0.95 rests entirely on the oracle table. Under the fixed-budget protocol, rather
> than the oracle peak, does τ=0.95 still beat τ=0.99?*

The reviewer identified the inconsistency correctly, and the answer to the question is
**no**. Table III is now reported at the fixed *a priori* budget throughout, with early
stopping disabled so that every cell is read at the same epoch. Under that protocol
τ=0.95 beats τ=0.99 in three of six cells and loses in three, with margins of 0.03 to
0.06 Dice on Pancreas-CT. Neither threshold is stable, and we have withdrawn the
τ=0.95 recommendation.

The corrected table supports a better recommendation, on a stronger criterion than
"best mean." Counting the cells in which each mask degrades the checkpoint:

| Mask | degraded cells |
|------|----------------|
| Full (no mask) | 3 of 6 |
| **τ_m = 0.90** | **0 of 6** |
| τ_m = 0.95 | 1 of 6 |
| τ_m = 0.99 | 2 of 6 |
| τ_m = 0.999 | 3 of 6 |

τ_m = 0.90 is the only setting that never degrades a checkpoint, and fragility rises
monotonically as the mask loosens toward the unmasked loss — with failures as large as
30 Dice points. This is precisely what the normalization argument of Sec. IV predicts,
so the corrected table supports the paper's theory more directly than the oracle table
did. The revised paper recommends τ_m = 0.90 on that basis, and the labeled-validation
rule independently selects it in all three settings.

We also separated the notation: τ now denotes only the diagnostic threshold in ρ_f(τ),
and τ_m the loss mask. They take different values and the shared symbol was confusing.

---

## Consequential changes to the results

Because hyperparameters are now selected without test access, Table II changes. We
report the new numbers plainly:

| Setting | Submitted | Revised |
|---------|-----------|---------|
| Pancreas-CT 20% | 84.01 (+1.12) | **83.91 (+1.02)** |
| LA 5% | 88.59 (+1.27) | **87.99 (+0.67)** |
| LA 10% | 90.02 (+0.62) | **90.02 (+0.62)** |

The submitted Pancreas-CT result additionally used the *unmasked* loss, not the
confident-shell loss the paper described, which contradicted our own analysis section.
All three settings now use one uniform configuration.

**Two claims have been withdrawn as unsupported:**

1. *"Every gain significant at p < 0.025."* Under per-case Wilcoxon tests, PEM improves
   on BCP at p = 0.030 (Pancreas-CT) and p = 0.030 (LA 10%), but **p = 0.097 on LA 5%**.
   The claim was false and is removed.

2. *"PEM obtains the largest mean gain on the two less-saturated configurations."* Under
   the equalized protocol PEM leads on Dice on Pancreas-CT only. We note, however, that
   PL-FT's nominal Dice leads on LA are themselves not significant (p = 0.729 and
   p = 0.674), so the revised paper states that the three methods are statistically
   indistinguishable on LA rather than claiming an ordering in either direction.

**What the evidence does support**, and how the paper is now framed: the comparison
separates by metric rather than by dataset. PEM is the best method on both boundary
metrics wherever the checkpoint retains headroom — HD95 of 5.87 against 7.62 for the
nearest competitor on Pancreas-CT, and 7.79 against 10.11 on LA 5%, a 23% margin in
both — and on all four metrics on Pancreas-CT. This is what a boundary-refinement step
should produce, since HD95 and ASD measure surface agreement, where the entropy
gradient acts, while Dice is volumetric and dominated by the interior the method leaves
untouched. PEM's Dice margin over PL-FT tracks the base checkpoint's headroom (+0.73,
−0.25, −0.07 at base Dice 82.89, 87.32, 89.40), which is consequence (ii) of Sec. IV
measured rather than argued.

Figure 1 was regenerated from the checkpoints now reported. Its correction maps are not
uniformly favourable — on the saturated LA 10% setting, PEM breaks part of the boundary
rim even in a case it improves overall — and we describe them as they are, since that is
what ρ_f → 1 predicts.

---

## Summary of revisions

1. Sec. V: the cross-validation claim replaced by the labeled-validation protocol, with
   its rank-agreement validation.
2. Table III: fixed *a priori* budget, extended to Pancreas-CT, with a column counting
   degraded cells; τ=0.95 recommendation withdrawn in favour of τ_m=0.90.
3. Table II: all methods selected by the identical rule, three seeds each.
4. New paragraph reporting per-case paired tests, including the non-significant results.
5. Abstract, introduction and conclusion aligned with the corrected claims; two claims
   withdrawn.
6. Notation separated (τ versus τ_m); Sec. IV overclaim corrected (the masked loss does
   not beat the unmasked loss at every learning rate — on instance-normalized Pancreas
   the unmasked loss is better by 0.08 Dice; the mask is insurance against collapse).
7. Figure 1 regenerated from the reported checkpoints.
8. Length: 4 pages of content with references on page 5, as submitted.

Roughly 65 additional training runs were required. The revision reports smaller Dice
gains than the submitted version, and one fewer superiority claim. We think it is the
stronger paper, and we are grateful to Reviewer #4 for the questions that produced it.

Sincerely,
The authors
