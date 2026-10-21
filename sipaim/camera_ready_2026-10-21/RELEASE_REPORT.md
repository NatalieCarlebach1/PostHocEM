# SIPAIM 2026 Paper 71 — camera-ready release report (2026-10-21)

**Final PDF:** `SIPAIM2026_paper71_camera_ready.pdf`
**SHA-256:** `b48c0ad529bb6192097056288daf53175762aec5470b52b6944d11298a32b6c3`

Nothing has been uploaded to CMT, and the copyright form has not been submitted.

## Sources

- **Authoritative source:** `sipaim/main.tex` + `references.bib` + the 18 Fig. 1 panels in `sipaim/figures/panels/`, at commit `0d86fa6`, the revision that passed the final review.
- **Reference PDF:** the reviewed build of that commit (2026-10-18). A clean rebuild of the source matched it exactly: identical extracted text, and all five pages pixel-identical.
- **Not edited:**
  - the originally submitted version (`f88a5b2` in git history)
  - the cover letters (`sipaim/cover_letter.{tex,md,pdf}`); these are not part of the camera-ready PDF
  - `sipaim/pg-5.png` (an untracked screenshot of the submitted page 5)
- `source/` holds exactly what produced the final PDF. Building it alone in an empty directory (pdflatex → bibtex → pdflatex ×2) regenerates an identical `main.bbl`, identical text, and all five pages pixel-identical to the final PDF.

## Metadata

| | |
|---|---|
| Title (PDF page 1, PDF metadata, proposed CMT title) | Post-hoc Entropy Minimization Improves Semi-Supervised Medical Image Segmentation |
| Authors, in order | Tal Shaharabany; Natalie Carlebach; Lior Wolf |
| Tal Shaharabany | Blavatnik School of Computer Science, Tel Aviv University, Tel Aviv, Israel — shaharabany@mail.tau.ac.il |
| Natalie Carlebach | Tel Aviv University, Tel Aviv, Israel — nataliecarlebach@gmail.com |
| Lior Wolf | Blavatnik School of Computer Science, Tel Aviv University, Tel Aviv, Israel — wolf@cs.tau.ac.il |
| Author block vs. submitted version | identical to `f88a5b2` |

## Checks

- **Build:** a clean build in an empty directory (pdflatex → bibtex → pdflatex ×2) gave:
  - 0 errors, 0 LaTeX warnings, 0 undefined references or citations
  - 0 overfull boxes, 0 underfull boxes, 0 BibTeX warnings
- **Pages:** 5 (US Letter).
  - The content, including the Conclusion, ends on page 4.
  - Page 5 begins with REFERENCES and contains only references.
  - No page numbers, headers or footers.
- **Fonts (`pdffonts`):** 27 fonts, all Type 1, embedded and subset; 0 Type 3.
- **PDF (`pdfinfo`, Ghostscript):**
  - PDF 1.7, not encrypted, opens and renders without errors.
  - No forms or JavaScript; no comments, highlights or markup.
- **Text (`pdftotext`):**
  - the text is searchable (about 4.2k words)
  - 0 matches for TODO, FIXME, XXX, TBD, placeholder, `??` or "undefined"
- **Visual:** every page was rendered and inspected. Fig. 1 was also checked at 300 dpi: column headers, row labels and per-case Dice overlays are legible at print size.
- **Against the reviewed PDF:**
  - Text identical on every page.
  - All five pages pixel-identical; the only change is PDF metadata, which is not rendered.
- **Numbers:** every value was checked against the run outputs:
  - Table II: all 9 method × setting rows (Dice, Jaccard, HD95, ASD) against the 3-seed final-epoch metrics
  - Table III: all 30 cells against the selection grid (`result/pem_selection_*.csv`)
  - Table IV: all 54 values against `result/seed_averaged_tests.json`
  - the abstract and results deltas (+1.02/+0.66/+0.62 Dice; 24.8/44.0/30.5% HD95; "25 to 44%"), computed from unrounded means
  - the seed SDs (0.06/0.19/0.07) and the supervised control (17.29 ± 5.47, at τ_m = 0.90, 5×10⁻⁵, E = 2, three seeds); ± is the sample SD (n = 3) throughout
- **Claims:**
  - The abstract, introduction, results and conclusion make the same claim: PEM improves all metrics on the three evaluated BCP checkpoints under dataset-specific settings, with a boundary advantage on Pancreas-CT in exploratory retrospective tests, and no significant difference from pseudo-label fine-tuning detected on Left Atrium.
  - Regression checks pass: 0.12 and 0.08 (not "0.08 at both rates"); τ_m = 0.90 has the higher mean over the six cells, while τ_m = 0.95 has only the largest single cell; "corrected retrospective reanalysis"; E = 2/5 carried over from the submitted protocol; the labeled volumes are excluded from every post-hoc objective but used for selection; the rank check is retrospective; no "all three methods indistinguishable" claim; test descriptions match the seed-averaged aggregation.

## Changes from the reviewed revision

1. **PDF metadata:** added `\hypersetup` with the title, the authors, "SIPAIM 2026" as subject, and the IEEE keywords. Previously the title and author fields were empty.

No other change. Nothing in the rendered pages changed; pagination is unchanged.

## Remaining items (not changed; for the authors to decide)

- **IEEE copyright notice.** IEEE camera-ready papers usually carry a copyright line at the bottom of page 1 (e.g. `979-8-…/26/$31.00 ©2026 IEEE`). The real code comes from the organizers or the IEEE copyright form, so none was invented. `\IEEEoverridecommandlockouts` is already in the preamble. If the line is required, re-check the 4-page fit after adding it.
- **IEEE PDF eXpress.** If SIPAIM requires it, it needs an upload to IEEE's site, so it was not run.
- **Minor legibility.** Fig. 1's correction-column marks are thin, and Tables II–IV are set at 7 pt. Both are legible at print size and both passed review.
- **Hash is for this exact file.** Rebuilding embeds new timestamps and changes the hash; upload the file in this folder.
- **nnU-Net run.** The Dataset520 run stopped on its own at epoch 151/250, and `checkpoint_latest.pth` is intact; resume it with `--c`. It is not part of this release.
