# NeurIPS paper workflow

This file is the operating manual for iterating on the NeurIPS paper. Keep it
up to date as conventions change.

---

## Where the paper lives

- **Source:** `neurips/paper/` — one file per section under `sections/`, plus
  `main.tex`, `references.bib`, `macros.tex`, and the bundled `neurips_2024.sty`.
- **Build artifact (Overleaf bundle):** `neurips/pem-neurips-overleaf.zip`.
  Gitignored — regenerate with the build script below.
- **Build artifact (local PDF):** `neurips/paper/build/main.pdf` if you run
  `latexmk` locally. Gitignored.

### Section files
| File | Content |
|---|---|
| `sections/00_abstract.tex` | Abstract environment |
| `sections/01_introduction.tex` | Introduction, motivation, contributions |
| `sections/02_related.tex` | Related work |
| `sections/03_method.tex` | Method (definition of $\rho_f$, PEM loss, implementation) |
| `sections/04_theory.tex` | Why PEM works ($p(1{-}p)$ argument, cluster assumption) |
| `sections/05_experiments.tex` | Experiments + main results table (placeholder) |
| `sections/06_ablations.tex` | Ablations (empty stub) |
| `sections/07_discussion.tex` | Failure modes, limitations, future work |
| `sections/99_appendix.tex` | Appendix (empty stub) |

`main.tex` `\input`'s all section files in order. To reorder or split a
section, edit `main.tex`.

---

## Editing the paper

1. Edit any file under `neurips/paper/sections/` in your IDE.
2. Iterate locally if you have `latexmk`:
   ```
   cd neurips/paper && latexmk -pdf main.tex
   ```
3. Or push to Overleaf and compile there (see next section).

### Changing the submission mode
`main.tex` has three options commented in the preamble:
```latex
\usepackage{neurips_2024}              % review (anonymized)  — default
%\usepackage[final]{neurips_2024}      % camera-ready / author names revealed
%\usepackage[preprint]{neurips_2024}   % preprint / arXiv (no page-limit banner)
```
Swap which one is uncommented to switch modes.

### Upgrading to a new NeurIPS style year
When NeurIPS releases `neurips_2025.sty` (or whichever year applies):
```
cd /tmp && curl -LO https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip
unzip -j Styles.zip "Styles/neurips_2025.sty" -d /home/tals/Documents/PostHocEM/neurips/paper/
```
Then edit `main.tex` to `\usepackage{neurips_2025}`. The options (`final`,
`preprint`, `nonatbib`) are stable across years.

---

## Uploading to Overleaf

### First-time upload (no Overleaf Premium needed)
```
bash neurips/scripts/build_overleaf_zip.sh
```
Output: `neurips/pem-neurips-overleaf.zip`. Upload at https://overleaf.com via
**New Project → Upload Project**.

### Updating an existing Overleaf project — manual
On every iteration:
```
bash neurips/scripts/build_overleaf_zip.sh
```
Then on Overleaf: delete the project (or drag the new zip into the file tree
to overwrite) and re-upload. Fastest cadence is ~30 seconds per iteration.

### Updating an existing Overleaf project — git push (Overleaf Premium)
Every Overleaf project has a git remote at `https://git.overleaf.com/<project_id>`.
One-time setup:
```
cd neurips/paper
git init                                         # only if not already inside a git repo
git remote add overleaf https://git.overleaf.com/<project_id>
git push overleaf master
```
Subsequent pushes: `git push overleaf master` from `neurips/paper/`.

*Caveat:* because `neurips/paper/` is a subdirectory of the main repo, you
can't `git init` inside it without nesting. The clean pattern for Overleaf
git sync is to use `git subtree split` or a shallow worktree:
```
git subtree push --prefix=neurips/paper overleaf master
```

### GitHub integration (Overleaf Premium, zero-maintenance)
In Overleaf project settings → GitHub → connect the repo
`NatalieCarlebach1/PostHocEM`, branch `feature/neurips-bootstrap`,
subdirectory `neurips/paper/`. Overleaf will pull on demand; we push to
GitHub as normal.

---

## Commit cadence

**One commit per logical iteration, pushed immediately.** We do not batch
"several small tweaks" into one commit — each distinct unit of work
(expanding a section, fixing a compile error, restructuring an argument,
adding a table, adding a figure) gets its own commit with a precise message,
and is pushed to `origin` as soon as it lands. This keeps the branch a
faithful record of the iteration history for the final paper-history
appendix and makes it trivial to revert a bad direction.

### Commit message format
```
NeurIPS: <short imperative summary of the change>

<1–3 sentence explanation of WHY, not just WHAT. The diff already
shows WHAT; the message should capture the intent, any tradeoff
that was weighed, and the expected downstream consequence.>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

### Typical iteration cycle
1. Edit a section file (or several).
2. Rebuild the Overleaf zip: `bash neurips/scripts/build_overleaf_zip.sh`
3. Test the compile on Overleaf (or `latexmk` locally).
4. `git add` the modified section files (not the zip — it's gitignored).
5. `git commit -m "NeurIPS: <summary>"` with the format above.
6. `git push origin feature/neurips-bootstrap`.
7. Repeat.

---

## Branch policy

- `master` — stable, nothing paper-related committed here yet.
- `feature/midl-3page-shrink` — **frozen** MIDL 3-page submission. Do not
  modify; branch from this if the MIDL paper needs a post-submission patch.
- `feature/neurips-bootstrap` — **active** NeurIPS iteration branch. All
  section writing, theory, experiments, and ablation work lands here.
- Future feature branches (e.g. `feature/neurips-experiments-most`,
  `feature/neurips-rebuttal`) branch from `feature/neurips-bootstrap` for
  parallel work streams.

Merges back into `master` happen only when a version of the paper is
considered submittable.

---

## Quick reference

| Task | Command |
|---|---|
| Rebuild Overleaf zip | `bash neurips/scripts/build_overleaf_zip.sh` |
| Compile locally | `cd neurips/paper && latexmk -pdf main.tex` |
| Clean local build artifacts | `cd neurips/paper && latexmk -C` |
| Push current branch | `git push origin feature/neurips-bootstrap` |
| See commits on this branch | `git log --oneline master..HEAD` |
| Switch to MIDL branch (read-only) | `git checkout feature/midl-3page-shrink` |
