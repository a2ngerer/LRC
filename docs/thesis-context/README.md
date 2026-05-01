# Thesis Context Snapshot

This folder is a **read-only snapshot** of thesis-planning documents that live
outside this repository, copied in so automated reviewers (e.g. `/ultrareview`)
and human readers can evaluate the code against the project's stated goals,
research questions, and implementation plan.

**Snapshot date:** 2026-05-01

## Source of truth

The authoritative versions live in the parent `master_thesis/` workspace
(not under version control here). Do **not** edit files in this folder
directly — edit the source and re-copy if a refresh is needed.

| File here | Original location | Purpose |
|-----------|-------------------|---------|
| `proposal-main.tex` | `proposal/Thesis_Proposal__Alexander_Angerer/main.tex` | Thesis proposal v3 — research questions, scope, hypotheses, methodology |
| `project-status.md` | `obsidian_master_thesis/Meta/claude-project-status.md` | Current phase, what is done, what is next |
| `project-plan.md` | `obsidian_master_thesis/Meta/masterarbeit-projektplan.md` | Work breakdown structure, critical path, phase plan |
| `benchmark-repo-roadmap.md` | `obsidian_master_thesis/Thesis/benchmark-repo-roadmap.md` | Code implementation plan — 4×2 matrix, planned repo restructure |
| `lrc-repo-overview.md` | `obsidian_master_thesis/Thesis/lrc-repo-overview.md` | Description of the LRC fork this repo started from |
| `work-documentation.md` | `obsidian_master_thesis/Thesis/work-documentation.md` | Decision log, work history |

## What to use this for

- Verify whether the code in `src/`, `classification/`, `neuralODE/` etc.
  matches the architecture matrix and roadmap described here.
- Detect drift between the implementation plan
  (`benchmark-repo-roadmap.md`) and the actual repo layout.
- Identify missing components from the planned 4×2 matrix
  (LSTM · CT-RNN · STC · LRC × Dense · NCP) plus intermediate models
  (CT-RNN + ε(w_i)) and the gradient-flow analysis (RQ4).

## Caveats for readers

- The Markdown files originate from an Obsidian vault and contain
  `[[wikilink]]` references to notes that are **not** included here.
  Treat unresolved wikilinks as pointers to external context, not as
  broken links.
- `proposal-main.tex` references `bibliography.bib` which is not copied.
- Mixed German / English: planning docs are in German, the proposal
  and the code are in English.

## Refresh procedure

When the planning docs change materially and a refresh is wanted, re-run
the copy from the workspace root:

```bash
cp ../proposal/Thesis_Proposal__Alexander_Angerer/main.tex \
   docs/thesis-context/proposal-main.tex
cp ../obsidian_master_thesis/Meta/claude-project-status.md \
   docs/thesis-context/project-status.md
cp ../obsidian_master_thesis/Meta/masterarbeit-projektplan.md \
   docs/thesis-context/project-plan.md
cp ../obsidian_master_thesis/Thesis/benchmark-repo-roadmap.md \
   docs/thesis-context/benchmark-repo-roadmap.md
cp ../obsidian_master_thesis/Thesis/lrc-repo-overview.md \
   docs/thesis-context/lrc-repo-overview.md
cp ../obsidian_master_thesis/Thesis/work-documentation.md \
   docs/thesis-context/work-documentation.md
```

Update the snapshot date above and append a line to the change log below.

## Change log

- **2026-05-01** — Initial snapshot of six documents (proposal, status,
  plan, roadmap, repo overview, work log). Created to give `/ultrareview`
  visibility into thesis goals and implementation plan during the cloud
  code review.
