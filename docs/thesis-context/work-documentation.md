# Work Documentation

Implementation log for the thesis benchmark codebase.
Repository: `master_thesis/code/` (GitHub: `a2ngerer/LRC`)
Workflow guide: `[[benchmark-repo-roadmap]]`

**Format**: one entry per work session, newest first.

---

## Entry Template

```
### YYYY-MM-DD
**Branch**: `phase{N}/step{M}-name`
**Status**: In progress | Merged to main

**What was done**:
- 

**Files changed**:
- 

**Notes / Issues**:
- 
```

---

## Log

### [2026-04-26] research | Deep Research v1 — drei Reports synthetisiert, Vault-Diff-Liste erstellt
**Branch**: N/A (reine Vault-Arbeit, keine Code-Änderungen)
**Status**: Done — wartet auf User-Review der Diff-Liste

**What was done**:
- Drei Deep-Research-Reports aus `deep_research/{chatgpt,gemini,claude}/` vollständig gelesen und nach Sections kreuzverglichen (NCP-Training, Empirie, LRC-Status, Liquid AI, L1–L6, Libraries).
- Aktuellen Vault-Stand systematisch abgeglichen: Concepts (LRC, LTC, NCP, EEC, Synapsen, REPPO, Neural-ODE), Papers (Farsang 2024 LRC + Chemical, Hasani 2020 LTC, Lechner 2020 NCP, Voelcker 2025), Proposal (scope-decisions, structure), Thesis (benchmark-repo-roadmap), `Literature/references.bib`, `Ideas/ideas.md`.
- Synthese-Dokument `Meta/Deep Research v1 compacted.md` angelegt mit:
  - Drei-Satz-Executive-Summary, dann Section-für-Section-Synthese mit konkreten Zahlen (Drone OOD: LSTM 27.5 % vs CfC 67.5 % Urban-Patio; LRC vs LSTM/GRU auf Localization/IMDB/psMNIST; Crash-Likelihood Lane Keeping; Walltime/Epoch).
  - Quellen-Tabelle mit 22 fehlenden BibLaTeX-Einträgen (mit ✓/⚠/✗-Klassifikation gegen Hallucination-Risiko — Gemini's BioNIC arXiv:2601.20876 explizit als wahrscheinlich erfunden geflaggt).
  - Vault-Diff-Liste mit 30 nummerierten Punkten: 11 neue Notizen (N1–N11), 19 Korrekturen/Ergänzungen an bestehenden Notizen (K1–K19) — jeder Punkt mit Quelle, Begründung, vorgeschlagener Aktion.
  - Konsens vs Diskrepanzen zwischen den drei Reports explizit aufgeführt (Solver-Wahl, Drone-Reproduzierbarkeit, L1–L6-Realismus, CfC-Status).
  - Empfohlener Roadmap-Update mit 3 Hebeln nach Impact-vs-Effort.
  - Top-5-Reading-List (Konsens), eine Contrarian-Take aus Claude-Report.
- **Wichtigste neue Befunde:**
  - **LrcSSM (Farsang et al. 2025, arXiv:2505.21717, NeurIPS 2025)** ist die direkte formale Brücke für RQ4: diagonale Jacobi-Strukturen sind nachweislich gradient-stabil, NCP-Wiring verletzt diese Bedingung am Command-Layer.
  - **ODE-LSTM (Lechner & Hasani 2020, arXiv:2006.04418)** beweist EVGP für *jedes* ODE-RNN unabhängig vom Solver — die formale Wurzel von RQ4, fehlt aktuell im Vault.
  - **Liquid AI's LFM2 ist hybrid** (Conv + Grouped-Query Attention), nicht pure liquid; LFM-40B liegt unabhängig benchmarkt unter dem Median seiner Preisklasse — Korrekturfolie gegen Marketing in Discussion.
  - **Ickin et al. 2025 (Telekom-Studie, arXiv:2504.02781)** ist die wichtigste *unabhängige* NCP-Studie und zeigt NCP-Effizienz ohne Accuracy-Überlegenheit.
  - **`ncps`-Bibliothek** unterstützt PyTorch + TensorFlow + Keras 3 (nicht nur TF wie aktuelle Concept-Notiz suggeriert); STC und LRC sind *nicht* in `ncps` enthalten — rechtfertigt Eigenimplementierung.

**Files changed**:
- `obsidian_master_thesis/Meta/Deep Research v1 compacted.md` (neu — ~10k tokens, Synthese + Diff-Liste zur User-Review)
- `obsidian_master_thesis/Thesis/work-documentation.md` (dieser Eintrag)

**Notes / Issues**:
- **Bewusst keine bestehenden Notizen verändert** — User entscheidet pro Diff-Eintrag (✅/❌/⏸) bevor Claude die Änderungen ausführt.
- **Hallucination-Warnungen explizit:** Gemini erfindet wahrscheinlich BioNIC arXiv-ID (2601.20876 entspricht keinem echten arXiv-Schema); Whittington 2025 Neuron 113 Seitenangabe vor Übernahme prüfen; Liquid-AI-Press-Releases als Marketing-Quellen behandeln, nicht als peer-reviewed Evidenz.
- Reports stammen alle aus 2026-04-26 (Tag der Erstellung); Prompt unter `deep_research/_prompts/state-of-the-art-research-prompt.md`.
- Nächster Schritt: Alexander reviewt `Deep Research v1 compacted.md` und markiert pro Diff-Eintrag die gewünschte Aktion.

---

### [2026-04-19] meta | Thesis LaTeX-Skelett restrukturiert (Brainstorm → Spec → Plan → Execute)
**Branch**: N/A (root-level `Thesis/` folder)
**Status**: Done

**What was done**:
- **Brainstorm** mit superpowers-Skill: Kapitelstruktur der Thesis durchdiskutiert, Referenz auf ARC-Template (TU-Wien-Titelblatt mit Logo) einbezogen.
- **Design-Spec** erstellt: `docs/superpowers/specs/2026-04-19-thesis-chapter-structure-design.md` — 9-Kapitel-Hybrid-Struktur, erweitertes Frontmatter (TU-Wien-Style), 4-teiliger Appendix. Nach spec-review überarbeitet (Kapitel-6-Benchmark-Granularität, Kapitel-5-Erfolgskriterium auf Ch. 2/4/7 verbreitert, Logo-Copy explizit in Scope).
- **Implementation-Plan** erstellt: `docs/superpowers/plans/2026-04-19-thesis-chapter-structure.md` — 25 Tasks mit exakten Schritten.
- **Ausführung (autonom, ohne Bestätigungsabfragen nach User-Instruktion):** Alle 25 Tasks batch-ausgeführt.
- **Neue Struktur:**
  - 6 Kapitel → 9 Kapitel: Introduction / Background / Related Work / Architectures & Models / Experimental Setup / Results: Benchmarks / Results: Mechanistic / Discussion / Conclusion
  - Frontmatter erweitert: TU-Wien-Titelpage mit Logo, Statutory Declaration (EN), Kurzfassung (DE Pflicht TU Wien), Abstract (EN), Acknowledgements, ToC/LoF/LoT, List of Abbreviations, List of Symbols
  - Appendix neu: A Extended Derivations / B Experimental Details / C Additional Results / D Reproducibility
- **Kapitelinhalte:** Jede Section/Subsection hat `% TODO`-Kommentar mit Quellenverweis (Proposal-Abschnitt, Concept-Note, Paper). Platzhaltertexte `\textit{[Placeholder]}` vorbereitet.
- **Kapitel 02 Background** enthält EEC-Familie explizit mit Subsections: CT-RNN, LTC, STC, LRC; RQ5-Intermediate-Modelle werden in Kapitel 04 separat behandelt.
- **Kapitel 04 Architectures & Models** enthält bedingte Section 4.7 für L1–L6 (Potential Extended Contribution).
- **Kapitel 06 Results: Benchmarks** mit Category-Level-Split (Time-Series vs. Control) × (Dense vs. NCP) für RQ1/RQ3.
- **Kapitel 07 Results: Mechanistic** bündelt RQ2/RQ4/RQ5 (Ablationen, Gradient Flow, Intermediate, Dynamik, Stabilität).
- **TU-Wien-Titelpage:** `TU_logo.png` aus ARC-Template (`~/Repositories/ARC/labs/lab1/.../img/TU_logo.png`) kopiert nach `Thesis/img/TU_logo.png`. Titelseite auf Englisch, TU-Wien-Master's-Thesis-Konvention mit Logo oben, Titel mittig, Supervisors/Autor/Matrikelnummer/Datum.
- **Alte Kapitel-Files gelöscht:** `04_methodology.tex`, `05_results.tex`, `06_conclusion.tex` (ersetzt durch neue/umbenannte).
- **Compile-Checkpoints:** nach Frontmatter 22 Seiten; nach Kapiteln 30 Seiten; **finaler Build 34 Seiten** (pdflatex + bibtex + pdflatex + pdflatex, keine Errors). Alle 9 Kapitel, 6 Frontmatter-Entries (gesternt) und 4 Appendix-Kapitel erscheinen korrekt im ToC.

**Files changed**:
- `Thesis/main.tex` — graphicspath um `img/` erweitert, 5 neue Frontmatter-Inputs, Kapitelliste auf 9, `\appendix` + 4 Appendix-Inputs
- `Thesis/img/TU_logo.png` (neu, aus ARC)
- `Thesis/frontmatter/titlepage.tex` — komplett neu (TU-Wien-Style)
- `Thesis/frontmatter/statutory_declaration.tex` (neu)
- `Thesis/frontmatter/kurzfassung.tex` (neu, DE)
- `Thesis/frontmatter/list_of_abbreviations.tex` (neu, tabular)
- `Thesis/frontmatter/list_of_symbols.tex` (neu, grouped tabular)
- `Thesis/chapters/01_introduction.tex` (restrukturiert: 5 Subsections)
- `Thesis/chapters/02_background.tex` (restrukturiert: 5 Sections, EEC-Subsections)
- `Thesis/chapters/03_related_work.tex` (restrukturiert: 6 Sections)
- `Thesis/chapters/04_architectures_and_models.tex` (neu, 8 Sections inkl. konditionale 4.7)
- `Thesis/chapters/05_experimental_setup.tex` (neu, 6 Sections)
- `Thesis/chapters/06_results_benchmarks.tex` (neu, 6 Sections)
- `Thesis/chapters/07_results_mechanistic.tex` (neu, 5 Sections)
- `Thesis/chapters/08_discussion.tex` (neu, 5 Sections)
- `Thesis/chapters/09_conclusion.tex` (neu, 3 Sections)
- `Thesis/chapters/{04_methodology,05_results,06_conclusion}.tex` (gelöscht)
- `Thesis/appendix/A_extended_derivations.tex` (neu)
- `Thesis/appendix/B_experimental_details.tex` (neu)
- `Thesis/appendix/C_additional_results.tex` (neu)
- `Thesis/appendix/D_reproducibility.tex` (neu)
- `Thesis/README.md` (Struktur-Tree aktualisiert)
- `docs/superpowers/specs/2026-04-19-thesis-chapter-structure-design.md` (neu)
- `docs/superpowers/plans/2026-04-19-thesis-chapter-structure.md` (neu)
- `obsidian_master_thesis/Thesis/latex-template.md` (aktualisiert)
- `obsidian_master_thesis/Meta/claude-project-status.md` (aktualisiert)

**Notes / Issues**:
- Keine Obsidian-MCP-Overwrite-Incidents — alle 3 Vault-Files vor Überschreiben vollständig via `obsidian_read_note` eingelesen.
- Repo ist nicht unter Git-Versionskontrolle → keine Commits pro Task; Work-Log-Eintrag ersetzt Commit-Historie.
- Offene Arbeit: Matrikelnummer in titlepage, Inhalte aus Proposal in Kapitel 01/02 übernehmen, Phase-3-Ergebnisse in Kapitel 06/07 + Appendix B/C einfüllen.

---

### [2026-04-19] meta | Thesis LaTeX-Skelett angelegt (AP 2.1)
**Branch**: N/A (root-level `Thesis/` folder, nicht in code-Repo)
**Status**: Done

**What was done**:
- Neuer Ordner `Thesis/` auf Repo-Root erstellt (neben `proposal/`, `code/`).
- LaTeX-Stil aus `proposal/Thesis_Proposal__Alexander_Angerer/main.tex` abgeleitet: identische Packages (`setspace`, `amsmath`, `hyperref`, `cite`), `setstretch{1.4}`, `bibliographystyle{alpha}`. Klasse `article` → `report` für `\chapter`.
- Gliederung nach Proposal "Structure of the Work" (6 Kapitel): Introduction, Background, Related Work, Methodology, Results, Conclusion.
- Jedes Chapter-File mit `% Source material:`-Kommentar (verweist auf Proposal-Quellabschnitt), `\textit{[Placeholder]}` + `% TODO`-Kommentaren für spätere Befüllung.
- Frontmatter: `titlepage.tex` (Proposal-Stil-Placeholder, TU-Wien-Vorlage TBD), `abstract.tex`, `acknowledgements.tex`.
- `bibliography.bib` = Kopie aus Proposal-bib (wird später gegen `Literature/references.bib` synchronisiert).
- Build verifiziert: `pdflatex main.tex` erzeugt 13-seitiges Placeholder-PDF ohne Errors.

**Files changed**:
- `Thesis/main.tex` (neu)
- `Thesis/chapters/{01_introduction,02_background,03_related_work,04_methodology,05_results,06_conclusion}.tex` (neu)
- `Thesis/frontmatter/{titlepage,abstract,acknowledgements}.tex` (neu)
- `Thesis/bibliography.bib` (neu, Kopie)
- `Thesis/README.md` (neu)
- `Thesis/{figures,tables}/.gitkeep` (neu)
- `CLAUDE.md` — Schlüsselpfade + Architektur-Tabelle um `Thesis/`-Layer ergänzt
- `claude_instructions/latex-workflow.md` — Thesis-Skelett-Abschnitt ergänzt
- `claude_instructions/karpathy-pattern.md` — Thesis-Layer + 2 neue Sync-Paare
- `Meta/claude-project-status.md` — Thesis-Skelett unter "Was ist fertig", Link zu [[latex-template]]
- `Meta/masterarbeit-projektplan.md` — AP 2.1 Aufgaben abgehakt, Status-Zeile + Link zu [[latex-template]]
- `Thesis/latex-template.md` (neu im Vault) — Ausbaustand-Tracker

**Notes / Issues**:
- **Incident:** Bei Obsidian MCP `obsidian_update_note overwrite` wurde `masterarbeit-projektplan.md` versehentlich mit leerem Inhalt überschrieben. Wiederhergestellt via Time Machine (Local-Snapshot 05:59:21). Lehre: Nie `wholeFile/overwrite` für bestehende Dateien ohne prior Read + bewussten Vollinhalt — für kleine Änderungen immer `Edit`-Tool.
- TU-Wien-Titelseite offen (Monika oder TISS-Vorlage anfragen).

---

### [2026-04-19] proposal | v3 — Monika-Feedback round 2 integrated
**Branch**: N/A (Proposal + Obsidian updates)
**Status**: Done

**What was done**:
- **Titel** erweitert: "A Comparative Study of Architectural Mechanisms" → "A Comparative Study of Architectural **and Model** Mechanisms" (Model-Axis explizit benannt)
- **RQ1** CT-RNN → GRU ersetzt: CT-RNN modelliert elektrische Synapsen → ist bio-inspired, nicht traditional. Traditional gated RNNs = LSTM, GRU.
- **Motivation §1** umformuliert: Trennt Traditional gated (LSTM/GRU) von EEC-Baseline (CT-RNN, elektrische Synapsen).
- **Aim / Experimental Matrix** neu kategorisiert: 3-zeilige Tabelle (Gated baseline LSTM / EEC baseline CT-RNN / Novel bio-inspired STC, LRC) × Dense/NCP.
- **Aim**: Paragraph "Motivating observation for the wiring axis" ergänzt — Monikas LRC+NCP-Gradient-Flow-Beobachtung explizit als Motivation für RQ4.
- **Potential Extended Contribution** neue Section: L1–L6 cortical-layer-inspired wiring als bedingte Erweiterung (nach Baseline-Completion). Setzt Monikas Email-Kommentar um.
- **Scope & Delimitation** von "explicitly out of scope" → "primary focus + out of scope" umgeschrieben; rechtfertigt warum LSTM/GRU drin und Transformer/S4/Mamba/xLSTM draußen (parameter scale / embedded CPS regime); stellt klar dass intermediate models *within scope* sind und L1–L6 oben als Potential Extended Contribution läuft.
- **Curriculum**: Kurs **191.119 Autonomous Racing Cars** ergänzt (F1TENTH, ROS 2, PID/Pure Pursuit, Mapping — robotics control grounding).
- **RQ-Intro**: "three research questions" → "five research questions".
- Proposal kompiliert sauber: 15 Seiten, nur harmlose typografische overfull/underfull hboxes.

**Files changed**:
- `proposal/Thesis_Proposal__Alexander_Angerer/main.tex`
- `obsidian_master_thesis/Proposal/structure.md` (Titel, Motivation, Aim, Curriculum)
- `obsidian_master_thesis/Proposal/scope-decisions.md` (Datum, Baseline-Framing, Titel, Extended Scope 2026-04-19)
- `obsidian_master_thesis/Meta/claude-project-status.md` (aktuelle Phase, Scope-Beschreibung, nächste Schritte)

**Notes**:
- GRU kommt nicht zusätzlich in die Experimental Matrix — wird nur als Familien-Vertreter in RQ1, Motivation und SotA genannt. Experimentell bleibt LSTM der einzige gated Baseline.
- Nächster Schritt: Proposal v3 PDF auf Overleaf teilen.

---

### [2026-04-17] lint | Claude-Setup & Vault-Cleanup
**Branch**: N/A (Infrastruktur + Vault-Pflege)
**Status**: Done

**What was done**:
- Claude-Setup auditiert, 11 Inkonsistenzen identifiziert; Karpathy-LLM-Wiki-Muster eingeführt (3-Schichten, log/index/schema)
- Projekt-lokale Slash-Commands: `/lint-wiki`, `/ingest-url`, `/process-inbox` unter `.claude/commands/`
- Projekt-lokale Sync-Hooks: `.claude/hooks/sync-check.sh` (PostToolUse auf Write/Edit) → warnt bei gepaarten Files automatisch
- CLAUDE.md neu strukturiert (Session-Start-Regel, 3-Schichten-Tabelle, Command-Liste)
- Workflow-Docs bereinigt: `paper-workflow.md` Citekey-Regel, `vault-structure.md` regeneriert, `ideas-workflow.md` Inbox-Pattern, neu `karpathy-pattern.md`
- `/lint-wiki` ausgeführt und Vault aufgeräumt:
  - Vault-bib von 5 → 35 Einträgen erweitert (Merge aus Proposal-bib, beide Casings koexistieren)
  - 7 Path-style Wikilinks zu Basename-Form umgeschrieben
  - 2 Paper-Stubs angelegt: `vaswani2017-attention`, `suzuki2025-grover`
  - 4 Concept-Stubs angelegt: `self-attention`, `neural-ode`, `policy-gradients`, `reinforcement-learning`
  - Indexes gepflegt: `Papers/index.md`, `Concepts/index.md`, `AI Tutoring/index.md`
- Obsidian `_inbox/` Ordner für fleeting notes eingerichtet

**Files changed**:
- `CLAUDE.md`, `claude_instructions/{paper,vault,ideas,karpathy}-*`, `MEMORY.md`
- `.claude/{settings,settings.local}.json`, `.claude/hooks/sync-check.sh`, `.claude/commands/*.md`
- `obsidian_master_thesis/Literature/references.bib` (+30 Einträge)
- `obsidian_master_thesis/{Papers,Concepts,AI Tutoring}/index.md`
- `obsidian_master_thesis/Papers/{vaswani2017-attention,suzuki2025-grover}.md` (neu)
- `obsidian_master_thesis/Concepts/{self-attention,neural-ode,policy-gradients,reinforcement-learning}.md` (neu)
- `obsidian_master_thesis/Meta/claude-project-status.md` (Link-Fixes)
- `obsidian_master_thesis/Papers/voelcker2025relative.md` (Link-Fixes)
- `obsidian_master_thesis/Thesis/thesis_corpus.md` (Link-Fixes)

**Notes**:
- Lint verifiziert: 0 broken links, 0 orphan citekeys
- Stubs (Paper + Concepts) müssen inhaltlich noch mit Substanz gefüllt werden — als "Stub" markiert
- Citekey-Groß-/Kleinschreibung (`Cho2014` vs. `cho2014…`) bewusst nicht normalisiert — würde `\cite{}` in `main.tex` brechen. Beide Varianten koexistieren in `references.bib`.

---

### 2026-03-29 (Proposal v2 — Supervisor Feedback)
**Branch**: N/A (Proposal + Obsidian updates)
**Status**: Done

**What was done**:
- Feedback von Monika Farsang erhalten und eingearbeitet
- Monika Farsang als `Assistance` in `\author{}` eingetragen
- RQ4 (Convergence & Gradient Flow: LRC+NCP Konvergenzproblem) hinzugefügt
- RQ5 (Intermediate Models: CT-RNN + liquid elastance als Ablation) hinzugefügt
- Contribution 4 (Intermediate Model Design) hinzugefügt
- Scope-Statement "No new neuron types or architectural variants are invented" entfernt
- STC als unblockiert markiert (kein Supervisor-Meeting mehr nötig)
- Obsidian-Pflege-Cheatsheet angelegt (`Meta/pflege-cheatsheet.md`)

**Files changed**:
- `proposal/Thesis_Proposal__Alexander_Angerer/main.tex`
- `obsidian_master_thesis/Proposal/scope-decisions.md`
- `obsidian_master_thesis/Meta/claude-project-status.md`
- `obsidian_master_thesis/Meta/masterarbeit-projektplan.md` (AP 3.2 + 3.3 erweitert)
- `obsidian_master_thesis/Thesis/benchmark-repo-roadmap.md` (Extended Scope, Intermediate Models, Gradient Flow)
- `obsidian_master_thesis/Meta/pflege-cheatsheet.md` (neu)

**Notes**:
- Nächster Schritt: Proposal auf Overleaf teilen (Monika bat darum)
- STC Cell kann jetzt implementiert werden (Phase 2)

---

### 2026-03-17 (Proposal Revision und Einreichung)
**Branch**: N/A
**Status**: Abgesendet — warte auf Feedback von Monika Farsang

**What was done**:
- Proposal ueberarbeitet: Kapitel 1 gekuerzt, 2x2-Matrix nach Kapitel 2 verschoben
- Faktenfehler behoben: Mamba/xLSTM Effizienz korrekt dargestellt; NCP Biologie korrigiert (C. elegans statt kortikale Schichten)
- Terminologie konsistent: liquid elastance ueberall
- 20 Seiten auf 12 Seiten gekuerzt
- Bibliographie bereinigt: Hasani2020 LTC-Paper korrekt; Severin2022 Cell Reports 2024; energyinference korrekte Autoren
- Proposal an Betreuerin Monika Farsang abgesendet

**Files changed**:
- proposal/Thesis_Proposal__Alexander_Angerer/main.tex
- proposal/Thesis_Proposal__Alexander_Angerer/bibliography.bib

---

### 2026-03-13 (Phase 3 / Step 10)
**Branch**: `phase3/step10-neural-ode-benchmark`
**Status**: Branch fertig, wartet auf manuellen Review vor Merge ⏳

**What was done**:
- Created `experiments/benchmark_neural_ode.py`: benchmark script for all 7 valid cell × wiring combinations × 6 ODE systems = 42 training runs
- Training config: 2000 iterations, batch_size=16, lr=1e-3, Dense units=16 (lrc_ar=2), NCP inter=16/command=8/motor=2
- Output: `results/neural_ode_benchmark_<timestamp>.json` + `.md` Markdown table
- Created `tests/experiments/test_benchmark_neural_ode.py`: 6 structural tests (no training)
- Total test suite: 70/70 passing
- Benchmark training NOT executed during development — user runs manually

**Files changed**:
- `experiments/benchmark_neural_ode.py` (new — 171 lines)
- `tests/experiments/test_benchmark_neural_ode.py` (new — 39 lines)

**Notes**:
- lrc_ar + NCP excluded (same constraint as step 9 smoke test)
- Branch pushed to origin, awaiting user review before merge
- Run with: `uv run python experiments/benchmark_neural_ode.py`

---

### 2026-03-12 (Phase 2 / Step 9)
**Branch**: `phase2/step9-smoke-test`
**Status**: Merged to main ✅

**What was done**:
- Created `experiments/smoke_test_combinations.py`: standalone smoke test running all 8 cell × wiring combinations
- Each combination: builds model + runs 3 gradient steps on synthetic random data (batch=2, timesteps=10, features=2)
- Result: 7 PASS + 1 XFAIL (`lrc_ar + NCP` — documented architectural incompatibility)
- Exit code 0 on expected outcome, 1 on any unexpected failure or XPASS
- Created `tests/experiments/test_smoke_combinations.py`: 5 structural tests (no training) validating config constants
- Total test suite: 64/64 passing

**Files changed**:
- `experiments/smoke_test_combinations.py` (new — 108 lines)
- `tests/experiments/test_smoke_combinations.py` (new — 30 lines)

**Notes**:
- `lrc_ar + NCP` incompatibility: lrc_ar passes raw inputs as `v_pre` into `_sigmoid` where `mu`/`sigma` have shape `(units, units)`. At the NCP inter layer, `input_dim=2` but `inter_neurons=8`, so `2 ≠ 8` → shape error. Documented as XFAIL, not a bug.
- STC cell excluded (blocked pending supervisor meeting)
- Next: Phase 3 (benchmarking / control tasks)

---

### 2026-03-12 (Phase 2 / Step 8)
**Branch**: `phase2/step8-ncp-wiring`
**Status**: Merged to main ✅

**What was done**:
- Created `src/wirings/ncp.py`: `SparseLinear` (fixed binary mask) + `NCPWiring` (3-layer inter→command→motor)
- Added `ncps` dependency (used only for `NCP` adjacency matrix generation — not for CfC/LTC cells)
- Replaced `make_model` with `make_dense_model` (stacked RNN, configurable `num_layers`) + `make_ncp_model` (NCP topology)
- Migrated `ODEFuncModel`, `verify_neural_ode.py`, `run_neural_ode.py` to `make_dense_model`
- 11 new tests — total suite: 59/59 passing

**Files changed**:
- `src/wirings/ncp.py` (new — `SparseLinear` + `NCPWiring`)
- `src/wirings/base_wiring.py` (cell=None optional)
- `src/wirings/__init__.py` (NCPWiring, SparseLinear exports)
- `src/models/rnn_model.py` (make_dense_model + make_ncp_model; make_model removed)
- `src/models/__init__.py` (updated exports)
- `src/tasks/neural_ode/ode_model.py` (migrated, wiring_type removed)
- `experiments/verify_neural_ode.py` (migrated)
- `experiments/run_neural_ode.py` (migrated)
- `experiments/configs/neural_ode_lrc_spiral.yaml` (wiring key removed)
- `tests/wirings/test_ncp_wiring.py` (new — 5 tests)
- `tests/models/test_make_model.py` (migrated + extended)
- `tests/neurons/test_cells.py` (2 tests migrated)
- `pyproject.toml` + `uv.lock` (ncps added)

**Notes**:
- NCP wiring: 3 stacked RNN layers (inter→command→motor), each using the chosen cell type (LRC/LSTM/CTRNN)
- `NCP` class used (not `AutoNCP`) for explicit fanout parameters; neuron ordering is `[motor|command|inter]` internally — masks extracted via `np.ix_` with `w._inter_neurons` etc.
- Sparse inter-layer connections from NCP adjacency matrix, applied as frozen binary mask in `SparseLinear`
- Input→Inter connections remain dense; Inter→Command and Command→Motor are sparse
- Next: `phase2/step9-smoke-test` (all cell×wiring combinations)

---

### 2026-03-12 (Phase 2 / Step 7)
**Branch**: `phase2/step7-new-cells`
**Status**: Merged to main ✅

**What was done**:
- Created `src/neurons/ctrnn_cell.py`: CT-RNN leaky integrator ODE, τ per neuron, Euler step
- Created `src/neurons/lstm_cell.py`: thin wrapper around `tf.keras.layers.LSTMCell`
- Extended `src/neurons/__init__.py`: exports `CTRNN_Cell`, `LSTM_Cell`
- Updated `src/models/rnn_model.py`: `'ctrnn'` and `'lstm'` in `_CELL_REGISTRY`
- 9 new tests in `tests/neurons/test_cells.py` — total suite: 48/48 passing

**Files changed**:
- `src/neurons/ctrnn_cell.py` (new)
- `src/neurons/lstm_cell.py` (new)
- `src/neurons/__init__.py` (modified)
- `src/models/rnn_model.py` (modified)
- `tests/neurons/test_cells.py` (modified — +9 tests)

**Notes**:
- LSTM overrides `state_size` → `[units, units]` and `get_initial_state` → two zero tensors
- CT-RNN: τ initialised to 1.0, NonNeg constraint, ε=1e-8 for stability
- Both cells support irregular sampling convention from BaseCell
- Next: `phase2/step8-ncp-wiring`

---

### 2026-03-12 (Phase 1 / Step 6)
**Branch**: `phase1/step6-verify-lrc-results`
**Status**: Merged to main ✅

**What was done**:
- Created `experiments/verify_neural_ode.py` with `run_verification()`, `check_convergence()`, `save_results()`, `main()`
- Runs LRC_AR + Dense on all 6 ODE systems for 2000 iterations each (matching original `neuralODE/run_ode.py`)
- Pass criterion: `final_loss < 0.5 * initial_loss` for each system
- Exit 0 if all converge, exit 1 otherwise
- Saves structured JSON to `results/neural_ode_lrc_baseline.json` (gitignored)
- Added `results/` to `.gitignore`
- 3 new tests — total suite: 39/39 passing

**Files changed**:
- `experiments/verify_neural_ode.py` (new)
- `tests/experiments/__init__.py` (new)
- `tests/experiments/test_verify_script.py` (new)
- `.gitignore` (modified — added results/)

**Notes**:
- Qualitative verification: loss decrease > 50% on all systems
- JSON baseline stored for Phase 3 benchmarking reference
- Next: `phase1/step7` (TBD — see roadmap)

---

### 2026-03-12 (Phase 1 / Step 5)
**Branch**: `phase1/step5-port-neural-ode-tasks`
**Status**: Merged to main ✅

**What was done**:
- Created `src/tasks/neural_ode/` package with 4 modules
- `datasets.py`: 6 ODE systems via `scipy.integrate.solve_ivp` (DOP853), replacing broken `tfdiffeq`
- `ode_model.py`: `ODEFuncModel` — Dense(units) → make_model RNN core → Dense(features)
- `solver.py`: `euler_odeint` — 15-line TF Euler integrator (equivalent to original `method='euler'`)
- `trainer.py`: `get_batch` + `train` loop using `GradientTape` + Adam + MAE loss
- `experiments/run_neural_ode.py`: CLI entry point reading YAML config
- 15 new tests (7 datasets + 2 ode_model + 2 solver + 4 trainer) — total suite: 36/36 passing

**Files changed**:
- `src/tasks/__init__.py` (new)
- `src/tasks/neural_ode/__init__.py` (new)
- `src/tasks/neural_ode/datasets.py` (new)
- `src/tasks/neural_ode/ode_model.py` (new)
- `src/tasks/neural_ode/solver.py` (new)
- `src/tasks/neural_ode/trainer.py` (new)
- `experiments/__init__.py` (new)
- `experiments/configs/neural_ode_lrc_spiral.yaml` (new)
- `experiments/run_neural_ode.py` (new)
- `tests/tasks/__init__.py` (new)
- `tests/tasks/neural_ode/__init__.py` (new)
- `tests/tasks/neural_ode/test_datasets.py` (new)
- `tests/tasks/neural_ode/test_ode_model.py` (new)
- `tests/tasks/neural_ode/test_solver.py` (new)
- `tests/tasks/neural_ode/test_trainer.py` (new)

**Notes**:
- `tfdiffeq` replaced: GitHub package broken on matplotlib ≥ 3.6 (seaborn-paper style removed)
- Original code used `method='euler'` for all LRC experiments → behaviour identical with our `euler_odeint`
- Next: `phase1/step6-verify-lrc-results`

---

### 2026-03-11 (Phase 1 / Step 4)
**Branch**: `phase1/step4-model-factory`
**Status**: Merged to main ✅

**What was done**:
- Created `src/wirings/` package: `BaseWiring` (abstract, abc.ABC) + `DenseWiring`
- Created `src/models/` package: `make_model(neuron_type, wiring_type, units, **kwargs)`
- String registry: `{'lrc': LRC_Cell, 'lrc_ar': LRC_AR_Cell}` / `{'dense': DenseWiring}`
- Factory accepts both string keys and class references
- `return_sequences=True` hardcoded (Phase 1: sequence tasks only)
- 11 new tests (4 wiring + 7 factory) — total suite: 21/21 passing

**Files changed**:
- `src/wirings/base_wiring.py` (new)
- `src/wirings/dense.py` (new)
- `src/wirings/__init__.py` (new)
- `src/models/rnn_model.py` (new)
- `src/models/__init__.py` (new)
- `tests/wirings/test_dense_wiring.py` (new)
- `tests/models/test_make_model.py` (new)

**Notes**:
- NCP wiring and CT-RNN/STC/LSTM cells deferred to Phase 2
- Next: `phase1/step5-port-neural-ode-tasks`

---

### 2026-03-10 (Phase 1 / Step 3)
**Branch**: `phase1/step3-tf-uv-setup`
**Status**: Merged to main

**What was done**:
- Created `pyproject.toml` mit TF 2.15 (last Keras-2-native release), numpy, matplotlib, scipy, pyyaml, tqdm, pandas; `tensorflow-metal` als optional `[metal]` extra
- Pinned Python 3.11 via `.python-version` (TF 2.15 hat kein Python-3.12-Wheel)
- Added hatchling src path config (`[tool.hatch.build.targets.wheel] packages = [\"src\"]`)
- Added `pytest` als dev dependency (`uv add --dev pytest`)
- Fixed `BaseCell` instantiation guard: `AbstractRNNCell` uses plain `type` (not ABCMeta), so added explicit `TypeError` in `__init__` when `type(self) is BaseCell`
- Wrote 10 smoke tests in `tests/neurons/test_cells.py` — all pass with TF 2.15.1
- Added deprecation comment to `environment.yml` (historical reference only)
- Updated `README.md` with uv setup instructions

**Files changed**:
- `pyproject.toml` (new + updated during setup)
- `.python-version` (new — pins Python 3.11)
- `uv.lock` (new)
- `tests/__init__.py`, `tests/neurons/__init__.py` (new)
- `tests/neurons/test_cells.py` (new — 10 tests)
- `src/neurons/base_cell.py` (fix: instantiation guard)
- `environment.yml` (deprecation comment)
- `README.md` (uv setup section added)

**Notes**:
- TF 2.15 chosen over 2.16 to avoid Keras 3 breaking changes
- `tfdiffeq` deferred to Phase 1 Step 5 (Neural ODE porting)
- `keras-ncps` deferred to Phase 2 (NCP wiring)
- Next: `phase1/step4-model-factory` (`make_model(neuron, wiring)` factory in `src/models/`)

---

### 2026-03-10 (Phase 1 / Step 2)
**Branch**: `phase1/step2-basecell-interface`
**Status**: Merged to main

**What was done**:
- Created `src/neurons/base_cell.py` — abstract `BaseCell(AbstractRNNCell)` mit: `units` in `__init__`, concrete `state_size`/`output_size` properties, abstract `build`/`call`, `get_initial_state`, documented irregular sampling convention
- Refactored `LRC_Cell` and `LRC_AR_Cell` to inherit from `BaseCell` — removed duplicated `self.units`, `state_size` property
- Updated `src/neurons/__init__.py` to export `BaseCell`
- Updated `Meta/claude-project-status.md`

**Files changed**:
- `src/neurons/base_cell.py` (new)
- `src/neurons/lrc_cell.py` (inherits BaseCell, -4 lines)
- `src/neurons/lrc_ar_cell.py` (inherits BaseCell, -4 lines)
- `src/neurons/__init__.py` (exports BaseCell)

**Notes**:
- 2 atomic commits + merge commit
- Runtime import test deferred (TF not yet installed — came mit Step 3)
- Next: TF 2.4.1 → 2.15 upgrade + `pyproject.toml` / `uv` setup

---

### 2026-03-10 (README Restructure)
**Branch**: `chore/readme-restructure`
**Status**: Merged to main

**What was done**:
- Moved original LRC paper README → `docs/lrc-original-readme.md`
- Created new root `README.md` describing thesis benchmark scope: 4×2 matrix, tasks, metrics, structure, phases

**Files changed**:
- `README.md` (replaced — now thesis project description)
- `docs/lrc-original-readme.md` (new — original paper README)

---

### 2026-03-10 (Phase 1 / Step 1)
**Branch**: `phase1/step1-migrate-lrc-cells`
**Status**: Merged to main

**What was done**:
- Created `src/` and `src/neurons/` as Python packages
- Migrated `classification/lrc_cell.py` → `src/neurons/lrc_cell.py` (exact copy, no code changes)
- Migrated `neuralODE/lrc_ar_cell.py` → `src/neurons/lrc_ar_cell.py` (exact copy, no code changes)
- Added `src/neurons/__init__.py` exporting `LRC_Cell`, `LRC_AR_Cell`

**Files changed**:
- `src/__init__.py` (new)
- `src/neurons/__init__.py` (new)
- `src/neurons/lrc_cell.py` (new — migrated from `classification/`)
- `src/neurons/lrc_ar_cell.py` (new — migrated from `neuralODE/`)

**Notes**:
- 3 atomic commits on branch + 1 merge commit on main
- Runtime import verification deferred to step3 (after TF/uv setup)
- Next: `phase1/step2-basecell-interface`

---

### 2026-03-10 (REPPO Research)
**Branch**: N/A (literature research, no code changes)
**Status**: Done

**What was done**:
- Analysed TU Wien lecture PDF (Brunnbauer, Policy Gradient Algorithms) — 46 slides
- Identified REPPO (Relative Entropy Pathwise Policy Optimization) as the novel algorithm from the Grosu research group
- Searched online: arXiv:2507.11019 (Voelcker, Brunnbauer, Hussing, Nauman, Abbeel, Eaton, Grosu, Farahmand, Gilitschenski)
- Added BibTeX entry `voelcker2025relative` to `Literature/references.bib`
- Created paper note `Papers/voelcker2025relative.md` with full summary, formulas, relevance
- Created concept note `Concepts/REPPO.md` with PPO comparison table and algorithm overview
- Updated `Thesis/benchmark-repo-roadmap.md`: REPPO designated as planned RL algorithm for control tasks

**Files changed**:
- `Literature/references.bib` (REPPO entry added)
- `Papers/voelcker2025relative.md` (new)
- `Concepts/REPPO.md` (new)
- `Thesis/benchmark-repo-roadmap.md` (RL algorithm section added)

**Notes**:
- REPPO resolves the open question "which RL algorithm for control tasks"
- Directly from Grosu group → thematically consistent with LRC benchmark
- Code available at https://github.com/cvoelcker/reppo (JAX + PyTorch)

---

### 2026-03-10 (Initial Setup)
**Branch**: N/A (setup)
**Status**: Done

**What was done**:
- Cloned forked LRC repo (`a2ngerer/LRC`) into `master_thesis/code/`
- Created `code/CLAUDE.md` with branch workflow and core rules
- Created `code/claude_instructions/code-workflow.md` with full phased development plan (Phase 1–4)
- Created this Work Documentation file
- Configured Claude hook to auto-remind about updating this file after `/code` changes

**Files changed**:
- `code/CLAUDE.md` (new)
- `code/claude_instructions/code-workflow.md` (new)
- `obsidian_master_thesis/Thesis/work-documentation.md` (new)
- `obsidian_master_thesis/Thesis/lrc-repo-overview.md` (new)
- `obsidian_master_thesis/Thesis/benchmark-repo-roadmap.md` (new)
- `~/.claude/settings.json` — hooks added
- `~/.claude/hooks/track-code-changes.sh` (new)
- `~/.claude/hooks/code-docs-reminder.sh` (new)

**Notes**:
- Phase 1 not yet started; next step: `phase1/step1-migrate-lrc-cells`
- Phase 2 / step2 (STC cell) is blocked pending supervisor meeting for spec
