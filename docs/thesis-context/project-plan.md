# Masterarbeit Projektplan

**Projekt:** Exploring Bio-inspired Neural Networks — A Comparative Study of Architectural and Model Mechanisms (Proposal v3)
**Zeitraum:** 1. März 2026 – 30. September 2026 (7 Monate, ~30 Wochen)
**Projektstatus:** Phase 3 aktiv (Benchmark-Repo + Gradient-Flow-Analyse). Phasen 1 & 2 abgeschlossen: Proposal v3 finalisiert, Thesis-LaTeX-Skelett mit 9 Kapiteln + 4-teiligem Appendix steht ([[latex-template]]).
**Letzte Aktualisierung:** 2026-04-20

## Forschungs-Scope (Proposal v3)

**RQs:**
1. **RQ1 Performance** — LRC/STC vs. LSTM/GRU (Zeitreihen + Kontrolle)
2. **RQ2 Mechanismen** — liquid elastance ε(w_i), Saturation, Gating
3. **RQ3 Wiring** — NCP vs. Dense, Interaktion mit Neuron-Modell
4. **RQ4 Konvergenz/Gradient Flow** — warum LRC+NCP schlechter konvergiert als LRC+Dense
5. **RQ5 Intermediate Models** — CT-RNN + ε(w_i) als Ablations-Werkzeug

**Experimental Matrix (4×2):** LSTM · CT-RNN · STC · LRC  ×  Dense · NCP
**Out of Scope:** Transformer, SSM (S4/Mamba), xLSTM, SNN, Hardware-Deployment
**Potential Extended Contribution (bedingt):** Neue Wiring-Architektur L1–L6 (kortikal inspiriert)

---

## Projektmanagement-Framework

Dieser Plan nutzt evidenzbasierte Ansätze aus dem wissenschaftlichen Projektmanagement:

- **Work Breakdown Structure (WBS):** Aufteilung in überschaubare Arbeitspakete
- **Critical Path Method:** Identifikation kritischer Abhängigkeiten
- **Time-boxing:** Feste Zeitfenster pro Arbeitspaket (verhindert Perfektionismus)
- **Buffer Management:** 20% Pufferzeit für Unvorhergesehenes
- **Iterative Deliverables:** Frühe, häufige Zwischenergebnisse
- **Daily Progress Tracking:** Kontinuierliches Momentum durch kleine tägliche Schritte

### ADHS-optimierte Strukturen

- Maximale Arbeitspaket-Größe: 2-3 Stunden (verhindert Overwhelm)
- Tägliche konkrete Aufgaben (keine vagen Ziele)
- Externe Deadlines als Ankerpunkte (externe Motivation)
- Visueller Fortschritt (sofortiges Feedback)
- Built-in Review-Zyklen (verhindert langes Driften)

---

## Critical Path & Dependencies

```
CRITICAL PATH (kann nicht parallelisiert werden):
Proposal → Struktur → Experimente → Analyse → Writing → Review → Abgabe

PARALLELISIERBAR:
- Literatur (kontinuierlich während gesamter Laufzeit)
- LaTeX Setup (parallel zu Literatur)
- Code Implementation (während Strukturierung)
```

**Gesamtdauer Critical Path:** ~200 Tage (bei sequenzieller Abarbeitung)
**Verfügbare Zeit:** 214 Tage
**Buffer:** 14 Tage (6.5%)

---

## Phasenübersicht

**WICHTIG:** Proposal wird bereits im Februar fertiggestellt und Anfang März abgegeben. Dies verschafft zusätzlichen Buffer (~6 Wochen) für die Hauptarbeit.

| Phase | Dauer | Zeitraum | Deliverable | Status |
|-------|-------|----------|-------------|--------|
| 1: Foundation & Proposal | 6 Wochen | Feb – 17.3 | Proposal v1 abgegeben, v2/v3 iteriert | ✓ |
| 2: Structure & Deep Dive | 5 Wochen | 18.3 – 19.4 | Thesis-Skelett (9 Ch. + 4 Appendix) + Lit-Basis | ✓ |
| 3: Implementation & Experiments | 12 Wochen | 20.4 – 12.7 | 4×2-Matrix + Gradient-Flow + Intermediate-Ablations | ☐ |
| 4: Writing & Analysis | 8 Wochen | 13.7 – 6.9 | Draft komplett (9 Kapitel + Appendix) | ☐ |
| 5: Review & Finalization | 3 Wochen | 7.9 – 30.9 | Finale Abgabe | ☐ |

**Gesamtdauer:** 34 Wochen (238 Tage verfügbar)
**Buffer:**
- Phase 3: +1 Woche (flexibel für Experimente)
- Phase 4: +2 Wochen (mehr Zeit für Writing)
- Phase 5: +2 Wochen (längere Review-Zyklen)
- Reserve: +1 Woche unallocated buffer

---

# Phase 1: Foundation & Proposal (Feb – Mitte April) ✓

**Ziel:** Fundierte Basis schaffen und Proposal abgeben
**Kritischer Erfolgsfaktor:** Research Question & Methodology klar definiert
**Status:** ✓ Abgeschlossen — Proposal v1 eingereicht 17.3., v2 (29.3.) mit Scope-Erweiterung, v3 (19.4.) mit Titel-Update, CT-RNN/GRU-Reshuffle, RQ4+RQ5, L1–L6 Extension, ARC-Kurs. Details: [[scope-decisions]], [[structure]].

## Arbeitspaket 1.1: Initiale Literaturrecherche
**Dauer:** ~2 Wochen (Februar)
**Aufwand:** ~20 Stunden
**Deliverable:** 5 Papers gelesen, zusammengefasst, in Obsidian dokumentiert

### Tägliche Aufgaben (Mo-Fr, je 2h)
- [ ] **Tag 1-2:** Reading list erstellen (10-15 Core Papers identifizieren)
- [ ] **Tag 3:** Paper #1 lesen (Vaswani et al. oder ähnlich Core Paper)
- [ ] **Tag 4:** Paper #1 zusammenfassen → `Papers/paper-name.md`
- [ ] **Tag 5:** Paper #2 lesen
- [ ] **Tag 6:** Paper #2 zusammenfassen
- [ ] **Tag 7:** Paper #3 lesen
- [ ] **Tag 8:** Paper #3 zusammenfassen
- [ ] **Tag 9:** Paper #4 lesen
- [ ] **Tag 10:** Paper #4 zusammenfassen
- [ ] **Tag 11:** Paper #5 lesen
- [ ] **Tag 12:** Paper #5 zusammenfassen
- [ ] **Tag 13:** Concept notes erstellen für wiederkehrende Themen
- [ ] **Tag 14:** BibLaTeX Einträge pflegen (`references.bib`)

**Checkpoint 1.1:** ✓ 5 Papers dokumentiert | [[reading-list]] aktualisiert

---

## Arbeitspaket 1.2: Research Question Development
**Dauer:** ~1 Woche (Februar)
**Aufwand:** ~10 Stunden
**Deliverable:** Finalisierte Research Question

### Aufgaben (3x 3h Sessions)
- [ ] **Session 1 (3h):** Larisa's Proposal analysieren, Struktur verstehen
- [ ] **Session 2 (3h):** Research Question Draft 1 schreiben, 3 Varianten formulieren
- [ ] **Session 3 (2h):** Betreuer-Meeting vorbereiten (Questions, Agenda)
- [ ] **Session 4 (2h):** Betreuer-Meeting durchführen, Feedback dokumentieren
- [ ] **Session 5 (1h):** Research Question finalisieren basierend auf Feedback

**Checkpoint 1.2:** ✓ Research Question approved by Betreuer

---

## Arbeitspaket 1.3: Proposal Drafting
**Dauer:** ~2 Wochen (Februar)
**Aufwand:** ~25 Stunden
**Deliverable:** Proposal Draft 1

### Tägliche Aufgaben (Mo-Fr, je 2.5h)
- [ ] **Tag 1-2:** Abstract schreiben (300 words)
- [ ] **Tag 3-4:** Introduction schreiben (2 pages)
- [ ] **Tag 5-6:** Related Work schreiben (3 pages, aus Paper-Summaries)
- [ ] **Tag 7-8:** Methodology Sektion schreiben (3 pages)
- [ ] **Tag 9:** Expected Results & Contributions (1 page)
- [ ] **Tag 10:** Timeline & Milestones definieren (1 page)

**Checkpoint 1.3:** ✓ Proposal Draft 1 komplett (10-12 Seiten)

---

## Arbeitspaket 1.4: Proposal Review & Finalization
**Dauer:** ~1 Woche (Ende Feb/Anfang März)
**Aufwand:** ~10 Stunden
**Deliverable:** Finales Proposal

### Aufgaben
- [ ] **Tag 1 (2h):** Self-Review mit Checkliste (Structure, Clarity, Grammar)
- [ ] **Tag 2 (1h):** Proposal an Betreuer schicken
- [ ] **Tag 3-5:** Warten auf Feedback (währenddessen: Paper #6 lesen)
- [ ] **Tag 6 (2h):** Feedback-Meeting mit Betreuer
- [ ] **Tag 7-8 (4h):** Feedback einarbeiten (strukturell & inhaltlich)
- [ ] **Tag 9 (1h):** Final polish (Typos, Formatierung, References check)
- [ ] **Tag 10:** **PROPOSAL ABGEBEN**

**Checkpoint 1.4:** ✓ Proposal abgegeben

---

### MILESTONE 1 COMPLETE: PROPOSAL ABGEGEBEN ✓
**Erreicht am:** 2026-03-17 (v1), 2026-03-29 (v2), 2026-04-19 (v3)
**Review-Fragen:**
- Wurden alle Deadlines eingehalten? → ✓ v1 fristgerecht
- Gibt es Lessons Learned für Phase 2? → Iterative Scope-Verfeinerung nötig (v2/v3 Feedback-Loops mit Monika)
- Muss der Zeitplan adjustiert werden? → Phase 2 +1 Woche, kompensiert durch +6 Wochen Buffer aus früher Proposal-Abgabe

---

# Phase 2: Structure & Deep Dive (18.3 – 19.4) ✓

**Ziel:** Thesis-Architektur finalisieren & Literaturbasis
**Kritischer Erfolgsfaktor:** Klare Struktur verhindert späteres Umschreiben
**Status:** ✓ Abgeschlossen — 9-Kapitel-Skelett + 4 Appendix-Teile + erweitertes Frontmatter kompiliert (34 Seiten Placeholder, 2026-04-19). 13 Papers im Vault dokumentiert (weitere 7 rollend in Phase 3).

## Arbeitspaket 2.1: LaTeX Environment Setup ✓
**Dauer:** 3 Tage (erreicht 2026-04-19)
**Aufwand:** ~6 Stunden
**Deliverable:** LaTeX kompiliert PDF
**Status:** Abgeschlossen — Details in [[latex-template]]

- [x] Template: Custom (report-Klasse, Proposal-Stil)
- [x] `Thesis/main.tex` + `bibliography.bib` aus Proposal übernommen
- [x] Chapter/Appendix/Frontmatter-Files angelegt
- [x] Test-Kompilierung: 34 Seiten ohne Errors

---

## Arbeitspaket 2.2: Thesis Architecture Design ✓
**Dauer:** 1 Woche (erreicht 2026-04-19)
**Aufwand:** ~15 Stunden
**Deliverable:** Thesis-Skelett nach Spec (9 Kapitel + 4 Appendix + erweitertes Frontmatter)
**Status:** Abgeschlossen — Brainstorm → Design-Spec (`docs/superpowers/specs/2026-04-19-thesis-chapter-structure-design.md`) → 25-Task-Implementation-Plan → ausgeführt.

**Finale Struktur:**

1. Introduction — Motivation, RQs 1–5, Contributions
2. Background — RNNs, EECs (CT-RNN → LTC → STC → LRC), liquid elastance, NCP, Neural ODEs
3. Related Work — architektonische Vergleiche, CPS-Anwendungen, Gap-Analyse
4. Architectures and Models — alle 4 Neuron-Typen + Intermediate Models, formal
5. Experimental Setup — Benchmarks, Metriken, Training-Protokoll, Solver-Budget
6. Results: Benchmarks — 4×2-Matrix (RQ1, RQ3)
7. Results: Mechanistic — Gradient Flow (RQ4), Intermediate-Ablations (RQ2, RQ5)
8. Discussion — Interpretation, Limitations, (ggf.) L1–L6 Ausblick
9. Conclusion — Beiträge, Future Work

**Appendix:** A Extended Derivations · B Experimental Details · C Additional Results · D Reproducibility
**Frontmatter:** TU-Titelseite · Eidesstattliche Erklärung · Kurzfassung/Abstract · Acknowledgements · ToC/LoF/LoT · List of Abbreviations · List of Symbols

**Offen in Phase 2-Ausklang:**
- [ ] Matrikelnummer in `Thesis/frontmatter/titlepage.tex`
- [ ] Betreuer-Review der Struktur (Monika / Radu)

---

## Arbeitspaket 2.3: Comprehensive Literature Review
**Dauer:** 2.5 Wochen (26.4 - 15.5)
**Aufwand:** ~30 Stunden
**Deliverable:** 15 weitere Papers (Total: 20)

### Wöchentliche Struktur (Mo-Fr, je 2h)
**Woche 1:**
- [ ] Paper #6-7 lesen & zusammenfassen
- [ ] Paper #8-9 lesen & zusammenfassen
- [ ] Paper #10 lesen & zusammenfassen

**Woche 2:**
- [ ] Paper #11-12 lesen & zusammenfassen
- [ ] Paper #13-14 lesen & zusammenfassen
- [ ] Paper #15 lesen & zusammenfassen

**Woche 3:**
- [ ] Paper #16-17 lesen & zusammenfassen
- [ ] Paper #18-19 lesen & zusammenfassen
- [ ] Paper #20 lesen & zusammenfassen
- [ ] Concept notes erweitern (`Concepts/`)
- [ ] Related Work pre-writing (key themes identifizieren)

**Checkpoint 2.3:** ✓ 20 Papers vollständig dokumentiert

---

### MILESTONE 2 COMPLETE: STRUKTUR & LITERATUR ✓
**Erreicht am:** 2026-04-19 (Target: 12. April — +1 Woche durch Proposal-v3-Iteration)
**Review-Fragen:**
- Ist die Thesis-Struktur mit Betreuer abgestimmt? → **offen**, Review mit Monika ausständig
- Sind alle 20 Papers ausreichend verstanden? → **13/20**, Rest rollend in Phase 3
- Ist LaTeX-Setup stabil? → ✓ 34 Seiten, 9 Ch. + 4 Appendix + Frontmatter kompiliert

---

# Phase 3: Implementation & Experiments (12 Wochen)

**Ziel:** Alle Experimente durchgeführt, Daten analysiert
**Kritischer Erfolgsfaktor:** Früh anfangen, iterativ arbeiten (nicht erst am Ende testen)

## Arbeitspaket 3.1: Development Environment & Baseline
**Dauer:** 2 Wochen (13.4 - 26.4)
**Aufwand:** ~30 Stunden
**Deliverable:** Funktionierende Baseline

### Woche 1: Environment Setup
- [ ] **Tag 1-2 (4h):** Python env setup (uv, dependencies)
- [ ] **Tag 3-4 (4h):** Dataset acquisition & exploration
- [ ] **Tag 5 (2h):** Data preprocessing pipeline

### Woche 2: Baseline Implementation
- [ ] **Tag 6-7 (6h):** Standard NN implementation
- [ ] **Tag 8-9 (6h):** Training loop & evaluation metrics
- [ ] **Tag 10 (4h):** Baseline test run, debugging
- [ ] **Tag 11 (2h):** Baseline results dokumentieren

**Checkpoint 3.1:** ✓ Baseline trained & evaluated

---

## Arbeitspaket 3.2: Novel Neuron Types + Intermediate Models (RQ5)
**Dauer:** 4 Wochen (30.5 - 26.6)
**Aufwand:** ~55 Stunden
**Deliverable:** Code für alle Neuron-Varianten inkl. Intermediate Models

### Woche 1: STC Cell
- [ ] **Tag 1-3 (9h):** Implement STC_Cell (Saturated LTC)
- [ ] **Tag 4-5 (6h):** Unit tests, integration tests

### Woche 2: Intermediate Models (CT-RNN + liquid elastance)
- [ ] **Tag 6-8 (9h):** CT-RNN mit ε(w_i) implementieren als Ablations-Werkzeug (RQ5)
- [ ] **Tag 9-10 (6h):** Unit tests, Vergleich CT-RNN vs. Intermediate vs. LRC

### Woche 3: Gradient Flow Tooling (RQ4)
- [ ] **Tag 11-12 (6h):** Gradient-Norm Logging & Visualisierung einbauen (LRC+NCP Konvergenz-Analyse)
- [ ] **Tag 13-14 (6h):** Code documentation, README
- [ ] **Tag 15 (3h):** Code review, refactoring

### Woche 4: Integration
- [ ] **Tag 16-17 (6h):** Alle Modelle in make_model / make_ncp_model integrieren
- [ ] **Tag 18-19 (6h):** Smoke-Tests für alle Konfigurationen inkl. Intermediate Models
- [ ] **Tag 20 (3h):** Baseline-JSON für Intermediate Models erstellen

**Checkpoint 3.2:** ✓ Alle Neuron-Typen inkl. Intermediate Models implementiert & getestet

---

## Arbeitspaket 3.3: Experimental Runs
**Dauer:** 3 Wochen (27.6 - 17.7)
**Aufwand:** ~45 Stunden (+ Rechenzeit)
**Deliverable:** Raw experimental results

### Woche 1: Baseline 4×2 Matrix
- [ ] **Tag 1 (2h):** Experiment config setup (Hyperparameters, etc.)
- [ ] **Tag 2-3:** Experiment 1 — LRC/STC/LSTM/CT-RNN × Dense (training läuft)
- [ ] **Tag 4-5:** Experiment 2 — LRC/STC/LSTM/CT-RNN × NCP (training läuft)
- [ ] **Tag 6 (2h):** Preliminary results check, LRC+NCP Konvergenz prüfen (RQ4)

### Woche 2: Gradient Flow & Convergence Analysis (RQ4)
- [ ] **Tag 7-8:** Experiment 3 — LRC+NCP Gradient Flow Analysis (Norm-Verläufe, Layer-wise)
- [ ] **Tag 9-10:** Experiment 4 — CT-RNN+NCP und LRC+Dense zum Vergleich
- [ ] **Tag 11 (2h):** Intermediate analysis, Hypothesen für RQ4 prüfen

### Woche 3: Intermediate Model Ablations (RQ5)
- [ ] **Tag 12-14:** Experiment 5 — CT-RNN + ε(w_i) in Dense & NCP (Ablation RQ5)
- [ ] **Tag 15-16:** Additional runs (if needed for significance)
- [ ] **Tag 17 (4h):** All results aggregation & backup

**Checkpoint 3.3:** ✓ Alle Experimente durchgeführt, Results gespeichert

---

## Arbeitspaket 3.4: Data Analysis & Visualization
**Dauer:** 2 Wochen (11.7 - 24.7)
**Aufwand:** ~25 Stunden
**Deliverable:** Publication-ready Figures & Tables

### Woche 1: Statistical Analysis
- [ ] **Tag 1-2 (6h):** Data cleaning & aggregation
- [ ] **Tag 3-4 (6h):** Statistical tests (significance, confidence intervals)
- [ ] **Tag 5 (3h):** Key metrics tables

### Woche 2: Visualization
- [ ] **Tag 6-8 (9h):** Plots erstellen (learning curves, comparisons, etc.)
- [ ] **Tag 9-10 (4h):** Error analysis, qualitative analysis
- [ ] **Tag 11 (2h):** Results summary in `Thesis/results-summary.md`

**Checkpoint 3.4:** ✓ Analyse komplett, Figures/Tables ready

---

## Arbeitspaket 3.5: Buffer Week
**Dauer:** 1 Woche (25.7 - 31.7)
**Zweck:** Puffer für Verzögerungen, zusätzliche Experimente, oder Vorarbeit für Writing

- [ ] Re-runs falls nötig
- [ ] Zusätzliche Analysen
- [ ] Oder: Start Chapter 1 Drafting (early start)

**Checkpoint 3.5:** ✓ Phase 3 komplett abgeschlossen, ready for writing

---

### MILESTONE 3 COMPLETE: EXPERIMENTE & ANALYSE
**Erreicht am:** _____ (Target: 5. Juli)
**Review-Fragen:**
- Sind die Results überzeugend?
- Gibt es Gaps in den Daten?
- Sind alle Figures/Tables publication-ready?

---

# Phase 4: Writing & Analysis (8 Wochen, 13.7 – 6.9)

**Ziel:** Alle 9 Kapitel + Appendix im Draft-Status, LaTeX kompiliert
**Kritischer Erfolgsfaktor:** Daily writing habit (2h/Tag minimum); Schreib-Reihenfolge folgt Argumentationsaufbau, nicht Kapitel-Nummerierung

## Writing Protocol

**Daily Structure (Mo-Fr):**
- Morning Session (2h): Schreiben (focus, no distractions)
- Afternoon Session (1h): Edit/Review vom Vortag

**Strategie:**
- Erst vollständiger Draft (imperfect), dann polieren
- Nicht am selben Tag schreiben & editieren (frische Perspektive)
- Target: 500–1000 words/day
- **Schreib-Reihenfolge:** Architectures & Models → Exp. Setup → Results (Benchmarks & Mechanistic) → Background → Related Work → Introduction → Discussion → Conclusion → Frontmatter/Appendix (Vorteil: harte Fakten zuerst, narrative Rahmung zuletzt)

## Arbeitspaket 4.1: Architectures and Models (Ch. 4)
**Dauer:** 5 Tage (13.7 – 17.7)
**Aufwand:** ~12 Stunden
**Target:** 5000 words (~10 pages)

- [ ] **Tag 1:** Architecture Overview + Notation (aus `list_of_symbols.tex`)
- [ ] **Tag 2:** LSTM + CT-RNN (Baselines) formal
- [ ] **Tag 3:** STC + LRC (Novel) formal, liquid elastance ε(w_i)
- [ ] **Tag 4:** Intermediate Models (CT-RNN + ε(w_i)) — Ablations-Design (RQ5)
- [ ] **Tag 5:** NCP-Wiring vs. Dense, Kopplungsregeln, Figures (Zellen-Diagramme)

**Checkpoint 4.1:** ✓ Ch. 4 Draft 1; technische Herleitungen → Appendix A

---

## Arbeitspaket 4.2: Experimental Setup (Ch. 5)
**Dauer:** 3 Tage (20.7 – 22.7)
**Aufwand:** ~7 Stunden
**Target:** 2500 words (~5 pages)

- [ ] **Tag 1:** Benchmarks (Lotka-Volterra, Van der Pol, irregular TS, Pendulum, CartPole)
- [ ] **Tag 2:** Training-Protokoll, Hyperparameter-Tuning pro Architektur, Seeds
- [ ] **Tag 3:** Metriken (MSE/NRMSE, episodic return), Solver-Budget, Hardware

**Checkpoint 4.2:** ✓ Ch. 5 Draft 1; Detail-Configs → Appendix B

---

## Arbeitspaket 4.3: Results — Benchmarks (Ch. 6)
**Dauer:** 4 Tage (23.7 – 28.7)
**Aufwand:** ~10 Stunden
**Target:** 3500 words (~7 pages)

- [ ] **Tag 1:** 4×2-Matrix Performance-Tabelle (RQ1)
- [ ] **Tag 2:** Dense vs. NCP Interaktion (RQ3)
- [ ] **Tag 3:** Training Efficiency + Solver-Overhead
- [ ] **Tag 4:** Qualitative Beispiele, Learning Curves

**Checkpoint 4.3:** ✓ Ch. 6 Draft 1; Extra-Plots → Appendix C

---

## Arbeitspaket 4.4: Results — Mechanistic (Ch. 7)
**Dauer:** 4 Tage (29.7 – 3.8)
**Aufwand:** ~10 Stunden
**Target:** 3500 words (~7 pages)

- [ ] **Tag 1:** Gradient-Flow-Analyse LRC+NCP vs. LRC+Dense (RQ4)
- [ ] **Tag 2:** Intermediate-Model-Ablations CT-RNN + ε(w_i) (RQ5, RQ2)
- [ ] **Tag 3:** Saturation vs. Gating Mechanismen (RQ2)
- [ ] **Tag 4:** Phase Portraits / Activation Patterns (Interpretability)

**Checkpoint 4.4:** ✓ Ch. 7 Draft 1; erweiterte Analysen → Appendix C

---

## Arbeitspaket 4.5: Background (Ch. 2)
**Dauer:** 4 Tage (4.8 – 7.8)
**Aufwand:** ~8 Stunden
**Target:** 3500 words (~7 pages)

- [ ] **Tag 1:** RNNs + Traditional Gating (LSTM, GRU)
- [ ] **Tag 2:** EEC-Familie CT-RNN → LTC → STC → LRC (konsolidieren aus [[Concepts/index]])
- [ ] **Tag 3:** Liquid Elastance, Membrane Capacitance (Howell, Severin, Kumar)
- [ ] **Tag 4:** NCP + Neural ODEs; Notation & Preliminaries

**Checkpoint 4.5:** ✓ Ch. 2 Draft 1

---

## Arbeitspaket 4.6: Related Work (Ch. 3)
**Dauer:** 3 Tage (10.8 – 12.8)
**Aufwand:** ~7 Stunden
**Target:** 3000 words (~6 pages)

- [ ] **Tag 1:** Taxonomy: EEC-Linie, Gated RNNs, NCP-Anwendungen, große Sequenzmodelle (Transformer/SSM/xLSTM als State-of-the-Art-Kontext)
- [ ] **Tag 2:** Architektonische Vergleiche in RNN-Literatur; CPS-Anwendungen
- [ ] **Tag 3:** Gap Analysis, Positioning (Scope-Abgrenzung Transformer/SSM/SNN)

**Checkpoint 4.6:** ✓ Ch. 3 Draft 1

---

## Arbeitspaket 4.7: Introduction (Ch. 1)
**Dauer:** 3 Tage (13.8 – 15.8)
**Aufwand:** ~6 Stunden
**Target:** 2500 words (~5 pages)

- [ ] **Tag 1:** Motivation & Problem Statement (aus Proposal v3 übernehmen/kürzen)
- [ ] **Tag 2:** 5 Research Questions + Contributions
- [ ] **Tag 3:** Thesis Outline (Verweise auf Ch. 2–9), Edit

**Checkpoint 4.7:** ✓ Ch. 1 Draft 1

---

## Arbeitspaket 4.8: Discussion (Ch. 8)
**Dauer:** 4 Tage (17.8 – 20.8)
**Aufwand:** ~8 Stunden
**Target:** 3000 words (~6 pages)

- [ ] **Tag 1:** Key Findings Summary (mapped auf RQ1–RQ5)
- [ ] **Tag 2:** Interpretation & Implikationen für CPS-Deployment
- [ ] **Tag 3:** Limitations (Scope, Skalen, fehlende Hardware-Tests)
- [ ] **Tag 4:** Future Work — inkl. L1–L6 Wiring-Architektur-Ausblick (falls nicht erreicht) oder Einbettung (falls Extended Contribution erreicht)

**Checkpoint 4.8:** ✓ Ch. 8 Draft 1

---

## Arbeitspaket 4.9: Conclusion (Ch. 9)
**Dauer:** 2 Tage (21.8 – 22.8)
**Aufwand:** ~4 Stunden
**Target:** 1500 words (~3 pages)

- [ ] **Tag 1:** Summary of Contributions (1–4 aus Proposal + ggf. L1–L6)
- [ ] **Tag 2:** Closing Remarks

**Checkpoint 4.9:** ✓ Ch. 9 Draft 1

---

## Arbeitspaket 4.10: Appendix A–D
**Dauer:** 4 Tage (24.8 – 27.8)
**Aufwand:** ~10 Stunden

- [ ] **Tag 1:** **A — Extended Derivations:** Euler-Schema LRC, ε(w_i)-Form, Stabilitätsargumente
- [ ] **Tag 2:** **B — Experimental Details:** Hyperparameter-Tabellen pro Konfiguration, Datensatz-Specs, REPPO-Setup
- [ ] **Tag 3:** **C — Additional Results:** alle 8 Konfigurationen × alle Benchmarks (seeds, confidence intervals)
- [ ] **Tag 4:** **D — Reproducibility:** Repo-Struktur, `make_model`/`make_ncp_model`-API, Seeds, HW-Spezifikation

**Checkpoint 4.10:** ✓ Appendix A–D Draft 1

---

## Arbeitspaket 4.11: Frontmatter
**Dauer:** 2 Tage (28.8 – 29.8)
**Aufwand:** ~5 Stunden

- [ ] **Tag 1:** Abstract (EN, 300 words), Kurzfassung (DE, 300 words), Acknowledgements
- [ ] **Tag 2:** List of Abbreviations, List of Symbols befüllen; Titelseite (Matrikelnr., Datum), Statutory Declaration

**Checkpoint 4.11:** ✓ Frontmatter vollständig

---

## Arbeitspaket 4.12: LaTeX Integration & Self-Review
**Dauer:** 4 Tage (31.8 – 3.9)
**Aufwand:** ~12 Stunden
**Deliverable:** Kompilierte Thesis Draft 1

- [ ] **Tag 1:** Alle 9 Kapitel einfügen, Cross-Refs prüfen
- [ ] **Tag 2:** Appendix A–D + Figures/Tables integrieren
- [ ] **Tag 3:** Bibliography-Check (alle Citekeys vorhanden; Proposal-bib vs. Vault-bib vs. Thesis-bib konsistent)
- [ ] **Tag 4:** Read-through Ch. 1–9 (Structure, Flow), Typo-Pass

**Checkpoint 4.12:** ✓ Thesis kompiliert (~70–90 pages PDF) + self-reviewed

---

## Arbeitspaket 4.13: Buffer / Pre-Review Polish
**Dauer:** 3 Tage (4.9 – 6.9)
**Zweck:** Puffer vor Betreuer-Review

- [ ] Offene Stellen schließen, fehlende Figures
- [ ] Executive Summary für Monika vorbereiten (Review-Fokuspunkte)

---

### MILESTONE 4 COMPLETE: DRAFT KOMPLETT
**Erreicht am:** _____ (Target: 6. September)
**Review-Fragen:**
- Sind alle 9 Kapitel + Appendix A–D im Draft-Status?
- Kompiliert LaTeX ohne Errors?
- Sind alle 5 RQs im Draft beantwortet?
- Ist der Argumentationsfluss klar (Ch. 4 → Ch. 5 → Ch. 6 → Ch. 7 → Ch. 8)?

---

# Phase 5: Review & Finalization (3 Wochen, 7.9 – 30.9)

**Ziel:** Finale Version abgabebereit
**Kritischer Erfolgsfaktor:** Schnelle Feedback-Zyklen, keine großen Änderungen mehr

## Arbeitspaket 5.1: Betreuer Review Cycle 1
**Dauer:** 1.5 Wochen (7.9 - 17.9)
**Aufwand:** ~15 Stunden

- [ ] **Tag 1 (2h):** PDF an Betreuer schicken
- [ ] **Tag 2-5:** Warten auf Feedback (währenddessen: Language polish)
- [ ] **Tag 6 (2h):** Feedback-Meeting
- [ ] **Tag 7-8 (8h):** Kritische Punkte einarbeiten (Structure, Content)
- [ ] **Tag 9 (3h):** Revised version kompilieren

**Checkpoint 5.1:** ✓ Draft 2 mit Betreuer-Feedback

---

## Arbeitspaket 5.2: Peer Review & Language Polish
**Dauer:** 4 Tage (18.9 - 21.9)
**Aufwand:** ~10 Stunden

- [ ] **Tag 1 (2h):** PDF an Peer schicken (Kollege/Freund)
- [ ] **Tag 2-3:** Warten auf Feedback (währenddessen: Grammarly/Language tool)
- [ ] **Tag 4 (4h):** Language corrections (Grammar, Style)
- [ ] **Tag 5 (4h):** Peer feedback einarbeiten

**Checkpoint 5.2:** ✓ Draft 3 language-polished

---

## Arbeitspaket 5.3: Final Review Cycle
**Dauer:** 3 Tage (22.9 - 24.9)
**Aufwand:** ~8 Stunden

- [ ] **Tag 1 (3h):** Final read-through (full thesis, fresh eyes)
- [ ] **Tag 2 (3h):** Check all references (BibLaTeX, Cross-refs)
- [ ] **Tag 3 (2h):** Betreuer final check (optional quick review)

**Checkpoint 5.3:** ✓ Draft 4 (near-final)

---

## Arbeitspaket 5.4: Formal Checks & Preparation
**Dauer:** 3 Tage (25.9 - 27.9)
**Aufwand:** ~8 Stunden

- [ ] **Tag 1 (3h):** Plagiatsprüfung durchführen
- [ ] **Tag 2 (3h):** Uni-Richtlinien checken (Formatierung, Seitenzahlen, etc.)
- [ ] **Tag 3 (2h):** Eidesstattliche Erklärung, TOC/LOF/LOT check

**Checkpoint 5.4:** ✓ Formale Anforderungen erfüllt

---

## Arbeitspaket 5.5: Final PDF Generation
**Dauer:** 2 Tage (28.9 - 29.9)
**Aufwand:** ~4 Stunden

- [ ] **Tag 1 (2h):** Final compilation (clean build)
- [ ] **Tag 2 (1h):** PDF quality check (fonts, figures, etc.)
- [ ] **Tag 3 (1h):** Backup erstellen (Cloud + lokale Kopien)

**Checkpoint 5.5:** ✓ Final PDF ready

---

## Arbeitspaket 5.6: ABGABE
**Datum:** 30. September
**Aufwand:** 2 Stunden

- [ ] Druckversion vorbereiten (falls erforderlich)
- [ ] Digitale Abgabe hochladen
- [ ] Bestätigung erhalten
- [ ] **THESIS ABGEGEBEN**

**Checkpoint 5.6:** ✓ ABGABE KOMPLETT

---

### MILESTONE 5 COMPLETE: MASTERARBEIT ABGEGEBEN
**Erreicht am:** _____ (Target: 30. September)
**Review-Fragen:**
- Wurde alles fristgerecht abgegeben?
- Sind alle formalen Anforderungen erfüllt?
- Post-mortem: Was lief gut? Was könnte beim nächsten Projekt besser laufen?

---

## Progress Dashboard

**Aktuelles Datum:** _____
**Tage bis Deadline:** _____
**Aktuelle Phase:** _____
**Nächster Checkpoint:** _____

### Milestone Completion

- [x] **Milestone 1:** Proposal v3 (erreicht 19.4.)
- [x] **Milestone 2:** Struktur + LaTeX-Skelett 9 Ch. + Appendix (erreicht 19.4.)
- [ ] **Milestone 3:** Experimente (Target: 12. Juli)
- [ ] **Milestone 4:** Writing Draft 1 (Target: 6. September)
- [ ] **Milestone 5:** Abgabe (Target: 30. September)

### Quantitative Metrics

| Metrik | Aktuell | Ziel | Status |
|--------|---------|------|--------|
| Papers gelesen (Vault) | 13 | 20 | ■■■■■ ■■■■■ ■■■☐☐ ☐☐☐☐☐ |
| Konfigurationen in 4×2-Matrix | 0 | 8 | ☐☐☐☐ ☐☐☐☐ |
| Kapitel geschrieben | 0 | 9 | ☐☐☐☐☐☐☐☐☐ |
| Appendix-Teile geschrieben | 0 | 4 | ☐☐☐☐ |
| LaTeX kompiliert (Skelett) | Ja | Ja | ✓ |
| Review-Zyklen | 0 | 3 | ☐☐☐ |

### Critical Path Status

| Arbeitspaket | Geplant | Tatsächlich | Δ (Tage) | Status |
|--------------|---------|-------------|----------|--------|
| AP 1.1: Lit Review 1 | 14.3 | | | ☐ |
| AP 1.2: Research Q | 21.3 | | | ☐ |
| AP 1.3: Proposal Draft | 4.4 | | | ☐ |
| AP 1.4: Proposal Final | 15.4 | | | ☐ |
| ... | | | | |

---

## Daily Work Log

**Datum:** _____
**Phase:** _____
**Arbeitspaket:** _____

**Geplant (heute):**
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

**Tatsächlich erledigt:**
- [ ] Task 1 - ✓/✗
- [ ] Task 2 - ✓/✗
- [ ] Task 3 - ✓/✗

**Tatsächlicher Aufwand:** _____ Stunden
**Notizen/Blockers:** _____
**Für morgen:** _____

---

## Weekly Review Template

**KW:** _____
**Datum:** _____

### Diese Woche geplant:
- [ ] Arbeitspaket X
- [ ] Arbeitspaket Y

### Tatsächlich erreicht:
- ✓/✗ Arbeitspaket X
- ✓/✗ Arbeitspaket Y

### Metrics:
- Papers gelesen diese Woche: _____
- Writing output (words): _____
- Code commits: _____
- Arbeitszeit total: _____ Stunden

### Reflections:
**Was lief gut:**
_____

**Was lief schlecht:**
_____

**Adjustments für nächste Woche:**
_____

### Nächste Woche geplant:
- [ ] Arbeitspaket Z
- [ ] ...

---

## Risk Management

| Risk | Wahrscheinlichkeit | Impact | Mitigation | Contingency |
|------|-------------------|--------|------------|-------------|
| Experimente länger als geplant | Mittel | Hoch | Buffer in Phase 3, früh anfangen | Scope reduzieren (weniger Ablations) |
| Betreuer-Feedback verspätet | Mittel | Mittel | Früh Draft schicken | Peer-Review als Ersatz |
| Motivation/Burnout | Mittel | Hoch | Daily small wins, Pausen | Notfall-Pause 2-3 Tage |
| Technische Probleme (GPU) | Niedrig | Hoch | Cloud-GPU Backup | Experimente auf Cloud migrieren |
| Scope Creep | Hoch | Mittel | Strict timeboxing | Nice-to-have → Future Work |

---

## ADHS-Specific Strategies (Integriert)

### Implemented in this Plan:

1. **External Structure:** Feste Deadlines für jedes Arbeitspaket
2. **Small Chunks:** Max. 2-3h Arbeitspakete
3. **Daily Concrete Tasks:** Keine vagen Ziele ("schreibe Kapitel"), sondern spezifisch ("schreibe Motivation section, 500 words")
4. **Visual Progress:** Checkboxen, Fortschrittsbalken
5. **Immediate Feedback:** Checkpoints nach jedem Arbeitspaket
6. **Momentum Maintenance:** Daily work log (verhindert mehrtägige Pausen)
7. **Time-boxing:** Feste Zeitlimits verhindern Perfektionismus-Rabbit-Holes
8. **Built-in Review:** Wöchentliche Reviews für Kurskorrektur
9. **Buffer Time:** Explizite Puffer (nicht versteckt)
10. **Low Activation Energy:** Klare "nächste Schritte" (kein Entscheidungs-Overhead am Morgen)

### Additional Recommendations:

- **Pomodoro Technique:** 25 min work, 5 min break (reduziert Overwhelm)
- **Body Doubling:** Mit Kollegen parallel arbeiten (virtuell/physisch)
- **Accountability Partner:** Wöchentliche Check-ins mit Betreuer/Peer
- **Environment Design:** Dedicated Workspace, Phone in anderen Raum
- **Reward System:** Nach Checkpoints bewusste Belohnungen (nicht XP, sondern real)

---

## Quality Gates (MUSS erfüllt sein für nächste Phase)

### Phase 1 → Phase 2:
- [ ] Proposal von Betreuer approved
- [ ] Mindestens 5 Papers dokumentiert
- [ ] Research Question finalisiert

### Phase 2 → Phase 3:
- [ ] Thesis-Outline von Betreuer approved
- [ ] 20 Papers gelesen
- [ ] LaTeX kompiliert

### Phase 3 → Phase 4:
- [ ] Alle Experimente durchgeführt
- [ ] Results statistisch signifikant (oder documented warum nicht)
- [ ] Alle Figures/Tables ready

### Phase 4 → Phase 5:
- [ ] Alle 9 Kapitel + Appendix A–D im Draft-Status
- [ ] Frontmatter vollständig (inkl. Abstract, Kurzfassung, LoA, LoS)
- [ ] Thesis kompiliert PDF
- [ ] Self-Review abgeschlossen

### Phase 5 → Abgabe:
- [ ] Betreuer approval
- [ ] Plagiatsprüfung passed
- [ ] Formale Anforderungen erfüllt

---

## Eskalationspfade

**Wenn Arbeitspaket > 20% überzogen:**
1. Analyse: Warum Verzögerung? (Scope, Skills, Motivation?)
2. Entscheidung: Re-scope oder Buffer nutzen?
3. Kommunikation: Betreuer informieren bei kritischen Verzögerungen

**Wenn Milestone gefährdet:**
1. **Plan B aktivieren:** Scope-Reduktion (siehe Risk Management)
2. **Hilfe holen:** Betreuer, Peers, Claude
3. **Zeitplan adjustieren:** Realistische Re-planung, nicht stur festhalten

**Wenn Burnout erkannt:**
1. **STOPP:** Sofortige Pause (1-2 Tage komplett frei)
2. **Ursachenanalyse:** Was triggert Überforderung?
3. **Adjustment:** Pace reduzieren, Expectations adjustieren
4. **Support:** Professionelle Hilfe (Beratung, Therapie) falls nötig

---

## Success Criteria

**Thesis gilt als erfolgreich wenn:**
1. Fristgerecht abgegeben (30. September)
2. Alle formalen Anforderungen erfüllt
3. Research Question beantwortet (auch wenn Ergebnis "negativ")
4. Mindestens 3 bedeutsame Experimente dokumentiert
5. Wissenschaftlich sound (Betreuer-Approval)

**Persönlicher Erfolg:**
- Lessons learned dokumentiert (für zukünftige Projekte)
- Ohne kompletten Burnout durchgekommen
- Stolz auf das Ergebnis (auch wenn nicht "perfekt")

---

**Letzte Aktualisierung:** 2026-04-20
**Aktuelle Phase:** Phase 3 — Benchmark-Repo (STC-Cell, Intermediate-Cell, Gradient-Flow-Tooling)
**Nächster Checkpoint:** STC-Cell implementiert (unblockiert seit Proposal v3; siehe [[benchmark-repo-roadmap]])
**Status:** On Track — Proposal v3 & Thesis-Skelett 4 Wochen vor Target; Schreib-Phase auf 9 Kapitel + Appendix erweitert

---

**HINWEIS:** Daten für Phase 3 (AP 3.1–3.5) und Phase 4/5 sind Richtwerte. Phase 3 hat flexible Binnenstruktur — konkrete Reihenfolge steuert [[benchmark-repo-roadmap]]. Phase 4 folgt der oben definierten Schreib-Reihenfolge (Architecture → Setup → Results → Background → Related Work → Intro → Discussion → Conclusion → Appendix/Frontmatter).

---

*Dieser Plan ist ein lebendes Dokument. Update wöchentlich im Weekly Review. Adjust bei Bedarf - Flexibilität ist Teil des Plans, nicht sein Feind.*
