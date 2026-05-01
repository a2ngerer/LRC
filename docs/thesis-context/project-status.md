# Projektstatus für Claude

**Stand:** 2026-04-26

## Aktuelle Phase
Proposal v3 (2026-04-19): Titel erweitert → "Architectural **and Model** Mechanisms"; CT-RNN in RQ1 durch GRU ersetzt (CT-RNN ist EEC-basiert/bio-inspired, nicht traditional); Scope & Delimitation umformuliert mit Rechtfertigung warum LSTM/GRU drin, Transformer/SSM/xLSTM draußen; LRC+NCP-Gradient-Flow-Beobachtung explizit als Motivation des Wiring-Axis aufgenommen (RQ4); Potential Extended Contribution "neue Wiring-Architektur L1–L6" ergänzt; Kurs 191.119 Autonomous Racing Cars zum Curriculum hinzugefügt. Code-Repo: Phase 2 abgeschlossen. Thesis-LaTeX-Skelett **restrukturiert** (2026-04-19) nach Spec: 9 Kapitel + 4-teiliger Appendix + erweitertes Frontmatter, kompiliert 34 Seiten (AP 2.1 ✓, AP 2.2 Skelett bereit). **2026-04-26:** Drei Deep-Research-Reports (ChatGPT, Gemini, Claude) synthetisiert in `Meta/Deep Research v1 compacted.md` mit Vault-Diff-Liste (30 Punkte) — wartet auf User-Review bevor Notizen aktualisiert werden.

## Was ist fertig
- Proposal Draft v1: 18 Seiten, abgesendet an Monika 2026-03-17
- Proposal v2 (2026-03-29): Scope erweitert basierend auf Feedback
- Proposal v3 (2026-04-19): Titel, CT-RNN-Re-Kategorisierung, Scope-Rechtfertigung, L1–L6 Extension, ARC-Kurs — siehe [[scope-decisions]]
- Scope final: 4×2 Matrix (LSTM, CT-RNN, STC, LRC × Dense, NCP) + Intermediate Models (CT-RNN + ε(w_i))
  - LSTM = gated Baseline (repräsentativ für LSTM/GRU-Familie)
  - CT-RNN = EEC-Baseline (simplest electrical-synapse EEC, auch Basis für RQ5)
  - STC = Saturated LTC (Farsang 2024) — **unblockiert** ✓
  - LRC = Liquid Resistance Liquid Capacitance (Farsang 2024)
  - Intermediate Models = CT-RNN + liquid elastance (neu, RQ5)
  - **Potential Extended Contribution (bedingt):** neue Wiring-Architektur L1–L6 (kortikal inspiriert)
- Benchmarks: Neural-ODE tasks + Robotics Control (REPPO als RL-Algorithmus)
- Code: LRC-Zellen nach `src/neurons/` migriert ✓; BaseCell-Interface implementiert ✓; `make_model` factory implementiert ✓
- Obsidian: 13 Papers, 9 Concepts (inkl. neue Stubs), 6 AI Tutoring Guides; Vault-bib 35 Einträge (merged)
- Claude-Setup (2026-04-17): Karpathy-LLM-Wiki-Muster eingeführt; projekt-lokale Slash-Commands `/lint-wiki`, `/ingest-url`, `/process-inbox`; Sync-Hooks aktiv (`.claude/hooks/sync-check.sh`)
- Thesis-LaTeX-Skelett (2026-04-19, initial): `Thesis/` auf Repo-Root angelegt, Stil aus Proposal abgeleitet (`report`-Klasse, identische Packages). 6 Kapitel-Files + Frontmatter + bib, kompiliert (13 Seiten Placeholder).
- **Thesis-LaTeX-Skelett Restrukturierung (2026-04-19):** Brainstorm mit superpowers-Skill → Design-Spec → 25-Task-Implementation-Plan → vollständig ausgeführt. Ergebnis: 9 Kapitel (Introduction/Background/Related Work/Architectures & Models/Experimental Setup/Results: Benchmarks/Results: Mechanistic/Discussion/Conclusion), erweitertes Frontmatter (TU-Wien-Titelseite mit Logo, Statutory Declaration, Kurzfassung+Abstract, Acknowledgements, ToC/LoF/LoT, List of Abbreviations, List of Symbols), 4-teiliger Appendix (A Extended Derivations, B Experimental Details, C Additional Results, D Reproducibility). Kompiliert 34 Seiten ohne Errors. Stand: [[latex-template]]. Spec: `docs/superpowers/specs/2026-04-19-thesis-chapter-structure-design.md`.

## Was als nächstes
- [ ] **`Meta/Deep Research v1 compacted.md` reviewen** und pro Diff-Eintrag (N1–N11, K1–K19) Aktion markieren (✅/❌/⏸); danach führt Claude die freigegebenen Vault-Updates aus
- [ ] Proposal v3 auf Overleaf teilen (Monika bat darum)
- [ ] STC Cell implementieren (Phase 2, jetzt unblockiert)
- [ ] Gradient Flow Analysis für LRC+NCP (RQ4) — neues Arbeitspaket in Phase 3
- [ ] Intermediate Model implementieren: CT-RNN + ε(w_i) (RQ5) — in Phase 3
- [ ] Phase 2: Structure & Deep Dive starten (Literatur-Vertiefung; Thesis-Outline ✓)
- [ ] Matrikelnummer in `Thesis/frontmatter/titlepage.tex` eintragen
- [ ] Thesis-Kapitel 01 (Introduction) aus Proposal-Inhalten befüllen — erste inhaltliche Arbeit — siehe [[latex-template]]
- [ ] Thesis-Kapitel 02 (Background) — EEC-Familie konsolidieren aus [[Concepts/index]]
- [ ] Paper-Stubs mit Inhalt füllen: `vaswani2017-attention`, `suzuki2025-grover`
- [ ] Concept-Stubs erweitern: `self-attention`, `neural-ode`, `policy-gradients`, `reinforcement-learning`

## Supervisors
- **Advisor:** Univ.Prof. Dipl.-Ing. Dr.rer.nat Radu Grosu
- **Assistance:** Monika Farsang (co-advisor, seit 2026-03-29 im Proposal eingetragen)

## Detail-Dateien
- Proposal-Struktur: [[structure]]
- Scope-Entscheidungen: [[scope-decisions]]
- Thesis-LaTeX-Stand: [[latex-template]]
- Code-Roadmap: [[benchmark-repo-roadmap]]
- Work-Log: [[work-documentation]]
- Projektplan: [[masterarbeit-projektplan]]
