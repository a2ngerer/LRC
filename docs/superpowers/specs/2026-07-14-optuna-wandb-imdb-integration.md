# Plan: Optuna + wandb + IMDB-Task ins Benchmark-Projekt integrieren

**Datum:** 2026-07-14
**Kontext:** Meeting Farsang 2026-07-02 — der "positive Kern" der Thesis (faire, parameter-matched Benchmark-Matrix mit HPO und Tracking) steht noch aus. Die tbt_cNCP-Experimente lieferten nur den Negativ-Ausblick.
**Ziel:** Drei fehlende Bausteine integrieren, damit die Hauptaussage der Thesis reproduzierbar und methodisch abgesichert entsteht: (1) wandb-Tracking, (2) Optuna-HPO, (3) IMDB-Sentiment-Task als dritte Task-Familie.

## Status

- **Entscheidungen 2026-07-14:** Param-Matching **budget-abgeleitet** (Variante a — echt parameter-matched); Umsetzungsreihenfolge **AP1 → AP3 → AP2**.
- **AP1 wandb: ERLEDIGT & lokal verifiziert (2026-07-14).** Modul `src/benchmark/tracking.py` (no-op ohne wandb), `--wandb`-Flag in beiden Keras-Runnern, `log_fn`-Hook in `neural_ode.trainer.train` + `run_one`, Cluster offline-Modus (`PA_WANDB`/`LV_WANDB`) + `cluster/wandb_sync.sh`. Verifiziert: 36 Tests grün (inkl. `test_equivalence`, byte-identity), wandb-Offline-Smoke auf lotka_volterra erzeugt einen `offline-run`. **Offen (User):** wandb Academic Plan beantragen + `uv run wandb login`, dann Cluster-Lauf mit `PA_WANDB=1` real testen.
- **AP2 Optuna: IMPLEMENTIERT & LAUFEND (2026-07-16).** Runner `experiments/run_lotka_volterra_hpo.py` (eine Optuna-study je (cell,wiring,seed), JournalStorage, TPE + MedianPruner, TFKerasPruningCallback, jeder Trial ein wandb-Run), Param-Matching via `size_for_param_budget` (deterministische Breite je Wiring auf festes Param-Budget — Variante a umgesetzt), sauberer train/val/test-Split (val==test-Leakage behoben), Aggregator `experiments/aggregate_hpo.py` (mean±std je (cell,wiring) über Seeds). Extra `hpo` (optuna 4.9 + optuna-integration) in `pyproject.toml`. Cluster: `cluster/lotka_volterra_hpo.sbatch` + `submit_/fetch_lotka_volterra_hpo.sh`; lokaler Fallback `cluster/run_lotka_volterra_hpo_local.sh`.
  - **Cluster war unerreichbar** (datalab-Login-Node Port 22 timeout trotz aktivem VPN — serverseitig/Wartung, nicht behebbar). Da der User schläft und Ergebnisse bis früh vorliegen sollen, läuft der Benchmark **lokal** (M1/M2, wandb offline). Cluster-Skripte sind submit-ready für sobald der Node zurück ist.
  - **Laufender Matrix-Umfang** (2026-07-16 ~04:33 UTC gestartet): {cfc_lrc, cfc, gru} × {dense, ncp, cncp} × seeds {0,1,2} = 27 Studies, 30 Trials × 120 Epochen, Param-Budget 4000 (alle Arme <8% Abweichung). `ltc` bewusst raus (~50x langsamer, iterativer ODE-Solver → würde 15h sprengen).
- **AP3 IMDB:** noch offen (dritte Task-Familie).

---

## Ist-Zustand (verifiziert 2026-07-14)

- **Tasks vorhanden:** `lotka_volterra`, `mujoco_rl`, `person_activity`, plus Explorationstasks (`active_sensing`, `committee`, `neural_ode`). **Kein IMDB/Text-Task.**
- **Zwei Trainings-Backends:**
  - Custom Loop: `src/tasks/neural_ode/trainer.py:42-112` (`train(...) -> list[float]`, Loss-Append `:107`), aufgerufen aus `experiments/run_benchmark.py:604` (`run_one`), Trainings-Aufruf `:619-647`.
  - Keras `.fit()`: `run_person_activity_benchmark.py:73-80` und `run_lotka_volterra_benchmark.py:116-120` — beide **ohne** `callbacks=`.
- **Config-/Kampagnen-System:** `src/benchmark/{registry,expand,config}.py`. `WIRINGS = ('dense','ncp','cncp')` (`registry.py:36`), `CELL_REGISTRY` mit 38 Zellen (`:74-114`), Accessor `cell_units`/`cell_ncp`/`cell_kwargs` (`:120-136`). YAML-Kampagnen unter `configs/campaigns/*.yaml`, Expansion via `expand.py:42`.
- **Param-Matching:** keine dedizierte Funktion. Aktuell manuelle `CELL_UNITS`-Overrides (z.B. `lrc_pm: 24`). Fairness-Test nur grob: `test_person_activity.py:63-69` prüft cncp/ncp-Param-Ratio ∈ [1/1.5, 1.5].
- **Dependencies:** `pyproject.toml`, Package-Manager **uv**, TF `>=2.15,<2.16` gepinnt, Extras `metal`/`cuda`/`rl`. **Optuna und wandb fehlen.**
- **Tests:** pro Task ein Ordner unter `tests/`, Lauf via `uv run pytest tests/`. Kritisch: `tests/benchmark/test_equivalence.py` (Registry↔Legacy byte-identisch — nicht brechen).

---

## Arbeitspaket 1 — wandb-Tracking (klein, Deadline-relevant)

**Warum zuerst:** kleinster Schnitt, unabhängig von den anderen zwei, liefert sofort die von Farsang geforderten Learning-Curves + GPU-Util.

**Schritte:**
1. `uv add --optional tracking wandb` → neues Extra `tracking` in `pyproject.toml`.
2. **Keras-Backend:** in `run_person_activity_benchmark.py:73` und `run_lotka_volterra_benchmark.py:116` `callbacks=[WandbMetricsLogger()]` ergänzen, davor `wandb.init(...)`, danach `wandb.finish()`. Import: `from wandb.integration.keras import WandbMetricsLogger` (nicht das alte `wandb.keras`).
3. **Custom-Loop-Backend:** `trainer.py` **task-agnostisch** halten — kein direkter wandb-Import. Stattdessen optionalen `log_fn`-Parameter zu `train(...)` hinzufügen (Default `None`), der pro Iteration `log_fn(step, loss_val)` aufruft. Der wandb-Aufruf lebt im Wrapper `run_benchmark.py:641`, der `log_fn=lambda i, l: wandb.log(...)` reingibt. (Ein `log_fn`-Hook existiert bereits im MuJoCo-Runner — Muster übernehmen.)
4. **Cluster-Offline-Modus:** `WANDB_MODE=offline` in den `.sbatch`-Skripten setzen (Compute-Nodes ohne Internet). `wandb sync` als Schritt in die `fetch_*.sh`-Skripte bzw. ein separates Sync-Skript am Login-Node.
5. **Konventionen:** `project="thesis-benchmarks"`, `group=<cell>`, `job_type=<wiring>`, `config={seed, units, lr, param_count, task}`. Seed IMMER in `config`, nie in Tags (Tags gruppieren nicht).

**User-Aktion (parallel, blockiert nichts):** wandb Academic Plan über `@tuwien.ac.at`-Mail beantragen (gratis, unbegrenzte Tracked Hours).

**Verifikation:** ein lokaler Kurzlauf pro Backend, der einen Offline-Run erzeugt; danach `wandb sync` und Sichtprüfung im Dashboard (Loss-Kurve + GPU-Util sichtbar).

---

## Arbeitspaket 2 — Optuna-HPO (methodisch zentral)

**Warum danach:** nutzt das wandb-Logging (jeder Trial = ein Run) und braucht alle Task-Runner als objective-Targets.

**Schritte:**
1. `uv add --optional hpo "optuna" "optuna-integration"` → Extra `hpo`.
2. **objective-Funktion** um den bestehenden Trainings-Kern: sampled Hyperparameter → baut Modell → trainiert → gibt Validierungs-Metrik zurück. Für ODE-Tasks um `run_one` (`run_benchmark.py:604`), für Keras-Tasks um den jeweiligen `run(args)`.
3. **Param-Matching (Kernentscheidung, siehe unten):** neue Funktion `units_for_param_budget(cell, wiring, target_params)` in `src/benchmark/`. Optuna tunt die "weichen" Hyperparameter (lr, batch_size, ode_unfolds, dropout), **units werden deterministisch aus einem festen Param-Budget je Zelle abgeleitet** — so bleibt der Vergleich parameter-matched, wie Farsang gefordert hat.
4. **Fairness:** ein `study` pro `(cell, wiring, task)`, `n_trials` fix (NICHT `timeout` — LTC ~9x langsamer würde benachteiligt), gleicher Sampler-Seed (TPE), `MedianPruner`.
5. **Verteiltes Storage:** Optuna `JournalStorage` (`JournalFileBackend`) auf NFS, damit SLURM-Array-Worker dieselbe study mit `load_if_exists=True` bearbeiten. Kein DB-Server nötig.
6. **val/test-Split-Fix:** `person_activity` hatte `val == test` → HPO-Leakage-Risiko. Vor dem Tuning sauberen train/val/test-Split einziehen. Für IMDB von Anfang an sauber anlegen.

**Verifikation:** eine kleine study (`n_trials=5`) auf einem Task lokal; prüfen, dass `study.best_params` plausibel ist und die Param-Counts über Zellen im Toleranzband bleiben.

---

## Arbeitspaket 3 — IMDB-Sentiment-Task (größter inhaltlicher Hebel)

**Warum wertvoll:** IMDB hat `seq_len≈256` → echtes Sequenz-Rollout. Nur da trainiert die cNCP-Rekurrenz sauber (im Gegensatz zu ODE-T=1-Tasks, wo die delayed-state-Kanten untrainiert bleiben). Dritte Task-Familie aus Farsangs `MoniFarsang/LRC/classification`.

**Schritte:**
1. **`src/tasks/imdb/`** analog `person_activity/`:
   - `datasets.py`: `load_imdb(vocab_size=20000, seq_len=256, seed=None) -> IMDBData(train_x, train_t, train_y, val_x, ..., test_x, ...)` via `keras.datasets.imdb`. Referenz für Preprocessing: Legacy `classification/run_imdb.py:68-100` + Farsangs Repo. Sauberer train/val/test-Split von Anfang an.
   - `model.py`: `build_imdb_model(wiring, cell, size=64, seed=42, vocab_size=20000, seq_len=256, lr=1e-3, **cell_kwargs)` — `Embedding → RNN(cell, wiring) → Dense(2)`, Wirings dense/NCP/cNCP wie person_activity.
   - `__init__.py`: Exporte.
2. **Runner** `experiments/run_imdb_benchmark.py` (copy+adapt von `run_person_activity_benchmark.py`), inkl. wandb-Callback aus AP1.
3. **Tests** `tests/tasks/imdb/test_imdb.py` (copy+adapt von `test_person_activity.py`): Forward-Shapes (3 Wirings × 2 Zellen), Gradient-Flow, Param-Fairness-Ratio.
4. **Cluster-Skripte** `cluster/{imdb.sbatch, submit_imdb.sh, fetch_imdb.sh}` nach person_activity-Muster.
5. **Kampagnen-Config** optional `configs/campaigns/imdb.yaml` für Batch-Runs.
6. Registry-Update **nicht** nötig (IMDB ist kein ODE-System, läuft über den Keras-Task-Pfad).

**Verifikation:** `uv run pytest tests/tasks/imdb/` grün; ein 1-Epoch-Smoke-Lauf pro Wiring lokal.

---

## Reihenfolge & Abhängigkeiten

```
AP1 wandb  ──┐  (unabhängig, klein, Deadline)
AP3 IMDB   ──┼──►  AP2 Optuna  (braucht Tasks als objective + nutzt wandb-Logging)
             │
   (AP1 und AP3 können parallel laufen)
```

Empfohlen: **AP1 → AP3 → AP2.** wandb zuerst grün machen (schnell, Deadline), dann IMDB als dritte Task-Familie, zuletzt Optuna über alle drei Familien legen.

---

## Offene Design-Entscheidungen (vor Ausführung klären)

1. **Param-Matching-Strategie** — die wichtigste:
   - **(a) [empfohlen]** Festes Param-Budget je Task (z.B. ~45k), `units` deterministisch daraus abgeleitet; Optuna tunt nur weiche Hyperparameter. → echt parameter-matched, deckt Farsangs Vorgabe exakt.
   - (b) `units` als Optuna-Suchdim frei; Param-Count nur als Kovariate reporten. → einfacher, aber Vergleich driftet, widerspricht "parameter-matched".
2. **Optuna study-Granularität:** ein study pro `(cell, wiring, task)` [empfohlen — jede Zelle findet ihr Optimum] vs. ein study pro Architektur.
3. **wandb-Kopplung von `trainer.py`:** via optionalem `log_fn`-Callback statt direktem Import [empfohlen — hält den Trainer task-agnostisch].
4. **val/test-Split-Fix** für person_activity vor dem HPO-Lauf (Leakage). Muss vor AP2 erledigt sein.

---

## Deliverables (auch fürs Meeting)

- Drei integrierte Bausteine, Tests grün, je ein Smoke-Lauf verifiziert.
- wandb-Dashboard mit ersten Learning-Curves (zeigbar für Farsang).
- Code-Stand für Farsang (Punkt 11: wie MM_lrc/cfc_lrc ausgewertet wurde) referenzierbar.
