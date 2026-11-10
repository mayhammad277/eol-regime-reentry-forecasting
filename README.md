# eol-regime-reentry-forecasting


Physics-informed, multi-task machine learning that reads the public TLE history of a low-Earth-orbit object and, causally at every new TLE, tells you:

1. **What regime it is in**: operational (manoeuvring), operational (normal station keeping), active end-of-life, natural decay, or unknown (abstain).
2. **Where it is going**: predicted perigee and apogee at 13 horizons from 0.25 to 90 days, and the probability it has re-entered by then.
3. **How long it has left**: remaining-lifetime median with a conformal 80% band.
4. **When and where it re-enters**: for objects heading for re-entry, a re-entry time, window and, only when the window is narrow enough to mean something, a position.

Two interchangeable models share one pipeline: a **LightGBM** multi-task model and a small **transformer** over a 10-step feature history. Both are trained on simulated trajectories and evaluated on a held-out solar cycle and on a few real objects.

> **Status.** The LightGBM system has been run on real objects. The transformer has been run end to end only on a small simulated smoke dataset, which shows the code works but is not evidence of real-world performance. See [Results and evidence](#results-and-evidence) and [Limitations](#limitations).

## How it works

```
TLEs + space weather + Earth orientation
        |  SGP4 mean elements, cleaning
        v
Physics layer: NRLMSIS 2.0 drag, drag integral C(t), fitted ballistic coefficient B_W (windows 2, 7, 30, 90 d)
        |  causal features (identical for training, back-testing and live use)
        v
Multi-task model  --  class (5)  |  perigee and gap path  |  P(re-entered by h)  |  remaining life quantiles
        |  smoothing, open-set UNKNOWN, 365-day gate at high altitude
        v
Re-entry chain: learned states -> hand-off -> drag physics to 100 km -> time, window, position
```

The key physical signal is the **fitted ballistic coefficient**: roughly constant and physical for passive decay, about zero or negative for station keeping, and far above any physical value for propulsive descent. Features that would let a simulator-trained model cheat (eccentricity, apogee, long-window TLE counts, TLE B*) are excluded on purpose.

| | LightGBM (`multitask.py`) | Transformer (`transformer_mt.py`) |
|---|---|---|
| History | Windowed summary features | The same features at 10 time offsets (0 to 14 days back) |
| Heads | Independent, horizon is an input | Shared encoder, one output per horizon |
| Uncertainty | Bagging and conformal bands | Ensemble spread and conformal bands |
| Extra decision rules | Open-set override | Softer OOD rule and a 4-epoch ACTIVE_EOL rule |
| Dependency | LightGBM | PyTorch (optional) |

Re-entry time uses one of four methods for the final descent: `physics` (no learned path), `measured` (learned states, B fitted from real TLEs), `learned` (B backed out of the learned path, found to be late-biased) and `auto` (physics when 30 days or less remain, otherwise measured; the default).

## Repository layout

```
src/        all Python modules and scripts (run from the repo root)
data/       raw inputs you provide, not committed (see below)
outputs/    generated pickles, models, figures and reports, not committed
```

| File | Role |
|---|---|
| `data_io.py`, `physics.py`, `sensitivity.py`, `features.py` | TLE/space-weather loading, drag physics, drag integral, causal features |
| `synth.py`, `run_synth.py` | Simulator and synthetic training set (patched versions also record the 5-class label) |
| `real_data.py`, `gate.py` | Real-object features and labels, 365-day gate |
| `multitask.py`, `transformer_mt.py` | The two multi-task models, targets and 5-class labels |
| `train_multitask.py`, `train_transformer.py`, `eval_mt.py` | Training and evaluation (same data, splits and metrics) |
| `reentry_mt.py` | Predicted states to re-entry time, interval and position |
| `live_engine_mt.py`, `live_engine_tf.py` | Causal live engines for the two models |
| `predict_mt.py`, `report_mt.py`, `sim_html_mt.py`, `live_dashboard_mt.py` | Single prediction, static report, playable simulation page, Dash dashboard |
| `compare_modes.py`, `viz_mt.py` | Comparison of the four re-entry methods, figures |

Older single-task scripts (`train_eval.py`, `models.py`, `live_engine.py`, `live_sim.py`, `live_dashboard.py`) are kept for reference and comparison.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch            # only if you want the transformer
```

Used with Python 3.11 and 3.12. `pymsis` provides NRLMSIS 2.0; `sgp4` parses the TLEs.

## Data you need to provide

Raw data is not redistributed here. Check each source's terms before sharing any of it.

| File | Source |
|---|---|
| `data/sw.txt` | CelesTrak space weather, SW-All (CSSI format) |
| `data/eop.txt` | CelesTrak Earth orientation parameters, EOP-All |
| `data/tles_<norad>.txt` | TLE history of each object, for example from Space-Track (2-line or 3-line files both work) |
| `data/object_properties.csv` | Columns `norad, area_min_m2, area_avg_m2, area_max_m2, mass_kg`, for example from ESA DISCOS |

## Quick start

All commands run from the repo root.

```bash
# 1. parse raw text into pickles, then build cleaned tracks and real-object features
python src/parse_raw_texts.py
python src/real_data.py

# 2. simulate the training set (slow), using the patched simulator
python src/run_synth.py 2000 7
python src/run_synth.py 2000 8

# 3. train and evaluate
python src/train_multitask.py 7 8 --pipeline 150                     # LightGBM
python src/train_transformer.py 7 8 --models 5 --epochs 40           # transformer

# 4. use it
python src/predict_mt.py data/tles_44735.txt                         # prediction at the newest TLE
python src/predict_mt.py data/tles_44735.txt --at 2025-08-20 --model transformer
python src/report_mt.py data/tles_44735.txt --start 2025-07-01 --end 2025-09-19
python src/sim_html_mt.py data/tles_*.txt                            # playable page: any object, any epoch
python src/live_dashboard_mt.py [--model transformer]                # live dashboard, http://127.0.0.1:8050
```

`parse_raw_texts.py` and `real_data.py` expect `outputs/clean_<norad>.pkl` track files, which `live_engine.prepare_track` creates the first time it sees a TLE file.

## Results and evidence

**LightGBM on real objects.** Median absolute relative error of the re-entry time (signed bias in brackets; positive means too late), 80 cases per set:

| Set | Lifetime head | Learned B | Measured B | Physics only | Auto |
|---|---|---|---|---|---|
| Real objects, overall | 0.22 | 0.30 | 0.11 | 0.14 | 0.11 |
| Synthetic test, overall | 0.20 | 0.33 | 0.20 | 0.19 | 0.20 |

Learned-B was late in every bin below 150 days. Measured-B and physics-only are within noise of each other, and the auto-mode threshold was chosen on these same cases, so it still needs validation on fresh ones.


## Limitations

- Only four real objects are used for evaluation, and the real ACTIVE_EOL class is small. Real-object statistics are indicative.
- Real labels combine catalogue events with physics diagnostics and judgement.
- UNKNOWN is both a trained class and a runtime abstention, so its metrics mix the two.
- Position is rarely well constrained and is never reported for ACTIVE_EOL; those re-entry times are indicative only.

## References

- Emmert, J. T., et al. (2021). NRLMSIS 2.0: A whole-atmosphere empirical model of temperature and neutral species densities. *Earth and Space Science*, 8, e2020EA001321.
- Vallado, D. A., Crawford, P., Hujsak, R., Kelso, T. S. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753.
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems 30*.
- CelesTrak space weather and Earth orientation data; ESA DISCOS.
