# CSCI532-Intelligent-Security-Systems
## Federated Learning Privacy and Security Enhancement (Group 1)
A semester long project on Federated Learning and applying a PID defense against label flipping attacks on a CIFAR dataset.

This repo implements a Conventional Federated Learning (FL) setup on CIFAR-10 using a
Flower/PyTorch app, a label-flipping data-poisoning attack, and a PID-based defense
strategy that detects and excludes malicious clients during aggregation. It contains two
parallel implementations of the FL server/client that share the same underlying model and
data pipeline:

- **`CIFAR_FL_Flower/`** — the PID-enhanced strategy (`CustomFedAvg` with anomaly
  detection and client exclusion). This is the code used to produce `results_1`–`results_4`.
- **`Non_PID/`** — the Conventional FL baseline (plain FedAvg, no defense, no client
  exclusion). This is a separate, standalone copy of the app used to produce the
  "no defense" numbers that the PID results are compared against. See
  [`Non_PID/README.md`](Non_PID/README.md) for details.

Group 1 attack configuration: **20 total clients, 2 poisoned clients (client_3 and
client_17)**, evaluated at 0%, 10%, 25%, 50%, 75%, 100% label-flipping severity.

## Repository layout

```
data_prep.py            # Downloads CIFAR-10 and splits it IID across 20 clients
label_flipping.py        # Generates poisoned label sets for the poisoned clients
CIFAR_FL_Flower/          # Flower ServerApp/ClientApp implementing FL + PID defense
Non_PID/                  # Standalone Conventional FL (no defense) baseline app + results
results_1 .. results_4/   # Metrics/plots from four PID experiment runs (see below)
```

## Code: `CIFAR_FL_Flower/` (PID-enhanced FL)

A Flower (`flwr`) app implementing federated CIFAR-10 image classification, with built-in
support for simulating data-poisoning attacks and an experimental PID-based anomaly
detector for excluding poisoned clients.

- **`task.py`**
  - `Net`: a small CNN (2 conv blocks → FC) used as the global/local model for CIFAR-10
    (10-class, 32x32 RGB images).
  - `CIFARDataset`: loads a single client's images/labels from disk. If the client is
    marked poisoned, it loads labels from `flipped_labels/<flip%>percent_flip/client_<id>/labels`
    instead of the clean `CIFAR_clients/client_<id>/labels`.
  - `load_data`: builds train/val `DataLoader`s for a given client (80/20 split).
  - `train` / `test`: standard local SGD training loop and evaluation loop (loss + accuracy).
  - `MetricsLogger`: collects per-round metrics (per-client loss/accuracy, averages, PID
    scores, exclusion decisions) and writes them to CSV under `results/...` (see below for
    the folder convention).

- **`client_app.py`**
  - Flower `ClientApp` with `@app.train()` and `@app.evaluate()` handlers.
  - Reads `poisoned-clients` and `flip-percent` from the run config to decide whether the
    current partition is poisoned, then loads the corresponding (clean or flipped) dataset
    and trains/evaluates the local model, returning weights + metrics to the server.

- **`server_app.py`**
  - Flower `ServerApp` entry point (`@app.main()`). Reads run configuration (rounds,
    learning rate, poisoned client list, flip percentage, PID hyperparameters `kp`/`ki`/`kd`/
    `pid-threshold`), initializes the global model, and runs `CustomFedAvg` for
    `num-server-rounds` rounds.
  - After training, saves the final model checkpoint, dumps all metrics to CSV via
    `MetricsLogger`, and prints a PID exclusion summary (true/false positives/negatives,
    precision/recall/F1 for malicious-client detection).

- **`custom_strategy.py`**
  - `CustomFedAvg(FedAvg)`: the core of the project's "enhanced FL" implementation.
    - Flattens each client's returned model weights into a vector and computes the
      **centroid** (mean) of all client vectors each round.
    - Computes a normalized L2 **distance** of each client's model from the centroid, then
      combines it into a **PID score**:
      `u(t) = Kp*D(t) + Ki*sum(D(i)) + Kd*(D(t)-D(t-1))` (round 1 uses only the
      proportional term).
    - Clients whose PID score exceeds `pid_threshold` are **permanently excluded** from
      aggregation, global-model distribution, and future rounds.
    - `aggregate_train`/`aggregate_evaluate` layer this filtering + logging on top of
      standard FedAvg aggregation; `configure_train` also halves the learning rate every 5
      rounds.
    - `start()` re-implements the Flower strategy driver loop (send/receive train and
      evaluate messages each round) so that PID state, exclusions, and metrics are tracked
      and persisted across the whole run, and prints a final exclusion summary.
  - When `pid_enabled=False` this strategy behaves as plain FedAvg — this is how the
    "Conventional FL" (no defense) experiments are produced with the same codebase.

- **`__init__.py`**: package marker only.

### Supporting scripts (repo root)

- **`data_prep.py`**: downloads CIFAR-10 and partitions it IID into 20 client folders
  (`CIFAR_clients/client_<1..20>/{images,labels}`), one image/label pair per file.
- **`label_flipping.py`**: for the poisoned clients (`client_3`, `client_17`), generates
  randomly-flipped label sets at 10/25/50/75/100% flip ratios and writes them to
  `flipped_labels/<flip%>percent_flip/client_<id>/labels/`, which `CIFARDataset` loads
  instead of the clean labels when a client is poisoned.

## Results folders

Each `results_N/` directory holds the metrics and plots from one experiment run (the
project was re-run several times while tuning the PID hyperparameters/threshold, or to
regenerate comparison plots — folder contents are otherwise identical in structure).

```
results_N/
├── attack/
│   ├── 025flipped_kp5.0_ki0.5_kd1.0_th2.5/
│   ├── 050flipped_kp5.0_ki0.5_kd1.0_th2.5/
│   ├── 075flipped_kp5.0_ki0.5_kd1.0_th2.5/
│   └── 100flipped_kp5.0_ki0.5_kd1.0_th2.5/
└── comparison_plots/            # results_2, _3, _4 only
```

Each `attack/<flip%>flipped_kp<Kp>_ki<Ki>_kd<Kd>_th<threshold>/` subfolder corresponds to
one run of the PID-enhanced FL strategy under label-flipping attack at that severity, with
the given PID coefficients and exclusion threshold baked into the folder name (this is the
`output_dir` naming produced by `MetricsLogger` in `task.py`). Each contains:

- `client_losses.csv`, `client_accuracies.csv` — per-client, per-round eval loss/accuracy
  (columns: `round, client_id, loss`/`accuracy`).
- `avg_losses.csv`, `avg_accuracies.csv` — average loss/accuracy across clients per round
  (columns: `round, avg_loss`/`avg_accuracy`).
- `pid_scores.csv` — each client's computed PID score and centroid distance per round
  (columns: `round, client_id, pid_score, distance`).
- `exclusions.csv` — whether each client was excluded from aggregation each round
  (columns: `round, client_id, excluded, pid_score`) — this is the client-removal-dynamics
  data.
- `plots/` (present in `results_1`) — rendered PNGs of the above: loss/accuracy per client,
  average loss/accuracy, loss-and-accuracy-vs-attack-severity, and client removal dynamics.

`comparison_plots/` (in `results_2`–`results_4`) holds the "Conventional FL vs. FL with
PID" comparison charts:
- `1_loss_comparison.png` — average loss over rounds, Conventional FL vs. PID-defended FL.
- `2_accuracy_comparison.png` — average accuracy over rounds, Conventional FL vs. PID-defended FL.
- `3_attack_severity_comparison.png` — final loss/accuracy as a function of flip percentage,
  for both strategies.
- `4_improvement_analysis.png` — the accuracy/loss improvement PID provides over
  Conventional FL at each attack severity.

`comparison_plots.zip` is a zipped copy of the same `comparison_plots/` folder, used for
uploading the plots as a single submission artifact.

Note: only the 25/50/75/100% flip levels are present under `attack/`; the 0% (no-attack)
and 10%-flip runs use the `no_attack` output path produced by `MetricsLogger` when
`poisoned=False` and are not included in these results folders.

## `Non_PID/` — Conventional FL baseline

`Non_PID/` is a self-contained copy of the Flower app with the PID defense stripped out
(plain `FedAvg`, no anomaly scoring, no client exclusion) plus its own `results/` folder
(no-attack and 25/50/75/100%-flip runs) and a `Metrics_plotter.ipynb` notebook used to
generate its plots. It's kept separate from `CIFAR_FL_Flower/` so the two strategies can be
run independently and their outputs compared directly. See
[`Non_PID/README.md`](Non_PID/README.md) for the full breakdown.
