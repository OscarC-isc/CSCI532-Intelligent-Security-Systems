# Non_PID — Conventional Federated Learning (no defense)

This folder is a standalone copy of the CIFAR-10 Flower FL app **without** the PID
anomaly-detection defense — plain `FedAvg` aggregation with no client filtering. It's the
"Conventional FL" baseline that the PID-enhanced results in `../CIFAR_FL_Flower/` and
`../results_1`–`../results_4` are compared against.

It shares the same model, data layout, and poisoning mechanism as `CIFAR_FL_Flower/`
(same `Net` CNN, same `CIFAR_clients/`-based dataset loading, same
`flipped_labels/<flip%>percent_flip/client_<id>/` poisoning convention), but the
aggregation strategy never inspects client models for anomalies or excludes any clients —
every client's update is always aggregated, even in the attack runs.

## Code

- **`task.py`** — same `Net` model, `CIFARDataset`, `load_data`, `train`/`test` as
  `CIFAR_FL_Flower/task.py`. `MetricsLogger` here only tracks per-client/average
  loss/accuracy (`client_losses.csv`, `client_accuracies.csv`, `avg_losses.csv`,
  `avg_accuracies.csv`) — there are no `pid_scores`/`exclusions` outputs since there's no
  PID logic to log. Output directory is always `results/attack/<flip%>flipped` (or
  `results/no_attack`), with no PID-parameter suffix.
- **`client_app.py`** — identical to `CIFAR_FL_Flower/client_app.py`: reads
  `poisoned-clients`/`flip-percent` from run config, loads clean or flipped labels
  accordingly, trains/evaluates locally, and returns weights + metrics.
- **`server_app.py`** — same as `CIFAR_FL_Flower/server_app.py` minus all PID
  configuration (`pid-enabled`, `kp`/`ki`/`kd`/`pid-threshold`) and the final PID exclusion
  summary (precision/recall/F1) — it just runs `CustomFedAvg` for `num-server-rounds` and
  saves the final model + metrics.
- **`custom_strategy.py`** — a much simpler `CustomFedAvg(FedAvg)` than the PID version:
  it logs per-client and average train/eval metrics each round and applies the same
  every-5-rounds learning-rate decay in `configure_train`, but `aggregate_train`/
  `aggregate_evaluate` pass every client straight through to the parent FedAvg
  implementation — no centroid distance, no PID score, no exclusion. `start()` re-implements
  the same round-by-round send/receive driver loop as the PID version, minus all
  exclusion-tracking state.
- **`data_prep.py`**, **`label_flipping.py`** — identical to the copies at the repo root;
  duplicated here so this folder can be run independently of `CIFAR_FL_Flower/`.
- **`Metrics_plotter.ipynb`** — notebook used to read the CSVs under `results/` and render
  the PNGs under each `results/.../plots/` folder.

## `results/`

```
results/
├── no_attack/                 # Clean run, no poisoned clients
│   ├── avg_accuracies.csv, avg_losses.csv
│   ├── client_accuracies.csv, client_losses.csv
│   └── plots/
│       ├── 1_average_metrics.png
│       ├── 2_client_performance.png
│       ├── 3_poisoned_vs_clean.png
│       ├── 4_final_round_heatmap.png
│       ├── 5_accuracy_distribution.png
│       └── 6_client_losses.png
└── attack/
    ├── 025flipped/
    ├── 050flipped/
    ├── 075flipped/
    └── 100flipped/            # each with avg/client loss & accuracy CSVs + plots/
```

Each CSV has the same schema as the corresponding file in `CIFAR_FL_Flower/`'s results
(`round, client_id, loss`/`accuracy` for per-client files; `round, avg_loss`/`avg_accuracy`
for the averages) — there just aren't `pid_scores.csv`/`exclusions.csv` files, since this
strategy never computes PID scores or excludes clients. These are the numbers plotted as
"Conventional FL" in the `comparison_plots/` charts under `../results_2`–`../results_4`.
