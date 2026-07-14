"""Custom FedAvg strategy with metrics logging and dynamic learning rate."""

import io
import time
import pickle
from collections import defaultdict
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional, List, Dict, Tuple

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from CIFAR_FL_Flower.task import MetricsLogger

PROJECT_NAME = "CIFAR-FL-Flower"


class CustomFedAvg(FedAvg):
    """Custom FedAvg with metrics logging and adaptive learning rate."""

    def __init__(
            self,
            *,
            fraction_train: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_train_nodes: int = 2,
            min_evaluate_nodes: int = 2,
            min_available_nodes: int = 2,
            metrics_logger: Optional[MetricsLogger] = None,
            poisoned_clients: Optional[set] = None,

            pid_enabled: bool = False,
            pid_threshold: float = 0.5,
            kp: float = 1.0,
            ki: float = 0.05,
            kd: float = 0.5,
            **kwargs
    ):
        """Initialize CustomFedAvg strategy.

        Args:
            fraction_train: Fraction of clients used for training
            fraction_evaluate: Fraction of clients used for evaluation
            min_train_nodes: Minimum number of clients for training
            min_evaluate_nodes: Minimum number of clients for evaluation
            min_available_nodes: Minimum number of available clients
            metrics_logger: MetricsLogger instance for collecting metrics
            poisoned_clients: Set of poisoned client IDs (1-indexed for client_1, client_2, etc.)
            initial_lr: Initial learning rate
            lr_decay_factor: Factor to multiply LR by when reducing (e.g., 0.5 = halve the LR)
            lr_patience: Number of rounds without improvement before reducing LR
            lr_min: Minimum learning rate threshold
            adaptive_lr: Whether to use adaptive LR based on accuracy plateau
        """
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            **kwargs
        )

        self.metrics_logger = metrics_logger
        self.poisoned_clients = poisoned_clients if poisoned_clients else set()

        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.pid_threshold = pid_threshold
        self.pid_enabled = pid_enabled

        # PID state tracking per client
        self.client_distances = defaultdict(list)  # History of distances
        self.client_integral = defaultdict(float)  # Cumulative error
        self.client_prev_distance = {}  # Previous distance for derivative
        self.client_pid_scores = {}  # Current PID scores

        self.excluded_clients = set()  # Clients excluded from aggregation
        self.exclusion_history = defaultdict(list)  # Track when clients were excluded

        log(INFO, "  Poisoned Clients: %s", sorted(list(self.poisoned_clients)) if self.poisoned_clients else "None")

    def set_save_path(self, path: Path):
        """Set the path for saving checkpoints and results."""
        self.save_path = path
        log(INFO, "Save path set to: %s", self.save_path)

    def _flatten_parameters(self, arrays: ArrayRecord) -> np.ndarray:
        """Flatten model parameters into a single vector."""
        state_dict = arrays.to_torch_state_dict()
        params = []
        for key in sorted(state_dict.keys()):
            params.append(state_dict[key].cpu().numpy().astype(np.float64).flatten())
        return np.concatenate(params)

    def _compute_centroid(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Compute the centroid (mean) of all client models."""
        if not vectors:
            return np.array([])
        stacked = np.vstack(vectors)
        return stacked.mean(axis=0)

    def _compute_l2_distance(self, vec: np.ndarray, centroid: np.ndarray) -> float:
        """Step 1: Calculate the normalized L2 distance (d) from the centroid to each client's model."""
        if len(vec) == 0 or len(centroid) == 0:
            return 0.0
        diff = vec - centroid
        l2 = float(np.linalg.norm(diff))
        centroid_norm = float(np.linalg.norm(centroid))
        # Normalized distance
        normalized_dist = l2 / (centroid_norm + 1e-12)
        return float(normalized_dist)

    def _compute_pid_score(
            self,
            client_id: int,
            distance: float,
            server_round: int
    ) -> float:
        """Calculate PID score following the exact pseudocode steps.

        Step 2: For round 1, PID = Kp * d (only proportional term)
        Step 3: For round > 1, PID = Kp*d + Ki*integral + Kd*derivative

        Where:
        - d = D(t) = current distance from centroid
        - integral = sum of all past distances
        - derivative = D(t) - D(t-1)
        """
        distance = float(distance)

        # Store distance in history
        self.client_distances[client_id].append(distance)

        # Step 2: Round 1 - only use proportional term (distance to centroid)
        if server_round == 1:
            pid = self.kp * distance
            # Initialize state for future rounds
            self.client_prev_distance[client_id] = distance
            self.client_integral[client_id] = distance
            self.client_pid_scores[client_id] = pid
            return float(pid)

        # Step 3: Round > 1 - use full PID formula
        # Proportional term: current distance
        p = self.kp * distance

        # Integral term: cumulative sum of distances (historical behavior)
        self.client_integral[client_id] += distance
        i = self.ki * self.client_integral[client_id]

        # Derivative term: change from previous round
        prev = float(self.client_prev_distance.get(client_id, distance))
        derivative = distance - prev
        d = self.kd * derivative

        # Full PID formula
        pid = p + i + d

        # Update state
        self.client_prev_distance[client_id] = distance
        self.client_pid_scores[client_id] = float(pid)

        return float(pid)

    def _filter_clients_by_pid(
            self,
            server_round: int,
            replies: List[Message]
    ) -> Tuple[List[Message], Dict[int, float], Dict[int, float]]:
        """Filter clients following the pseudocode:

        Server executes:
        - receive weights from clients
        - compute PID for each client
        - if client PID > PID threshold then exclude client from aggregation

        Step 4: Compare PID to threshold and discard clients with PID > threshold
        """
        if not self.pid_enabled or server_round == 0:
            return replies, {}, {}

        # Collect client models (not already excluded)
        client_vectors = []
        client_ids = []
        reply_map = {}

        for r in replies:
            if not r.has_content():
                continue

            arrays = r.content.get("arrays")
            metrics = r.content.get("metrics")
            if arrays is None or metrics is None:
                continue

            client_id = metrics.get("partition_id", 0)
            if client_id is None:
                continue

            # Skip if already permanently excluded
            if client_id in self.excluded_clients:
                continue

            # Flatten parameters
            vec = self._flatten_parameters(arrays)

            client_vectors.append(vec)
            client_ids.append(int(client_id))
            reply_map[int(client_id)] = r

        if len(client_vectors) < 1:
            log(INFO, "No active clients available for aggregation")
            return [], {}, {}

        # Step 1: Calculate centroid (mean of all client model parameters)
        centroid = self._compute_centroid(client_vectors)

        # Compute distances and PID scores for each client
        pid_scores = {}
        distances = {}

        for client_id, vec in zip(client_ids, client_vectors):
            # Step 1: Calculate normalized L2 distance from centroid
            dist = self._compute_l2_distance(vec, centroid)
            distances[client_id] = dist

            # Steps 2-3: Calculate PID score
            pid = self._compute_pid_score(client_id, dist, server_round)
            pid_scores[client_id] = pid

        # Step 4: Filter clients based on PID threshold
        # "if client PID > PID threshold then exclude client from aggregation"
        filtered_replies = []
        excluded_this_round = []

        for client_id in client_ids:
            pid = pid_scores.get(client_id, 0.0)
            dist = distances.get(client_id, 0.0)

            # Pseudocode: if PID > threshold, exclude from aggregation
            if pid > self.pid_threshold:
                # Exclude permanently (as per pseudocode: "discard from aggregation,
                # global model distribution, and further communication")
                self.excluded_clients.add(client_id)
                excluded_this_round.append(client_id)
                self.exclusion_history[server_round].append(client_id)
                log(INFO,
                    f"🚫 EXCLUDING client {client_id}: PID={pid:.6f} > threshold={self.pid_threshold:.6f} (dist={dist:.6e})")
            else:
                # Include in aggregation
                filtered_replies.append(reply_map[client_id])
                log(INFO,
                    f"✓ Including client {client_id}: PID={pid:.6f} ≤ threshold={self.pid_threshold:.6f} (dist={dist:.6e})")

        # Summary
        if excluded_this_round:
            log(INFO, f"Round {server_round}: Excluded {len(excluded_this_round)} clients: {excluded_this_round}")
        log(INFO, f"Total excluded so far: {len(self.excluded_clients)} clients: {sorted(list(self.excluded_clients))}")

        return filtered_replies, pid_scores, distances

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords from training."""

        replies_list = list(replies)

        filtered_replies, pid_scores, distances = self._filter_clients_by_pid(
            server_round, replies_list
        )
        log(INFO, f"aggregating {len(filtered_replies)}/{len(replies_list)} clients after filtering")

        # Collect training metrics from all clients
        train_losses = []
        partition_ids = []

        for reply in replies:
            if reply.has_content():
                # Extract metrics
                metrics = reply.content["metrics"]
                train_loss = metrics.get("train_loss", 0.0)
                partition_id = metrics.get("partition_id")
                is_poisoned = metrics.get("poisoned", 0)

                train_losses.append(train_loss)
                partition_ids.append(partition_id)

                pid_score = pid_scores.get(partition_id, None)
                dist = distances.get(partition_id, None)

                # Retrieve and log metadata
                config_record = reply.content.get("train_metadata")
                if config_record:
                    metadata_bytes = config_record["meta"]
                    train_meta = pickle.loads(metadata_bytes)
                    pid_info = f", PID={pid_score:.4f}, dist={dist:.6e}" if pid_score is not None else ""
                    log(INFO, f"Round {server_round} - Client {partition_id}: "
                              f"train_loss={train_loss:.4f}, time={train_meta.get('training_time', 0):.2f}s{pid_info}")

        # Calculate and log average training metrics
        if train_losses:
            avg_train_loss = float(np.mean(train_losses))

            log(INFO, "─" * 70)
            log(INFO, f"Round {server_round} Training Summary:")
            log(INFO, f"  Avg Train Loss: {avg_train_loss:.4f}")
            log(INFO, f"  Clients trained: {len(train_losses)}")
            log(INFO, f"  Excluded clients: {len(self.excluded_clients)}")
            log(INFO, "─" * 70)

            # Log PID scores and distances
            if hasattr(self.metrics_logger, 'log_pid_score'):
                for client_id in partition_ids:
                    if client_id in pid_scores and client_id in distances:
                        self.metrics_logger.log_pid_score(
                            round_num=server_round,
                            client_id=client_id,
                            pid_score=pid_scores[client_id],
                            distance=distances[client_id]
                        )

            if hasattr(self.metrics_logger, 'log_exclusion'):
                for client_id in partition_ids:
                    self.metrics_logger.log_exclusion(
                        round_num=server_round,
                        client_id=client_id,
                        excluded=(client_id in self.excluded_clients),
                        pid_score=pid_scores.get(client_id, 0.0)
                    )
                for client_id in self.excluded_clients:
                    self.metrics_logger.log_exclusion(
                        round_num=server_round,
                        client_id=client_id,
                        excluded=True,
                        pid_score=self.client_pid_scores.get(client_id, 0.0)
                    )

        # Aggregate the ArrayRecords and MetricRecords as usual
        return super().aggregate_train(server_round, filtered_replies)

    def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation metrics and log to MetricsLogger."""

        # Collect per-client metrics
        client_losses = []
        client_accuracies = []
        client_ids = []

        poisoned_losses = []
        clean_losses = []

        poisoned_accs = []
        clean_accs = []
        excluded_count = 0

        for reply in replies:
            if reply.has_content():
                metrics = reply.content["metrics"]

                eval_loss = metrics.get("eval_loss", 0.0)
                eval_acc = metrics.get("eval_acc", 0.0)
                client_id = metrics.get("partition_id")
                is_poisoned = metrics.get("poisoned", 0)

                client_losses.append(eval_loss)
                client_accuracies.append(eval_acc)
                client_ids.append(client_id)

                # Separate poisoned vs clean metrics
                if is_poisoned:
                    poisoned_losses.append(eval_loss)
                    poisoned_accs.append(eval_acc)
                else:
                    clean_losses.append(eval_loss)
                    clean_accs.append(eval_acc)

                # Log per-client metrics to MetricsLogger
                if self.metrics_logger:
                    self.metrics_logger.log_client_metrics(
                        round_num=server_round,
                        client_id=client_id,
                        loss=eval_loss,
                        accuracy=eval_acc
                    )

                # Console logging
                excluded_marker = " [EXCLUDED]" if is_poisoned else ""
                log(INFO, f"Client {client_id}{excluded_marker} - "
                          f"eval_loss={eval_loss:.4f}, eval_acc={eval_acc:.4f}")

        # Calculate and log average metrics
        if client_losses and client_accuracies:
            avg_loss = np.mean(client_losses)
            avg_accuracy = np.mean(client_accuracies)

            # Log to MetricsLogger
            if self.metrics_logger:
                self.metrics_logger.log_average_metrics(
                    round_num=server_round,
                    avg_loss=avg_loss,
                    avg_accuracy=avg_accuracy
                )

            # Print detailed round summary
            log(INFO, f"{'=' * 70}")
            log(INFO, f"ROUND {server_round} EVALUATION SUMMARY")
            log(INFO, f"{'=' * 70}")
            log(INFO, f"Overall Metrics:")
            log(INFO, f"  Average Loss:     {avg_loss:.4f}")
            log(INFO, f"  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
            log(INFO, f"  Clients evaluated: {len(replies)}")
            log(INFO, f"  Clients excluded: {len(self.excluded_clients)}")

            # Show breakdown if there are poisoned clients
            if poisoned_accs:
                avg_poisoned_acc = float(np.mean(poisoned_accs))
                avg_poisoned_loss = float(np.mean(poisoned_losses))
                log(INFO, f"")
                log(INFO, f"Poisoned Clients ({len(poisoned_accs)}):")
                log(INFO, f"  Avg Loss:     {avg_poisoned_loss:.4f}")
                log(INFO, f"  Avg Accuracy: {avg_poisoned_acc:.4f} ({avg_poisoned_acc*100:.2f}%)")

            if clean_accs:
                avg_clean_acc = float(np.mean(clean_accs))
                avg_clean_loss = float(np.mean(clean_losses))
                log(INFO, f"")
                log(INFO, f"Clean Clients ({len(clean_accs)}):")
                log(INFO, f"  Avg Loss:     {avg_clean_loss:.4f}")
                log(INFO, f"  Avg Accuracy: {avg_clean_acc:.4f} ({avg_clean_acc*100:.2f}%)")

        # Call parent's aggregate_evaluate
        return super().aggregate_evaluate(server_round, replies)

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training with optional LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            log(INFO, "LR decreased to: %f", config["lr"])

        # Pass the updated config to the parent class
        return super().configure_train(server_round, arrays, config, grid)


    def get_exclusion_summary(self) -> Dict:
        """Get summary of client exclusions."""
        summary = {
            "total_excluded": len(self.excluded_clients),
            "excluded_clients": sorted(list(self.excluded_clients)),
            "exclusion_by_round": dict(self.exclusion_history),
            "true_positives": len(self.excluded_clients & self.poisoned_clients),
            "false_positives": len(self.excluded_clients - self.poisoned_clients),
            "false_negatives": len(self.poisoned_clients - self.excluded_clients),
        }
        return summary

    def start(
            self,
            grid: Grid,
            initial_arrays: ArrayRecord,
            num_rounds: int = 3,
            timeout: float = 3600,
            train_config: Optional[ConfigRecord] = None,
            evaluate_config: Optional[ConfigRecord] = None,
            evaluate_fn: Optional[
                Callable[[int, ArrayRecord], Optional[MetricRecord]]
            ] = None,
    ) -> Result:
        """Execute the federated learning strategy with metrics logging."""

        self.excluded_clients = set()
        self.client_distances = defaultdict(list)
        self.client_integral = defaultdict(float)
        self.client_prev_distance = {}
        self.client_pid_scores = {}
        self.exclusion_history = defaultdict(list)

        log(INFO, "")
        log(INFO, "=" * 70)
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log(INFO, "=" * 70)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "=" * 70)
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config

        result = Result()

        t_start = time.time()

        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, f" ROUND {current_round}/{num_rounds}")

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated Train MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log evaluation metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated Eval MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation (if provided)
            if evaluate_fn:
                log(INFO, "Global server-side evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        exclusion_summary = self.get_exclusion_summary()

        log(INFO, "")
        log(INFO, "=" * 70)
        log(INFO, "FEDERATED LEARNING COMPLETE")
        log(INFO, "=" * 70)
        log(INFO, "Duration: %.2f seconds", time.time() - t_start)
        log(INFO, "=" * 70)
        log(INFO, "")
        log(INFO, "PID Exclusion Summary:")
        log(INFO, "  Total Excluded: %d", exclusion_summary["total_excluded"])
        log(INFO, "  Excluded Clients: %s", exclusion_summary["excluded_clients"])
        log(INFO, "  True Positives (Correctly Excluded Poisoned): %d", exclusion_summary["true_positives"])
        log(INFO, "  False Positives (Wrongly Excluded Clean): %d", exclusion_summary["false_positives"])
        log(INFO, "  False Negatives (Missed Poisoned): %d", exclusion_summary["false_negatives"])
        log(INFO, "")

        # Save metrics to CSV
        if self.metrics_logger:
            self.metrics_logger.save_to_csv()
            log(INFO, " Metrics saved to disk")

            if self.save_path:
                import json
                exclusion_path = self.save_path.parent / "exclusion_summary.json"
                with open(exclusion_path, 'w') as f:
                    json.dump(exclusion_summary, f, indent=2)
                log(INFO, "✅ Exclusion summary saved to: %s", exclusion_path)

            log(INFO, "✅ Metrics saved to disk")

        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result