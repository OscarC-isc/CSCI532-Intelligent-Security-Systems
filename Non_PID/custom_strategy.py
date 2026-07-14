"""Custom FedAvg strategy with metrics logging and dynamic learning rate."""

import io
import time
import pickle
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

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
        self.save_path = None

        log(INFO, "  Poisoned Clients: %s", sorted(list(self.poisoned_clients)) if self.poisoned_clients else "None")

    def set_save_path(self, path: Path):
        """Set the path for saving checkpoints and results."""
        self.save_path = path
        log(INFO, "Save path set to: %s", self.save_path)

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords from training."""

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

                # Retrieve and log metadata
                config_record = reply.content.get("train_metadata")
                if config_record:
                    metadata_bytes = config_record["meta"]
                    train_meta = pickle.loads(metadata_bytes)
                    poison_marker = "⚠️ " if is_poisoned else "✓ "
                    log(INFO, f"{poison_marker}Round {server_round} - Client {partition_id}: "
                              f"train_loss={train_loss:.4f}, time={train_meta.get('training_time', 0):.2f}s")

        # Calculate and log average training metrics
        if train_losses:
            avg_train_loss = float(np.mean(train_losses))

            log(INFO, "─" * 70)
            log(INFO, f"Round {server_round} Training Summary:")
            log(INFO, f"  Avg Train Loss: {avg_train_loss:.4f}")
            log(INFO, f"  Clients trained: {len(train_losses)}")
            log(INFO, "─" * 70)

            if self.metrics_logger:
                self.metrics_logger.log_average_metrics(
                    round_num=server_round,
                    avg_loss=avg_train_loss,
                    avg_accuracy=0.0  # No accuracy during training
                )

        # Aggregate the ArrayRecords and MetricRecords as usual
        return super().aggregate_train(server_round, replies)

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
                poison_marker = "⚠️ " if is_poisoned else "✓ "
                log(INFO, f"{poison_marker}Client {client_id} - "
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

        log(INFO, "")
        log(INFO, "=" * 70)
        log(INFO, "FEDERATED LEARNING COMPLETE")
        log(INFO, "=" * 70)
        log(INFO, "Duration: %.2f seconds", time.time() - t_start)
        log(INFO, "=" * 70)
        log(INFO, "")

        # Save metrics to CSV
        if self.metrics_logger:
            self.metrics_logger.save_to_csv()
            log(INFO, "✅ Metrics saved to disk")

        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result