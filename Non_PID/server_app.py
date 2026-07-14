"""CIFAR_FL_Flower.server_app: Flower Server App for CIFAR-10."""

import torch
from pathlib import Path
from flwr.serverapp import Grid, ServerApp
from flwr.app import ArrayRecord, ConfigRecord, Context

from CIFAR_FL_Flower.task import Net, MetricsLogger
from CIFAR_FL_Flower.custom_strategy import CustomFedAvg


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for CIFAR-10 ServerApp with custom strategy."""

    # Get configuration from pyproject.toml
    fraction_train = float(context.run_config.get("fraction-train", 1.0))
    num_rounds = int(context.run_config.get("num-server-rounds", 5))
    initial_lr = float(context.run_config.get("lr", 0.1))

    # Poisoning configuration
    poisoned_clients_str = context.run_config.get("poisoned-clients", "")
    flip_percent = context.run_config.get("flip-percent", "000")

    # Parse poisoned clients
    poisoned_clients = set()
    if poisoned_clients_str and "000" != flip_percent:
        poisoned_clients = {int(x.strip()) for x in poisoned_clients_str.split(',') if x.strip()}

    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        poisoned=(len(poisoned_clients) > 0 and "000" != flip_percent),
        flip_percent=flip_percent
    )

    # Create save directory
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("CIFAR-10 Federated Learning Server")
    print("=" * 70)
    print(f"Total Rounds: {num_rounds}")
    print(f"Fraction Train: {fraction_train}")
    print(f"Initial Learning Rate: {initial_lr}")
    print(f"Poisoned Clients: {sorted(list(poisoned_clients)) if poisoned_clients else 'None'}")
    print(f"Flip Percentage: {flip_percent}%")
    print(f"Checkpoint Directory: {save_dir.absolute()}")
    print("=" * 70 + "\n")

    # Initialize global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Create custom strategy with adaptive learning rate
    strategy = CustomFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1,  # Evaluate all clients
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        metrics_logger=metrics_logger,
        poisoned_clients=poisoned_clients,
    )

    # Set save path for checkpoints
    strategy.set_save_path(save_dir)

    # Start federated learning
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": initial_lr}),
        num_rounds=num_rounds,
    )

    # Save final model
    print("\n" + "=" * 70)
    print("Saving final model...")
    final_model_path = save_dir / "final_model.pt"
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")

    # Save metrics
    metrics_logger.save_to_csv()
    print(f"✅ Training metrics saved to: {metrics_logger.output_dir}/")
    print("=" * 70 + "\n")