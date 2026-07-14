"""Adapted from flower-tutorial: A Flower / PyTorch app with poisoning support."""

import torch
import time
import pickle
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from CIFAR_FL_Flower.task import Net, load_data
from CIFAR_FL_Flower.task import test as test_fn
from CIFAR_FL_Flower.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    start_time = time.time()

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get partition ID and configuration
    partition_id = context.node_config["partition-id"] + 1
    batch_size = context.run_config.get("batch-size", 1)

    # Get poisoning configuration from run_config (from server)
    poisoned_clients = context.run_config.get("poisoned-clients", [])
    flip_percent = context.run_config.get("flip-percent", "000")
    if "000" != flip_percent and len(poisoned_clients) > 0:
        poisoned_clients = [int(id_str) for id_str in poisoned_clients.split(',')]
        is_poisoned = (partition_id in poisoned_clients)
    else:
        is_poisoned = False

    # Print client status
    if is_poisoned:
        print(f"   CLIENT {partition_id} (partition {partition_id}): POISONED")
        print(f"    Flip percentage: {flip_percent}%")
        print(f"    Loading labels from: flipped_labels/{flip_percent}percent_flip/client_{partition_id}/labels/")
    else:
        print(f"   CLIENT {partition_id} (partition {partition_id}):")
        print(f"    Loading labels from: CIFAR_clients/client_{partition_id}/labels/")

    # Load the data with appropriate poisoning
    trainloader, _ = load_data(
        partition_id=partition_id,
        batch_size=batch_size,
        poisoned=is_poisoned,
        flip_percent=flip_percent,
        poisoned_clients=poisoned_clients
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Create training metadata
    train_metadata = {
        "training_time": training_time,
        "partition_id": partition_id,
        "poisoned": is_poisoned,
        "flip_percent": flip_percent if is_poisoned else "000",
    }

    # Serialize the metadata object to bytes
    train_meta_bytes = pickle.dumps(train_metadata)
    config_record = ConfigRecord({"meta": train_meta_bytes})

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "training_time": training_time,
        "partition_id": partition_id,
        "poisoned": int(is_poisoned) if is_poisoned else 0,
    }
    metric_record = MetricRecord(metrics)

    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
        "train_metadata": config_record,
    })

    print(f"Client {partition_id} - Training complete. Loss: {train_loss:.4f}, Time: {training_time:.2f}s")

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get partition ID and configuration
    partition_id = context.node_config["partition-id"] + 1
    batch_size = context.run_config.get("batch-size", 1)
    # Get poisoning configuration from run_config (from server)
    poisoned_clients = context.run_config.get("poisoned-clients", [])
    flip_percent = context.run_config.get("flip-percent", "000")
    if "000" != flip_percent and len(poisoned_clients) > 0:
        poisoned_clients = [int(id_str) for id_str in poisoned_clients.split(',')]
        is_poisoned = (partition_id in poisoned_clients)
    else:
        is_poisoned = False

    # Load the data with same poisoning as training (IMPORTANT for consistency)
    _, valloader = load_data(
        partition_id=partition_id,
        batch_size=batch_size,
        poisoned=is_poisoned,
        flip_percent=flip_percent,
        poisoned_clients = poisoned_clients,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Print evaluation results
    status_icon = "⚠️ " if is_poisoned else "✓ "
    print(f"{status_icon}Client {partition_id} - Eval Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "partition_id": partition_id,
        "poisoned": int(is_poisoned),  # ✅ convert bool to int
        "flip_percent": int(flip_percent) if is_poisoned else 0,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)