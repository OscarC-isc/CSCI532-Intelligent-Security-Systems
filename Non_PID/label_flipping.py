import os
import glob
import random
from typing import List

# --- Configuration ---
BASE_DIR: str = "CIFAR_clients"
OUTPUT_ROOT: str = "flipped_labels"  # Renamed output root for clarity
CLIENT_RANGE: range = range(1, 21)
POISONED_CLIENT_IDS: List[int] = [3, 17]
FLIP_PERCENTAGES: List[int] = [10, 25, 50, 75, 100]
NUM_CLASSES: int = 10  # CIFAR-10 has 10 classes (0-9)

# Set random seed for reproducibility
random.seed(42)


def load_label(filepath: str) -> int:
    """Load a label from a text file."""
    with open(filepath, 'r') as f:
        return int(f.read().strip())


def save_label(label: int, filepath: str):
    """Save a label to a text file."""
    with open(filepath, 'w') as f:
        f.write(str(label))


def get_flipped_label(original_label: int) -> int:
    """
    Apply label flipping randomly.
    The new label is chosen uniformly at random from the other 9 classes.
    """
    # Create a list of all possible class labels (0 to 9)
    all_labels = list(range(NUM_CLASSES))

    # Remove the original label from the list of choices
    possible_flips = [label for label in all_labels if label != original_label]

    # Select one of the remaining 9 labels uniformly at random
    return random.choice(possible_flips)


def run_label_flipping_for_poisoned_clients():
    """
    Create randomly flipped labels for poisoned clients.
    """
    print(f"Starting RANDOM label flipping for poisoned clients {POISONED_CLIENT_IDS}")

    # Load label files for poisoned clients
    poisoned_labels = {}
    for client_id in POISONED_CLIENT_IDS:
        client_dir = os.path.join(BASE_DIR, f"client_{client_id}", "labels")
        if not os.path.isdir(client_dir):
            raise FileNotFoundError(f"Directory not found: {client_dir}")

        label_files = sorted(glob.glob(os.path.join(client_dir, "*.txt")))
        if not label_files:
            raise FileNotFoundError(f"No label files in: {client_dir}")

        poisoned_labels[client_id] = label_files
        print(f"  Client {client_id} has {len(label_files)} label files")

    # Process each flip percentage
    for flip_perc in FLIP_PERCENTAGES:
        flip_ratio = flip_perc / 100.0
        print(f"\n>> Processing flip percentage: {flip_perc}% (Random Flip)")

        for client_id in POISONED_CLIENT_IDS:
            label_files = poisoned_labels[client_id]
            num_files = len(label_files)
            num_to_flip = int(num_files * flip_ratio)

            # Create output directory
            output_dir = os.path.join(
                OUTPUT_ROOT,
                f"{flip_perc:03d}percent_flip",
                f"client_{client_id}",
                "labels"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Randomly select indices to flip
            # Note: This uses the global random seed set at the top.
            flip_indices = set(random.sample(range(num_files), num_to_flip))

            flipped_count = 0
            kept_count = 0

            for idx, label_path in enumerate(label_files):
                filename = os.path.basename(label_path)
                output_path = os.path.join(output_dir, filename)

                # Load original label
                original_label = load_label(label_path)

                if idx in flip_indices and flip_perc > 0:
                    # Flip the label
                    flipped_label = get_flipped_label(original_label)
                    save_label(flipped_label, output_path)
                    flipped_count += 1
                else:
                    # Keep original label
                    save_label(original_label, output_path)
                    kept_count += 1

            print(f"  Client {client_id}: {flipped_count} labels flipped, {kept_count} kept")

    print("\nFinished random label flipping for poisoned clients.")


def verify_flipped_labels():
    """Verify that labels were flipped correctly by sampling a few examples."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Checking randomly flipped labels")
    print("=" * 60)

    for client_id in POISONED_CLIENT_IDS[:1]:  # Check first poisoned client
        original_dir = os.path.join(BASE_DIR, f"client_{client_id}", "labels")
        original_files = sorted(glob.glob(os.path.join(original_dir, "*.txt")))

        if not original_files:
            continue

        # Check first 5 files across different flip percentages
        sample_files = original_files[:5]

        print(f"\nClient {client_id} - Sample verification:")
        print(f"{'File':<15} {'Original':<10} ", end="")
        for perc in FLIP_PERCENTAGES:
            print(f"{perc}%{'':<8}", end="")
        print()
        print("-" * 70)

        for sample_file in sample_files:
            filename = os.path.basename(sample_file)
            original_label = load_label(sample_file)

            print(f"{filename:<15} {original_label:<10} ", end="")

            for perc in FLIP_PERCENTAGES:
                flipped_path = os.path.join(
                    OUTPUT_ROOT,
                    f"{perc:03d}percent_flip",
                    f"client_{client_id}",
                    "labels",
                    filename
                )
                if os.path.exists(flipped_path):
                    flipped_label = load_label(flipped_path)
                    marker = "✓" if flipped_label != original_label else " "
                    print(f"{flipped_label} {marker:<8}", end="")
                else:
                    print(f"N/A{'':<8}", end="")
            print()

    print("\n✓ marks flipped labels (different from original)")


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("CIFAR-10 RANDOM Label Flipping Script")
        print("=" * 60)
        print(f"Base directory: {BASE_DIR}")
        print(f"Output directory: {OUTPUT_ROOT}")
        print(f"Poisoned clients: {POISONED_CLIENT_IDS}")
        print(f"Flip percentages: {FLIP_PERCENTAGES}")
        print("Label Flip Strategy: Random (Target $\neq$ Original)")
        print("=" * 60 + "\n")

        # Create flipped labels for poisoned clients
        run_label_flipping_for_poisoned_clients()

        # Verify the results
        verify_flipped_labels()

        print("\n" + "=" * 60)
        print("✅ Random label flipping process complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()