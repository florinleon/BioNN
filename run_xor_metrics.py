import csv
import math
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from bionn import BioNN, MakeBioNNConfig
from activation_structure_metrics import evaluate_model_structure, write_metric_reports


NoRuns = 3
RandomSeed = 753159

ResultsDir = "bionn_xor"
InputDimension = 40000
TrainSize = 200
TestSize = 200
LabelNoise = 0.05
MuScale = 2.5
NoEpochs = 5000
EvaluationInterval = 1
LearningRate = 0.05
WeightDecay = 1e-4

# XOR overrides the input dimension and scale-sensitive quantities required by the normalized high-dimensional input.
# The normalization produces smaller initial activations, so the activity cutoff and threshold shift range are
# set to correspondingly small values.
BioNNConfigOverrides = {
    "InputDim": InputDimension,
    "ThresholdStrength": 0.05,
    "ActivityEpsilon": 1e-7,
    "StructuralDensity": 1,
    "UseOutputBias": False,
}

UseInputGate = False
UseGainModulation = False
UseThresholdModulation = False
UseLateralInhibition = True
UseHomeostasis = False
UseStructuralPlasticity = False
UseActivationDecorrelation = False

PerfectAccuracy = 1.0
HighAccuracyThreshold = 0.95
StableEvaluationCount = 5
DeviceName = "cuda" if torch.cuda.is_available() else "cpu"

# VisualizationCheckpoints = [1] + list(range(500, NoEpochs + 1, 500))
# VisualizationRunIndex = 0
# VisualizationFilePrefix = "hidden_groups_epoch"
GroupLabels = ["+mu1", "-mu1", "+mu2", "-mu2"]


# The model configuration combines task-specific numerical overrides with the selected mechanism switches.
# Mechanism switches are applied after numerical overrides so they do not alter numerical parameter values.
def make_config():
    config = dict(BioNNConfigOverrides)
    config.update({
        "UseInputGate": UseInputGate,
        "UseGainModulation": UseGainModulation,
        "UseThresholdModulation": UseThresholdModulation,
        "UseLateralInhibition": UseLateralInhibition,
        "UseHomeostasis": UseHomeostasis,
        "UseStructuralPlasticity": UseStructuralPlasticity,
        "UseActivationDecorrelation": UseActivationDecorrelation,
    })
    return MakeBioNNConfig(config)


def apply_mechanism_settings(settings=None, results_suffix=None):
    global UseInputGate, UseGainModulation, UseThresholdModulation, UseLateralInhibition
    global UseHomeostasis, UseStructuralPlasticity, UseActivationDecorrelation, Config, ResultsDir

    if settings is not None:
        allowed = {
            "UseInputGate",
            "UseGainModulation",
            "UseThresholdModulation",
            "UseLateralInhibition",
            "UseHomeostasis",
            "UseStructuralPlasticity",
            "UseActivationDecorrelation",
        }
        unknown = sorted(set(settings) - allowed)
        if len(unknown) > 0:
            raise ValueError("Unknown mechanism setting(s): {}".format(", ".join(unknown)))
        UseInputGate = settings.get("UseInputGate", UseInputGate)
        UseGainModulation = settings.get("UseGainModulation", UseGainModulation)
        UseThresholdModulation = settings.get("UseThresholdModulation", UseThresholdModulation)
        UseLateralInhibition = settings.get("UseLateralInhibition", UseLateralInhibition)
        UseHomeostasis = settings.get("UseHomeostasis", UseHomeostasis)
        UseStructuralPlasticity = settings.get("UseStructuralPlasticity", UseStructuralPlasticity)
        UseActivationDecorrelation = settings.get("UseActivationDecorrelation", UseActivationDecorrelation)

    Config = make_config()
    if results_suffix is not None:
        ResultsDir = "bionn_xor_{}_results".format(results_suffix)


Config = make_config()


# These fields are kept in a fixed order so reports from different configurations are directly comparable.
SummaryFields = [
    "run_index", "seed", "final_train_accuracy", "final_test_accuracy", "first_perfect_epoch", "first_high_epoch",
    "stable_high_accuracy", "activation_sparsity", "structural_sparsity", "wall_clock_seconds"
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_group_indices(clean_labels, center_signs):
    # Group indices follow the underlying XOR cluster rather than the possibly flipped observed label.
    # This preserves the latent task structure and supports analysis of whether the hidden representation becomes
    # organized around the four true clusters: +mu1, -mu1, +mu2, and -mu2.
    group_index = np.empty(clean_labels.shape[0], dtype=np.int64)
    positive_class = clean_labels > 0.0
    positive_center = center_signs > 0.0
    group_index[np.logical_and(positive_class, positive_center)] = 0
    group_index[np.logical_and(positive_class, np.logical_not(positive_center))] = 1
    group_index[np.logical_and(np.logical_not(positive_class), positive_center)] = 2
    group_index[np.logical_and(np.logical_not(positive_class), np.logical_not(positive_center))] = 3
    return group_index


def sample_xor_dataset(rng):
    # The distribution has two informative axes embedded in high-dimensional isotropic noise. Labels select the axis,
    # while an independent sign selects the cluster center on that axis.
    total_size = TrainSize + TestSize
    mu_norm = MuScale * np.sqrt(InputDimension / TrainSize)

    clean_labels = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=total_size)
    center_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=total_size)
    group_index = build_group_indices(clean_labels, center_signs)

    signal_coordinates = rng.standard_normal((total_size, 2)).astype(np.float32)
    positive_class = clean_labels > 0.0
    signal_coordinates[positive_class, 0] += center_signs[positive_class] * mu_norm
    signal_coordinates[~positive_class, 1] += center_signs[~positive_class] * mu_norm

    residual_coordinates = rng.standard_normal((total_size, InputDimension - 2)).astype(np.float32)
    x = np.concatenate([signal_coordinates, residual_coordinates], axis=1)
    x = (x / np.sqrt(InputDimension)).astype(np.float32)

    train_labels = clean_labels[:TrainSize].copy()
    flipped = rng.random(TrainSize) < LabelNoise
    train_labels[flipped] *= -1.0
    test_labels = clean_labels[TrainSize:].copy()

    return {
        "TrainX": torch.tensor(x[:TrainSize], dtype=torch.float32, device=DeviceName),
        "TestX": torch.tensor(x[TrainSize:], dtype=torch.float32, device=DeviceName),
        "TrainY": torch.tensor(train_labels, dtype=torch.float32, device=DeviceName),
        "TestY": torch.tensor(test_labels, dtype=torch.float32, device=DeviceName),
        "TrainGroupIndex": group_index[:TrainSize],
        "TestGroupIndex": group_index[TrainSize:],
    }


def logistic_loss(scores, labels):
    return F.softplus(-labels * scores).mean()


def accuracy_from_scores(scores, labels):
    predictions = torch.where(scores >= 0.0, torch.ones_like(labels), -torch.ones_like(labels))
    return predictions.eq(labels).float().mean().item()


def evaluate_model(model, dataset):
    model.eval()
    with torch.no_grad():
        train_scores = model(dataset["TrainX"], update_homeostasis=False)
        test_scores = model(dataset["TestX"], update_homeostasis=False)
        train_accuracy = accuracy_from_scores(train_scores, dataset["TrainY"])
        test_accuracy = accuracy_from_scores(test_scores, dataset["TestY"])
    return train_accuracy, test_accuracy


def measure_layer_sparsity(model, x):
    # Activation sparsity is measured on the final hidden layer. Structural sparsity reports the active mask fraction only
    # when structural plasticity is enabled; otherwise the dense weight tensor is the effective circuit.
    model.eval()
    with torch.no_grad():
        hidden = model.hidden_activity(x)
    activation_sparsity = (hidden <= Config["ActivityEpsilon"]).float().mean().item()
    structural_sparsity = 1.0 - model.active_connection_fraction(layer_index=0)
    return activation_sparsity, structural_sparsity



def make_structure_groups(dataset):
    # The structure report uses the latent four-cluster partition rather than the noisy observed training labels because
    # it measures whether the hidden layer recovers the underlying XOR geometry.
    all_x = torch.cat([dataset["TrainX"], dataset["TestX"]], dim=0)
    all_groups = np.concatenate([dataset["TrainGroupIndex"], dataset["TestGroupIndex"]], axis=0)
    return all_x, all_groups, GroupLabels


def first_reach_time(times, values, threshold):
    for index, value in enumerate(values):
        if value >= threshold:
            return int(times[index])
    return None


def has_stable_high_accuracy(values):
    # Stability is defined as several consecutive high-accuracy evaluations. This avoids counting a single lucky spike.
    if len(values) < StableEvaluationCount:
        return False
    for start in range(0, len(values) - StableEvaluationCount + 1):
        window = values[start:start + StableEvaluationCount]
        if np.all(window >= HighAccuracyThreshold):
            return True
    return False


# def should_save_visualization(run_index, epoch):
#     # Hidden units are exchangeable across independent runs, so averaging corresponding unit indices across runs
#     # would blur cluster structure. The visualization therefore uses a single reference run.
#     return run_index == VisualizationRunIndex and epoch in VisualizationCheckpoints


# def save_hidden_group_heatmap(model, dataset, run_index, epoch, output_dir):
#     # The heatmap shows one row for each latent XOR cluster and one column for each hidden unit. We average over both
#     # train and test inputs so the image reflects the learned representation itself rather than the noisy train labels.
#     model.eval()
#     with torch.no_grad():
#         all_x = torch.cat([dataset["TrainX"], dataset["TestX"]], dim=0)
#         hidden = model.hidden_activity(all_x).detach().cpu().numpy()
#
#     all_group_index = np.concatenate([dataset["TrainGroupIndex"], dataset["TestGroupIndex"]], axis=0)
#     group_means = np.zeros((len(GroupLabels), hidden.shape[1]), dtype=np.float64)
#     for group_index in range(len(GroupLabels)):
#         mask = all_group_index == group_index
#         if np.any(mask):
#             group_means[group_index] = hidden[mask].mean(axis=0)
#
#     # We plot the raw group means directly and use a fixed color scale from 0 to 1. This makes differences in
#     # absolute activation magnitude visible across checkpoints, which is important when early activations are weak.
#     raw_max = float(group_means.max()) if group_means.size > 0 else 0.0
#
#     plt.figure(figsize=(10, 3.6))
#     image = plt.imshow(group_means, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
#     plt.colorbar(image, fraction=0.046, pad=0.04, label="mean activation")
#     plt.xticks(np.arange(hidden.shape[1]), np.arange(1, hidden.shape[1] + 1))
#     plt.yticks(np.arange(len(GroupLabels)), GroupLabels)
#     plt.xlabel("hidden unit")
#     plt.ylabel("latent XOR group")
#     plt.title("Run {} epoch {} hidden-group heatmap (raw max {:.4f})".format(run_index, epoch, raw_max))
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "{}_{:04d}.png".format(VisualizationFilePrefix, epoch)), dpi=200)
#     plt.close()


def train_single_run(run_index, run_seed, save_visualizations=False, visualization_dir=None):
    set_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    dataset = sample_xor_dataset(rng)
    model = BioNN(Config).to(DeviceName)
    optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate, weight_decay=WeightDecay)
    start_time = time.perf_counter()

    epochs = list(range(0, NoEpochs + 1, EvaluationInterval))
    train_accuracy = np.zeros(len(epochs), dtype=np.float64)
    test_accuracy = np.zeros(len(epochs), dtype=np.float64)
    evaluation_index = 0

    train_accuracy[evaluation_index], test_accuracy[evaluation_index] = evaluate_model(model, dataset)
    evaluation_index += 1

    for epoch in range(1, NoEpochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores, details = model(dataset["TrainX"], update_homeostasis=True, return_details=True)
        loss = logistic_loss(scores, dataset["TrainY"]) + model.regularization_loss(details)
        loss.backward()
        optimizer.step()

        # if save_visualizations and epoch in VisualizationCheckpoints:
        #     save_hidden_group_heatmap(model, dataset, run_index, epoch, visualization_dir)

        if epoch % EvaluationInterval == 0:
            train_accuracy[evaluation_index], test_accuracy[evaluation_index] = evaluate_model(model, dataset)
            evaluation_index += 1

    wall_clock_seconds = time.perf_counter() - start_time
    activation_sparsity, structural_sparsity = measure_layer_sparsity(model, dataset["TrainX"])
    structure_x, structure_groups, structure_group_labels = make_structure_groups(dataset)
    structure_metrics = evaluate_model_structure(
        model, structure_x, structure_groups, Config["ActivityEpsilon"], structure_group_labels
    )
    run_summary = {
        "run_index": run_index,
        "seed": run_seed,
        "final_train_accuracy": float(train_accuracy[-1]),
        "final_test_accuracy": float(test_accuracy[-1]),
        "first_perfect_epoch": first_reach_time(epochs, test_accuracy, PerfectAccuracy),
        "first_high_epoch": first_reach_time(epochs, test_accuracy, HighAccuracyThreshold),
        "stable_high_accuracy": has_stable_high_accuracy(test_accuracy),
        "activation_sparsity": float(activation_sparsity),
        "structural_sparsity": float(structural_sparsity),
        "wall_clock_seconds": float(wall_clock_seconds),
    }
    history_rows = []
    for index, epoch in enumerate(epochs):
        history_rows.append({
            "run_index": run_index,
            "seed": run_seed,
            "epoch": int(epoch),
            "train_accuracy": float(train_accuracy[index]),
            "test_accuracy": float(test_accuracy[index]),
        })
    structure_metrics["run_index"] = run_index
    structure_metrics["seed"] = run_seed
    return np.array(epochs, dtype=np.int64), train_accuracy, test_accuracy, run_summary, history_rows, structure_metrics


def aggregate_numeric(values):
    finite_values = [value for value in values if value is not None]
    if len(finite_values) == 0:
        return None, None
    return min(finite_values), sum(finite_values) / len(finite_values)


def aggregate_summaries(run_summaries):
    aggregate = {}
    for field in SummaryFields:
        if field in ["run_index", "seed"]:
            continue
        if field == "stable_high_accuracy":
            values = [1.0 if row[field] else 0.0 for row in run_summaries]
        else:
            values = [row[field] for row in run_summaries]
        aggregate[field] = aggregate_numeric(values)
    return aggregate


def format_value(value):
    if value is None:
        return "not_reached"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "{:.6f}".format(value)
    return str(value)


def write_history_csv(history_rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_index", "seed", "epoch", "train_accuracy", "test_accuracy"])
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


def write_run_metrics_csv(run_summaries, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SummaryFields)
        writer.writeheader()
        for row in run_summaries:
            writer.writerow(row)


def write_summary_text(run_summaries, path):
    aggregate = aggregate_summaries(run_summaries)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("XOR BioNN metrics\n")
        handle.write("========================\n\n")
        handle.write("This file summarizes the XOR BioNN configuration with fixed parameters across ablations.\n")
        stable_note = "Stable high accuracy means at least {} consecutive evaluations with test accuracy >= {:.2f}.\n\n"
        handle.write(stable_note.format(StableEvaluationCount, HighAccuracyThreshold))
        handle.write("Configuration notes:\n")
        handle.write("- learned context, affine-ReLU hidden layer, and linear readout are fixed across ablations\n")
        handle.write("- mechanism switches only enable or disable their own computational contribution\n")
        handle.write("- numerical mechanism parameters remain fixed even when the corresponding switch is off\n\n")
        handle.write("Per-run metrics:\n")
        for row in run_summaries:
            handle.write("\nRun {} seed {}\n".format(row["run_index"], row["seed"]))
            for field in SummaryFields:
                if field in ["run_index", "seed"]:
                    continue
                handle.write("{}: {}\n".format(field, format_value(row[field])))
        handle.write("\nAggregate metrics across {} runs:\n".format(len(run_summaries)))
        for field in SummaryFields:
            if field in ["run_index", "seed"]:
                continue
            minimum, average = aggregate[field]
            handle.write("{} minimum: {}\n".format(field, format_value(minimum)))
            handle.write("{} average: {}\n".format(field, format_value(average)))


def plot_accuracy(epochs, all_train_accuracy, all_test_accuracy, output_path):
    train_mean = np.mean(all_train_accuracy, axis=0)
    test_mean = np.mean(all_test_accuracy, axis=0)
    train_low, train_high = np.quantile(all_train_accuracy, [0.2, 0.8], axis=0)
    test_low, test_high = np.quantile(all_test_accuracy, [0.2, 0.8], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mean, label="train mean")
    plt.plot(epochs, test_mean, label="test mean")
    plt.fill_between(epochs, train_low, train_high, alpha=0.2, label="train 20-80%")
    plt.fill_between(epochs, test_low, test_high, alpha=0.2, label="test 20-80%")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="random guess")
    plt.axhline(HighAccuracyThreshold, linestyle=":", linewidth=1, label="high accuracy")
    plt.ylim(0.45, 1.02)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("BioNN on XOR-cluster task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_experiment(mechanism_settings=None, results_suffix=None):
    apply_mechanism_settings(mechanism_settings, results_suffix)

    os.makedirs(ResultsDir, exist_ok=True)
    set_seed(RandomSeed)
    master_rng = np.random.default_rng(RandomSeed)
    all_train_accuracy = []
    all_test_accuracy = []
    all_history_rows = []
    run_summaries = []
    structure_metric_rows = []
    epochs = None

    for run_index in range(NoRuns):
        run_seed = int(master_rng.integers(0, 2 ** 31 - 1))
        save_visualizations = False
        epochs, train_accuracy, test_accuracy, run_summary, history_rows, structure_metrics = train_single_run(
            run_index, run_seed, save_visualizations=save_visualizations, visualization_dir=ResultsDir
        )
        all_train_accuracy.append(train_accuracy)
        all_test_accuracy.append(test_accuracy)
        run_summaries.append(run_summary)
        structure_metric_rows.append(structure_metrics)
        all_history_rows.extend(history_rows)
        message = "XOR run {:03d}/{:03d}: train {:.3f}, test {:.3f}"
        print(message.format(run_index + 1, NoRuns, train_accuracy[-1], test_accuracy[-1]))

    all_train_accuracy = np.stack(all_train_accuracy, axis=0)
    all_test_accuracy = np.stack(all_test_accuracy, axis=0)
    write_history_csv(all_history_rows, os.path.join(ResultsDir, "accuracy_history.csv"))
    write_run_metrics_csv(run_summaries, os.path.join(ResultsDir, "run_metrics.csv"))
    write_summary_text(run_summaries, os.path.join(ResultsDir, "metrics_summary.txt"))
    write_metric_reports(ResultsDir, "XOR BioNN", structure_metric_rows)
    plot_accuracy(epochs, all_train_accuracy, all_test_accuracy, os.path.join(ResultsDir, "accuracy.png"))
    return run_summaries


if __name__ == "__main__":
    run_experiment()
