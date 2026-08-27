import csv
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from bionn import BioNN, MakeBioNNConfig
from activation_structure_metrics import evaluate_model_structure, write_metric_reports


NoRuns = 3
RandomSeed = 753159

OutputDir = "bionn_parity"
InputDim = 40
ParitySize = 3
TrainSize = 1000
TestSize = 100
BatchSize = 32
EpochCount = 300
LearningRate = 0.1
WeightDecay = 0.01
DeviceName = "cuda" if torch.cuda.is_available() else "cpu"

# Sparse parity overrides only task-specific model settings; all other numerical parameters use their configured defaults.
# Structural density, hidden width, context size, mechanism strengths, and regularization scales remain unchanged.
BioNNConfigOverrides = {
    "InputDim": InputDim,
    "UseOutputBias": False,
}

UseInputGate = True
UseGainModulation = False
UseThresholdModulation = False
UseLateralInhibition = True
UseHomeostasis = False
UseStructuralPlasticity = False
UseActivationDecorrelation = False
StructuralUpdateEpoch = 10

PerfectAccuracy = 1.0
HighAccuracyThreshold = 0.95
StableEvaluationCount = 5
ParityIndices = list(range(ParitySize))
ParityGroupLabels = ["target=-1", "target=+1"]


# The parity runner uses a single configuration path that combines task-specific settings with the current
# mechanism settings while leaving all other BioNN parameters at their configured defaults.
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
    global UseHomeostasis, UseStructuralPlasticity, UseActivationDecorrelation, Config, OutputDir

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
        OutputDir = "bionn_sparse_parity_{}_output".format(results_suffix)


Config = make_config()


SummaryFields = [
    "run_index", "seed", "final_train_accuracy", "final_test_accuracy", "first_perfect_epoch", "first_high_epoch",
    "stable_high_accuracy", "activation_sparsity", "structural_sparsity", "final_active_subnetwork_size",
    "wall_clock_seconds"
]


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, output, target):
        return torch.relu(1.0 - output.squeeze() * target.squeeze())


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_sparse_parity_data(sample_count, seed):
    # Sparse parity has a small rule embedded in many irrelevant signs. The target is the product of the first k bits.
    generator = random.Random(seed)
    samples = [[generator.choice([-1.0, 1.0]) for _ in range(InputDim)] for _ in range(sample_count)]
    x = torch.tensor(samples, dtype=torch.float32)
    y = torch.prod(x[:, ParityIndices], dim=1)
    return x, y


def make_data_loaders(seed):
    # The test set remains fixed across seeds, while the training set and model initialization vary with the run seed.
    train_x, train_y = make_sparse_parity_data(TrainSize, seed * 17)
    test_x, test_y = make_sparse_parity_data(TestSize, 2001)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=TestSize, shuffle=False)
    return train_loader, test_loader, train_x, train_y, test_x, test_y


def train_for_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DeviceName)
        y_batch = y_batch.to(DeviceName)
        optimizer.zero_grad(set_to_none=True)
        if UseActivationDecorrelation:
            output, details = model(x_batch, update_homeostasis=True, return_details=True)
            loss = loss_fn(output, y_batch).mean() + model.regularization_loss(details)
        else:
            loss = loss_fn(model(x_batch, update_homeostasis=True), y_batch).mean() + model.regularization_loss()
        loss.backward()
        optimizer.step()


def signed_predictions(logits):
    return torch.where(logits >= 0.0, torch.ones_like(logits), -torch.ones_like(logits))


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total_count = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(DeviceName)
            y_batch = y_batch.to(DeviceName)
            logits = model(x_batch, update_homeostasis=False)
            correct += signed_predictions(logits).eq(y_batch).sum().item()
            total_count += x_batch.shape[0]
    return correct / total_count


def feature_norms(model):
    # Active-subnetwork selection uses the effective first-layer feature norm, so masked connections do not count.
    return model.effective_weight_norms(layer_index=0)


def active_subnetwork_indices(model, support_x):
    model.eval()
    x = support_x.to(DeviceName)
    with torch.no_grad():
        hidden = model.hidden_activity(x)
        output_weight = model.readout.weight.detach().squeeze(0)
        contributions = hidden * output_weight.unsqueeze(0)
        full_sign = signed_predictions(contributions.sum(dim=1))
        order = torch.argsort(feature_norms(model), descending=True)
        cumulative_logits = torch.cumsum(contributions[:, order], dim=1)
        cumulative_signs = signed_predictions(cumulative_logits)
        matches = cumulative_signs.eq(full_sign.unsqueeze(1)).all(dim=0)
        match_positions = torch.nonzero(matches, as_tuple=False).flatten()
    if match_positions.numel() == 0:
        return order.detach().cpu().numpy()
    active_count = int(match_positions[0].item()) + 1
    return order[:active_count].detach().cpu().numpy()


def measure_layer_sparsity(model, support_x):
    model.eval()
    x = support_x.to(DeviceName)
    with torch.no_grad():
        hidden = model.hidden_activity(x)
    activation_sparsity = (hidden <= Config["ActivityEpsilon"]).float().mean().item()
    structural_sparsity = 1.0 - model.active_connection_fraction(layer_index=0)
    return activation_sparsity, structural_sparsity



def make_structure_groups(train_x, train_y, test_x, test_y):
    # Sparse parity has a binary target, so the structure report uses the target sign as the regime label. This keeps the
    # metric file directly tied to the supervised rule without introducing any extra assumptions about latent factors.
    all_x = torch.cat([train_x, test_x], dim=0)
    all_targets = torch.cat([train_y, test_y], dim=0)
    all_groups = torch.where(all_targets > 0.0, torch.ones_like(all_targets), torch.zeros_like(all_targets))
    return all_x, all_groups.detach().cpu().numpy(), ParityGroupLabels


def first_reach_time(times, values, threshold):
    for index, value in enumerate(values):
        if value >= threshold:
            return int(times[index])
    return None


def has_stable_high_accuracy(values):
    # Stability is defined on consecutive evaluation points. With one evaluation per epoch, the default is 5 epochs.
    if len(values) < StableEvaluationCount:
        return False
    for start in range(0, len(values) - StableEvaluationCount + 1):
        window = values[start:start + StableEvaluationCount]
        if np.all(window >= HighAccuracyThreshold):
            return True
    return False


def train_one_seed(run_index, seed):
    set_all_seeds(seed)
    train_loader, test_loader, train_x, train_y, test_x, test_y = make_data_loaders(seed)
    model = BioNN(Config).to(DeviceName)
    optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate, weight_decay=WeightDecay)
    loss_fn = HingeLoss()
    start_time = time.perf_counter()

    epochs = list(range(0, EpochCount + 1))
    train_accuracy = np.zeros(len(epochs), dtype=np.float64)
    test_accuracy = np.zeros(len(epochs), dtype=np.float64)
    train_accuracy[0] = evaluate_accuracy(model, train_loader)
    test_accuracy[0] = evaluate_accuracy(model, test_loader)

    for epoch in range(1, EpochCount + 1):
        train_for_one_epoch(model, train_loader, optimizer, loss_fn)
        if UseStructuralPlasticity and epoch % StructuralUpdateEpoch == 0:
            model.structural_update()
        train_accuracy[epoch] = evaluate_accuracy(model, train_loader)
        test_accuracy[epoch] = evaluate_accuracy(model, test_loader)

    wall_clock_seconds = time.perf_counter() - start_time
    final_active = active_subnetwork_indices(model, train_x)
    activation_sparsity, structural_sparsity = measure_layer_sparsity(model, train_x)
    structure_x, structure_groups, structure_group_labels = make_structure_groups(train_x, train_y, test_x, test_y)
    structure_metrics = evaluate_model_structure(
        model, structure_x, structure_groups, Config["ActivityEpsilon"], structure_group_labels
    )
    run_summary = {
        "run_index": run_index,
        "seed": seed,
        "final_train_accuracy": float(train_accuracy[-1]),
        "final_test_accuracy": float(test_accuracy[-1]),
        "first_perfect_epoch": first_reach_time(epochs, test_accuracy, PerfectAccuracy),
        "first_high_epoch": first_reach_time(epochs, test_accuracy, HighAccuracyThreshold),
        "stable_high_accuracy": has_stable_high_accuracy(test_accuracy),
        "activation_sparsity": float(activation_sparsity),
        "structural_sparsity": float(structural_sparsity),
        "final_active_subnetwork_size": int(len(final_active)),
        "wall_clock_seconds": float(wall_clock_seconds) }
    history_rows = []
    for index, epoch in enumerate(epochs):
        history_rows.append({
            "run_index": run_index,
            "seed": seed,
            "epoch": int(epoch),
            "train_accuracy": float(train_accuracy[index]),
            "test_accuracy": float(test_accuracy[index]) })
    structure_metrics["run_index"] = run_index
    structure_metrics["seed"] = seed
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
        handle.write("Sparse-parity BioNN metrics\n")
        handle.write("==========================\n\n")
        handle.write("This file summarizes the sparse-parity BioNN configuration with fixed parameters across ablations.\n")
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
    train_std = np.std(all_train_accuracy, axis=0)
    test_std = np.std(all_test_accuracy, axis=0)
    train_low = np.clip(train_mean - train_std, 0.0, 1.0)
    train_high = np.clip(train_mean + train_std, 0.0, 1.0)
    test_low = np.clip(test_mean - test_std, 0.0, 1.0)
    test_high = np.clip(test_mean + test_std, 0.0, 1.0)

    plt.figure(figsize=(8, 5))
    plot_epochs = epochs + 1
    plt.plot(plot_epochs, train_mean, label="train mean")
    plt.plot(plot_epochs, test_mean, label="test mean")
    plt.fill_between(plot_epochs, train_low, train_high, alpha=0.2, label="train ±1 std")
    plt.fill_between(plot_epochs, test_low, test_high, alpha=0.2, label="test ±1 std")
    plt.axhline(HighAccuracyThreshold, linestyle=":", linewidth=1, label="high accuracy")
    plt.ylim(0.45, 1.02)
    plt.xlabel("Epochs")  # epochs + 1
    plt.ylabel("Accuracy")
    plt.xscale("log")
    #plt.title("BioNN on sparse-parity task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_experiment(mechanism_settings=None, results_suffix=None):
    apply_mechanism_settings(mechanism_settings, results_suffix)

    os.makedirs(OutputDir, exist_ok=True)
    all_train_accuracy = []
    all_test_accuracy = []
    all_history_rows = []
    run_summaries = []
    structure_metric_rows = []
    epochs = None

    for run_index in range(NoRuns):
        run_seed = RandomSeed + run_index
        run_result = train_one_seed(run_index, run_seed)
        epochs, train_accuracy, test_accuracy, run_summary, history_rows, structure_metrics = run_result
        all_train_accuracy.append(train_accuracy)
        all_test_accuracy.append(test_accuracy)
        run_summaries.append(run_summary)
        structure_metric_rows.append(structure_metrics)
        all_history_rows.extend(history_rows)
        message = "Parity run {:03d}/{:03d}: train {:.3f}, test {:.3f}"
        print(message.format(run_index + 1, NoRuns, train_accuracy[-1], test_accuracy[-1]))

    all_train_accuracy = np.stack(all_train_accuracy, axis=0)
    all_test_accuracy = np.stack(all_test_accuracy, axis=0)
    write_history_csv(all_history_rows, os.path.join(OutputDir, "accuracy_history.csv"))
    write_run_metrics_csv(run_summaries, os.path.join(OutputDir, "run_metrics.csv"))
    write_summary_text(run_summaries, os.path.join(OutputDir, "metrics_summary.txt"))
    write_metric_reports(OutputDir, "Sparse-parity BioNN", structure_metric_rows)
    plot_accuracy(epochs, all_train_accuracy, all_test_accuracy, os.path.join(OutputDir, "accuracy.png"))
    return run_summaries


if __name__ == "__main__":
    run_experiment()
