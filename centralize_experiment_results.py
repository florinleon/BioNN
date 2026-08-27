import os
import pandas as pd


OutputFileName = "central_experiment_results.csv"

RequestedColumns = [
    "ExperimentCode",
    "Task",
    "IG",
    "GM",
    "TM",
    "LI",
    "HO",
    "SP",
    "AD",
    "Train09",
    "Test09",
    "Train1",
    "Test1",
    "PopulationSparsityAvg",
    "LifetimeSparsityAvg",
    "SampleHoyerAvg",
    "NeuronHoyerAvg",
    "DeadNeuronAvg",
    "MeanAbsCorrAvg",
    "EffRankAvg",
    "SpecializationAvg",
    "PurityAvg",
    "CentroidSepAvg",
    "WithinJaccardAvg",
    "BetweenJaccardAvg",
    "JaccardGapAvg",
    "ActivationSparsity",
    "StructuralSparsity",
    "FinalActiveSubnetworkSize",
]

StructureAverageColumns = [
    "PopulationSparsityAvg",
    "LifetimeSparsityAvg",
    "SampleHoyerAvg",
    "NeuronHoyerAvg",
    "DeadNeuronAvg",
    "MeanAbsCorrAvg",
    "EffRankAvg",
    "SpecializationAvg",
    "PurityAvg",
    "CentroidSepAvg",
    "WithinJaccardAvg",
    "BetweenJaccardAvg",
    "JaccardGapAvg",
]

SwitchColumns = ["IG", "GM", "TM", "LI", "HO", "SP", "AD"]

HardcodedSwitchesByExperimentCode = {
    "three_on__without_homeostasis": {"IG": 1, "GM": 0, "TM": 0, "LI": 1, "HO": 0, "SP": 1, "AD": 0},
    "three_on__without_inhibition": {"IG": 1, "GM": 0, "TM": 0, "LI": 0, "HO": 1, "SP": 1, "AD": 0},
    "three_on__without_input_gate": {"IG": 0, "GM": 0, "TM": 0, "LI": 1, "HO": 1, "SP": 1, "AD": 0},
    "three_on__without_structural": {"IG": 1, "GM": 0, "TM": 0, "LI": 1, "HO": 1, "SP": 0, "AD": 0},
}


def make_nan():
    return float("nan")


def parse_experiment_directory_name(directory_name):
    if directory_name.startswith("bionn_xor_") and directory_name.endswith("_results"):
        experiment_code = directory_name[len("bionn_xor_"):-len("_results")]
        return "xor", experiment_code

    if directory_name.startswith("bionn_sparse_parity_") and directory_name.endswith("_output"):
        experiment_code = directory_name[len("bionn_sparse_parity_"):-len("_output")]
        return "parity", experiment_code

    return None, None


def is_experiment_directory(directory_name):
    task, experiment_code = parse_experiment_directory_name(directory_name)
    return task is not None and experiment_code is not None


def make_empty_row(task, experiment_code):
    row = {}
    for column_name in RequestedColumns:
        row[column_name] = make_nan()
    row["ExperimentCode"] = experiment_code
    row["Task"] = task
    return row


def load_structure_summary_csv(csv_path, row):
    if not os.path.isfile(csv_path):
        return

    frame = pd.read_csv(csv_path)
    if frame.shape[0] == 0:
        return

    first_row = frame.iloc[0]

    # Keep ExperimentCode and Task from the directory name.
    # The exported summaries may contain generic values such as "ThreeOn",
    # which are not specific enough for the centralized table.
    for column_name in SwitchColumns:
        if column_name in frame.columns:
            row[column_name] = first_row[column_name]

    for column_name in StructureAverageColumns:
        if column_name in frame.columns:
            row[column_name] = first_row[column_name]


def load_run_metrics_csv(csv_path, row):
    if not os.path.isfile(csv_path):
        return

    frame = pd.read_csv(csv_path)
    if frame.shape[0] == 0:
        return

    if "activation_sparsity" in frame.columns:
        row["ActivationSparsity"] = frame["activation_sparsity"].astype(float).mean()

    if "structural_sparsity" in frame.columns:
        row["StructuralSparsity"] = frame["structural_sparsity"].astype(float).mean()

    if "final_active_subnetwork_size" in frame.columns:
        row["FinalActiveSubnetworkSize"] = frame["final_active_subnetwork_size"].astype(float).mean()
    else:
        row["FinalActiveSubnetworkSize"] = make_nan()


def detect_time_column(frame):
    if "epoch" in frame.columns:
        return "epoch"
    if "step" in frame.columns:
        return "step"
    return None


def first_reach_time(values_frame, accuracy_column, time_column, threshold):
    ordered_frame = values_frame.sort_values(by=time_column)
    for _, row in ordered_frame.iterrows():
        if float(row[accuracy_column]) >= threshold:
            return float(row[time_column])
    return -1.0


def average_nonnegative_or_minus_one(values):
    nonnegative_values = [value for value in values if value >= 0.0]
    if len(nonnegative_values) == 0:
        return -1.0
    return float(sum(nonnegative_values) / len(nonnegative_values))


def load_accuracy_history_csv(csv_path, row):
    if not os.path.isfile(csv_path):
        return

    frame = pd.read_csv(csv_path)
    if frame.shape[0] == 0:
        return

    time_column = detect_time_column(frame)
    if time_column is None:
        return

    if "run_index" in frame.columns:
        grouped_runs = frame.groupby("run_index", sort=True)
    else:
        grouped_runs = [(0, frame)]

    train_09_values = []
    test_09_values = []
    train_1_values = []
    test_1_values = []

    for _, run_frame in grouped_runs:
        train_09_values.append(first_reach_time(run_frame, "train_accuracy", time_column, 0.9))
        test_09_values.append(first_reach_time(run_frame, "test_accuracy", time_column, 0.9))
        train_1_values.append(first_reach_time(run_frame, "train_accuracy", time_column, 0.999))
        test_1_values.append(first_reach_time(run_frame, "test_accuracy", time_column, 0.999))

    row["Train09"] = average_nonnegative_or_minus_one(train_09_values)
    row["Test09"] = average_nonnegative_or_minus_one(test_09_values)
    row["Train1"] = average_nonnegative_or_minus_one(train_1_values)
    row["Test1"] = average_nonnegative_or_minus_one(test_1_values)


def apply_hardcoded_switches_from_experiment_code(row, directory_experiment_code):
    if directory_experiment_code not in HardcodedSwitchesByExperimentCode:
        return

    row["ExperimentCode"] = directory_experiment_code
    switch_values = HardcodedSwitchesByExperimentCode[directory_experiment_code]
    for column_name in SwitchColumns:
        row[column_name] = switch_values[column_name]


def build_row_from_directory(directory_path):
    task, directory_experiment_code = parse_experiment_directory_name(os.path.basename(directory_path))
    row = make_empty_row(task, directory_experiment_code)

    structure_summary_path = os.path.join(directory_path, "structure_metrics_summary_compact.csv")
    run_metrics_path = os.path.join(directory_path, "run_metrics.csv")
    accuracy_history_path = os.path.join(directory_path, "accuracy_history.csv")

    load_structure_summary_csv(structure_summary_path, row)
    load_run_metrics_csv(run_metrics_path, row)
    load_accuracy_history_csv(accuracy_history_path, row)
    apply_hardcoded_switches_from_experiment_code(row, directory_experiment_code)
    return row


def sort_rows(rows):
    return sorted(rows, key=lambda row: (str(row["Task"]), str(row["ExperimentCode"])))


def main():
    root_directory = os.getcwd()
    rows = []

    for entry_name in os.listdir(root_directory):
        entry_path = os.path.join(root_directory, entry_name)
        if not os.path.isdir(entry_path):
            continue
        if not is_experiment_directory(entry_name):
            continue
        rows.append(build_row_from_directory(entry_path))

    rows = sort_rows(rows)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=RequestedColumns)
    output_path = os.path.join(root_directory, OutputFileName)
    frame.to_csv(output_path, index=False)
    print("Wrote {} rows to {}.".format(frame.shape[0], output_path))


if __name__ == "__main__":
    main()
