import csv
import math
import os
import numpy as np
import torch


NumericalStabilityEpsilon = 1e-12
DefaultPairSampleCount = 8192

MetricFieldNames = [
    "PopulationSparsityMean",
    "LifetimeSparsityMean",
    "SampleHoyerSparsityMean",
    "NeuronHoyerSparsityMean",
    "DeadNeuronFraction",
    "MeanAbsoluteNeuronCorrelation",
    "NormalizedEffectiveRank",
    "MeanNeuronSpecialization",
    "MeanNeuronPurity",
    "CentroidSeparationRatio",
    "WithinGroupJaccard",
    "BetweenGroupJaccard",
    "JaccardGap",
]

MetricDescriptions = {
    "PopulationSparsityMean": "Mean fraction of hidden units that stay below the activity threshold for one sample.",
    "LifetimeSparsityMean": "Mean fraction of samples for which one hidden unit stays below the activity threshold.",
    "SampleHoyerSparsityMean": (
        "Average Hoyer sparsity of a sample activation vector. Higher means fewer strong units per sample."
    ),
    "NeuronHoyerSparsityMean": (
        "Average Hoyer sparsity of a neuron activation profile across samples. Higher means narrower firing support."
    ),
    "DeadNeuronFraction": "Fraction of hidden units that never rise above the activity threshold on the evaluation set.",
    "MeanAbsoluteNeuronCorrelation": "Mean absolute off-diagonal correlation between neuron activation profiles.",
    "NormalizedEffectiveRank": (
        "Entropy-based effective rank divided by the maximum possible rank for the activation matrix."
    ),
    "MeanNeuronSpecialization": (
        "Average normalized specialization score derived from each neuron's group-conditioned activity mass."
    ),
    "MeanNeuronPurity": "Average fraction of a neuron's activity mass assigned to its dominant group.",
    "CentroidSeparationRatio": (
        "Mean squared distance between group centroids divided by mean within-group squared distance."
    ),
    "WithinGroupJaccard": "Mean Jaccard overlap of active-unit sets for sample pairs drawn from the same group.",
    "BetweenGroupJaccard": "Mean Jaccard overlap of active-unit sets for sample pairs drawn from different groups.",
    "JaccardGap": (
        "Within-group Jaccard overlap minus between-group Jaccard overlap. Higher means cleaner regime separation."
    ),
}

MetadataFieldNames = ["SampleCount", "UnitCount", "GroupCount"]


MechanismFieldNames = [
    "UseInputGate",
    "UseGainModulation",
    "UseThresholdModulation",
    "UseLateralInhibition",
    "UseHomeostasis",
    "UseStructuralPlasticity",
    "UseActivationDecorrelation",
]

MechanismShortNames = {
    "UseInputGate": "IG",
    "UseGainModulation": "GM",
    "UseThresholdModulation": "TM",
    "UseLateralInhibition": "LI",
    "UseHomeostasis": "HO",
    "UseStructuralPlasticity": "SP",
    "UseActivationDecorrelation": "AD",
}

MechanismTokenNames = {
    "UseInputGate": "input_gate",
    "UseGainModulation": "gain",
    "UseThresholdModulation": "threshold",
    "UseLateralInhibition": "inhibition",
    "UseHomeostasis": "homeostasis",
    "UseStructuralPlasticity": "structural",
    "UseActivationDecorrelation": "decorrelation",
}

MetricShortNames = {
    "PopulationSparsityMean": "PopulationSparsity",
    "LifetimeSparsityMean": "LifetimeSparsity",
    "SampleHoyerSparsityMean": "SampleHoyer",
    "NeuronHoyerSparsityMean": "NeuronHoyer",
    "DeadNeuronFraction": "DeadNeuron",
    "MeanAbsoluteNeuronCorrelation": "MeanAbsCorr",
    "NormalizedEffectiveRank": "EffRank",
    "MeanNeuronSpecialization": "Specialization",
    "MeanNeuronPurity": "Purity",
    "CentroidSeparationRatio": "CentroidSep",
    "WithinGroupJaccard": "WithinJaccard",
    "BetweenGroupJaccard": "BetweenJaccard",
    "JaccardGap": "JaccardGap",
}

PhaseCodeNames = {
    "all_on": "AllOn",
    "all_off": "AllOff",
    "one_on": "OneOn",
    "one_off": "OneOff",
    "two_on": "TwoOn",
    "two_off": "TwoOff",
    "three_on": "ThreeOn",
    "three_off": "ThreeOff",
    "four_on": "FourOn",
    "four_off": "FourOff",
    "five_on": "FiveOn",
    "five_off": "FiveOff",
    "six_on": "SixOn",
    "six_off": "SixOff",
}


def _to_numpy_array(values):
    # Accept NumPy arrays, Python lists, and torch tensors, and normalize them to a NumPy representation so the
    # numerical routines receive a consistent input type; tensor values are detached and moved to CPU memory.
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _normalize_group_indices(groups, group_names=None):
    # Group labels may use any distinct hashable values. They are mapped to a dense 0..K-1 encoding for metric
    # computation, while optional display names remain associated with the corresponding groups.
    group_array = _to_numpy_array(groups).reshape(-1)
    unique_values = list(np.unique(group_array))
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    dense_groups = np.array([value_to_index[value] for value in group_array], dtype=np.int64)

    if group_names is None:
        resolved_names = [str(value) for value in unique_values]
    else:
        resolved_names = list(group_names)

    return dense_groups, resolved_names


def _safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _hoyer_sparsity(vectors):
    # Hoyer sparsity is well suited to nonnegative hidden activations because it responds both to support size and
    # amplitude concentration. All-zero vectors are treated as maximally sparse because they activate nothing at all.
    vectors = np.abs(np.asarray(vectors, dtype=np.float64))
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    scores = []
    for vector in vectors:
        size = vector.size
        l1_norm = np.linalg.norm(vector, ord=1)
        l2_norm = np.linalg.norm(vector, ord=2)
        if l2_norm <= NumericalStabilityEpsilon:
            scores.append(1.0)
            continue
        if size <= 1:
            scores.append(0.0)
            continue
        score = (math.sqrt(size) - l1_norm / l2_norm) / (math.sqrt(size) - 1.0)
        scores.append(float(score))

    return _safe_mean(scores)


def _mean_absolute_neuron_correlation(hidden):
    # Correlation is computed across neuron activation profiles, not across samples. This exposes whether several
    # neurons are effectively carrying the same code, which is one operational form of representational redundancy.
    if hidden.shape[1] < 2:
        return 0.0

    centered = hidden - hidden.mean(axis=0, keepdims=True)
    column_std = centered.std(axis=0)
    valid = column_std > NumericalStabilityEpsilon
    if np.sum(valid) < 2:
        return 0.0

    correlation = np.corrcoef(centered[:, valid], rowvar=False)
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(correlation[mask])))


def _normalized_effective_rank(hidden):
    # Effective rank uses the entropy of singular values rather than the raw algebraic rank. It therefore decreases
    # smoothly when the representation collapses into a smaller number of dominant directions.
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    singular_values = singular_values[singular_values > NumericalStabilityEpsilon]
    if singular_values.size == 0:
        return 0.0

    probabilities = singular_values / singular_values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + NumericalStabilityEpsilon))
    effective_rank = float(np.exp(entropy))
    max_rank = float(min(centered.shape[0], centered.shape[1]))
    if max_rank <= 0.0:
        return 0.0
    return effective_rank / max_rank


def _specialization_metrics(hidden, groups):
    # Each neuron receives a distribution over groups based on its mean absolute activation in that group. Low entropy
    # means the neuron spends most of its activation budget in a narrow regime, which matches the intended notion
    # of specialization better than raw class selectivity on signed outputs.
    unit_count = hidden.shape[1]
    group_count = int(np.max(groups)) + 1 if groups.size > 0 else 0
    if unit_count == 0 or group_count <= 1:
        return 0.0, 0.0, 0.0

    mean_activity = np.zeros((unit_count, group_count), dtype=np.float64)
    absolute_hidden = np.abs(hidden)
    for group_index in range(group_count):
        mask = groups == group_index
        if np.any(mask):
            mean_activity[:, group_index] = absolute_hidden[mask].mean(axis=0)

    specialization_scores = []
    purity_scores = []
    dead_units = 0
    log_group_count = math.log(group_count)
    for unit_index in range(unit_count):
        activity_mass = mean_activity[unit_index]
        total_mass = activity_mass.sum()
        if total_mass <= NumericalStabilityEpsilon:
            dead_units += 1
            specialization_scores.append(0.0)
            purity_scores.append(0.0)
            continue
        probabilities = activity_mass / total_mass
        entropy = -np.sum(probabilities * np.log(probabilities + NumericalStabilityEpsilon))
        specialization_scores.append(1.0 - entropy / log_group_count)
        purity_scores.append(float(np.max(probabilities)))

    dead_fraction = dead_units / float(unit_count)
    return _safe_mean(specialization_scores), _safe_mean(purity_scores), dead_fraction


def _centroid_separation_ratio(hidden, groups):
    # This ratio asks whether group-conditioned activation clouds are tighter around their own centroids than the
    # centroids are to one another. It is a direct numeric analogue of a heatmap that visibly separates regimes.
    group_count = int(np.max(groups)) + 1 if groups.size > 0 else 0
    if group_count <= 1:
        return 0.0

    centroids = []
    within_distances = []
    for group_index in range(group_count):
        mask = groups == group_index
        if not np.any(mask):
            continue
        group_hidden = hidden[mask]
        centroid = group_hidden.mean(axis=0)
        centroids.append(centroid)
        squared_distance = np.sum((group_hidden - centroid) ** 2, axis=1)
        within_distances.extend(squared_distance.tolist())

    if len(centroids) <= 1:
        return 0.0

    centroid_array = np.stack(centroids, axis=0)
    between_distances = []
    for left_index in range(centroid_array.shape[0]):
        for right_index in range(left_index + 1, centroid_array.shape[0]):
            squared_distance = np.sum((centroid_array[left_index] - centroid_array[right_index]) ** 2)
            between_distances.append(float(squared_distance))

    within_mean = _safe_mean(within_distances)
    between_mean = _safe_mean(between_distances)
    return between_mean / max(within_mean, NumericalStabilityEpsilon)


def _sample_pair_indices(groups, same_group, pair_sample_count):
    # Exact all-pairs overlap can be expensive and can overweight larger groups. This sampler uses a balanced,
    # deterministic Monte Carlo estimate to keep the metric computationally efficient and reproducible.
    rng = np.random.default_rng(0)
    group_to_indices = {}
    for index, group_index in enumerate(groups):
        group_to_indices.setdefault(int(group_index), []).append(index)

    valid_same_groups = [indices for indices in group_to_indices.values() if len(indices) >= 2]
    valid_group_ids = sorted(group_to_indices.keys())
    if same_group and len(valid_same_groups) == 0:
        return []
    if not same_group and len(valid_group_ids) < 2:
        return []

    sampled_pairs = []
    for _ in range(pair_sample_count):
        if same_group:
            group_indices = valid_same_groups[int(rng.integers(0, len(valid_same_groups)))]
            left_choice, right_choice = rng.choice(group_indices, size=2, replace=False)
            sampled_pairs.append((int(left_choice), int(right_choice)))
        else:
            left_group, right_group = rng.choice(valid_group_ids, size=2, replace=False)
            left_choice = rng.choice(group_to_indices[int(left_group)])
            right_choice = rng.choice(group_to_indices[int(right_group)])
            sampled_pairs.append((int(left_choice), int(right_choice)))

    return sampled_pairs


def _mean_jaccard(binary_hidden, pairs):
    if len(pairs) == 0:
        return 0.0

    overlaps = []
    for left_index, right_index in pairs:
        left = binary_hidden[left_index]
        right = binary_hidden[right_index]
        intersection = np.logical_and(left, right).sum()
        union = np.logical_or(left, right).sum()
        if union == 0:
            overlaps.append(1.0)
        else:
            overlaps.append(float(intersection) / float(union))

    return _safe_mean(overlaps)


def evaluate_model_structure(model, x, groups, activation_epsilon, group_names=None, layer_index=-1, pair_sample_count=None):
    # Structure metrics are computed from hidden activations on a fixed evaluation set. Evaluation reads activations
    # without updating model parameters or optimizer state.
    if pair_sample_count is None:
        pair_sample_count = DefaultPairSampleCount

    if isinstance(x, torch.Tensor):
        first_parameter = next(model.parameters(), None)
        if first_parameter is not None:
            x = x.to(device=first_parameter.device)

    model.eval()
    with torch.no_grad():
        hidden = model.hidden_activity(x, layer_index=layer_index)

    hidden_array = _to_numpy_array(hidden)
    dense_groups, resolved_group_names = _normalize_group_indices(groups, group_names)
    binary_hidden = np.abs(hidden_array) > float(activation_epsilon)

    report = {
        "SampleCount": int(hidden_array.shape[0]),
        "UnitCount": int(hidden_array.shape[1]),
        "GroupCount": int(len(resolved_group_names)),
        "GroupNames": resolved_group_names,
    }

    report["PopulationSparsityMean"] = float(np.mean(np.mean(np.logical_not(binary_hidden), axis=1)))
    report["LifetimeSparsityMean"] = float(np.mean(np.mean(np.logical_not(binary_hidden), axis=0)))
    report["SampleHoyerSparsityMean"] = _hoyer_sparsity(hidden_array)
    report["NeuronHoyerSparsityMean"] = _hoyer_sparsity(hidden_array.T)
    specialization, purity, dead_fraction = _specialization_metrics(hidden_array, dense_groups)
    report["DeadNeuronFraction"] = float(dead_fraction)
    report["MeanAbsoluteNeuronCorrelation"] = _mean_absolute_neuron_correlation(hidden_array)
    report["NormalizedEffectiveRank"] = _normalized_effective_rank(hidden_array)
    report["MeanNeuronSpecialization"] = float(specialization)
    report["MeanNeuronPurity"] = float(purity)
    report["CentroidSeparationRatio"] = _centroid_separation_ratio(hidden_array, dense_groups)

    within_pairs = _sample_pair_indices(dense_groups, True, pair_sample_count)
    between_pairs = _sample_pair_indices(dense_groups, False, pair_sample_count)
    report["WithinGroupJaccard"] = _mean_jaccard(binary_hidden, within_pairs)
    report["BetweenGroupJaccard"] = _mean_jaccard(binary_hidden, between_pairs)
    report["JaccardGap"] = report["WithinGroupJaccard"] - report["BetweenGroupJaccard"]
    return report


def aggregate_metric_rows(metric_rows):
    # Aggregate reports use min, mean, and max so the reader can see both the central tendency and the spread across
    # random seeds without having to reconstruct them manually from the per-run text file.
    aggregate = {}
    for field_name in MetricFieldNames:
        values = [float(row[field_name]) for row in metric_rows]
        aggregate[field_name] = {
            "minimum": float(np.min(values)) if len(values) > 0 else 0.0,
            "average": float(np.mean(values)) if len(values) > 0 else 0.0,
            "maximum": float(np.max(values)) if len(values) > 0 else 0.0,
        }

    return aggregate


def _format_metric_value(value):
    return "{:.6f}".format(float(value))


def write_metric_definitions(path, title):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{} structure metric definitions\n".format(title))
        handle.write("{}\n\n".format("=" * (len(title) + 27)))
        for field_name in MetricFieldNames:
            handle.write("{}\n".format(field_name))
            handle.write("{}\n\n".format(MetricDescriptions[field_name]))


def write_metric_run_report(metric_rows, path, title):
    # The per-run report records structure metrics separately for each run and includes shared evaluation metadata
    # so results can be compared consistently across runs.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{} structure metrics by run\n".format(title))
        handle.write("{}\n\n".format("=" * (len(title) + 25)))

        if len(metric_rows) == 0:
            handle.write("No runs were recorded.\n")
            return

        first_row = metric_rows[0]
        handle.write("Shared metadata\n")
        handle.write("---------------\n")
        for field_name in MetadataFieldNames:
            handle.write("{}: {}\n".format(field_name, first_row[field_name]))
        handle.write("GroupNames: {}\n\n".format(", ".join(first_row["GroupNames"])))

        for row in metric_rows:
            handle.write("Run {} seed {}\n".format(row["run_index"], row["seed"]))
            handle.write("----------------\n")
            for field_name in MetricFieldNames:
                handle.write("{}: {}\n".format(field_name, _format_metric_value(row[field_name])))
            handle.write("\n")


def write_metric_summary_report(metric_rows, path, title):
    aggregate = aggregate_metric_rows(metric_rows)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{} structure metric summary\n".format(title))
        handle.write("{}\n\n".format("=" * (len(title) + 25)))

        if len(metric_rows) == 0:
            handle.write("No runs were recorded.\n")
            return

        handle.write("Interpretation notes\n")
        handle.write("--------------------\n")
        handle.write(
            "Higher sparsity, specialization, centroid separation, and Jaccard gap indicate cleaner internal structure.\n"
        )
        handle.write(
            "Lower mean absolute correlation and lower normalized effective rank indicate less redundant activity.\n\n"
        )

        handle.write("Aggregate values across {} runs\n".format(len(metric_rows)))
        handle.write("------------------------------\n")
        for field_name in MetricFieldNames:
            summary = aggregate[field_name]
            handle.write("{} minimum: {}\n".format(field_name, _format_metric_value(summary["minimum"])))
            handle.write("{} average: {}\n".format(field_name, _format_metric_value(summary["average"])))
            handle.write("{} maximum: {}\n\n".format(field_name, _format_metric_value(summary["maximum"])))





def _extract_task_and_suffix(output_dir):
    # The summary CSV uses a compact task label and experiment code derived from the output-directory name.
    # Recognized directory prefixes determine the task label and experiment suffix.
    directory_name = os.path.basename(os.path.normpath(output_dir))

    xor_prefix = "bionn_xor_"
    parity_prefix = "bionn_sparse_parity_"
    if directory_name.startswith(xor_prefix) and directory_name.endswith("_results"):
        return "xor", directory_name[len(xor_prefix):-len("_results")]
    if directory_name.startswith(parity_prefix) and directory_name.endswith("_output"):
        return "parity", directory_name[len(parity_prefix):-len("_output")]

    return "", directory_name



def _decode_experiment_suffix(experiment_suffix):
    # The compact CSV stores the seven mechanism switches as explicit 0/1 columns. Experiment suffixes such as
    # all_off or two_on__gain__threshold encode the switch values and can be decoded deterministically.
    mechanism_values = {field_name: "" for field_name in MechanismFieldNames}
    token_to_field = {token_name: field_name for field_name, token_name in MechanismTokenNames.items()}

    if experiment_suffix == "all_on":
        for field_name in MechanismFieldNames:
            mechanism_values[field_name] = 1
        return mechanism_values

    if experiment_suffix == "all_off":
        for field_name in MechanismFieldNames:
            mechanism_values[field_name] = 0
        return mechanism_values

    if experiment_suffix == "four_on":
        retained_fields = [
            "UseInputGate",
            "UseLateralInhibition",
            "UseHomeostasis",
            "UseStructuralPlasticity",
        ]
        for field_name in MechanismFieldNames:
            mechanism_values[field_name] = 1 if field_name in retained_fields else 0
        return mechanism_values

    parts = experiment_suffix.split("__")
    phase_name = parts[0]
    mechanism_tokens = parts[1:]

    if phase_name == "three_on" and any(token_name.startswith("without_") for token_name in mechanism_tokens):
        # A three-on ablation with a without_* suffix uses input gate, lateral inhibition, homeostasis, and structural
        # plasticity as its retained set; the suffix identifies the retained mechanism disabled for that experiment.
        retained_fields = [
            "UseInputGate",
            "UseLateralInhibition",
            "UseHomeostasis",
            "UseStructuralPlasticity",
        ]
        for field_name in MechanismFieldNames:
            mechanism_values[field_name] = 1 if field_name in retained_fields else 0
        for token_name in mechanism_tokens:
            if not token_name.startswith("without_"):
                continue
            field_name = token_to_field.get(token_name[len("without_"):], None)
            if field_name is not None:
                mechanism_values[field_name] = 0
        return mechanism_values

    if phase_name.endswith("_on"):
        default_value = 0
        listed_value = 1
    elif phase_name.endswith("_off"):
        default_value = 1
        listed_value = 0
    else:
        return mechanism_values

    for field_name in MechanismFieldNames:
        mechanism_values[field_name] = default_value

    for token_name in mechanism_tokens:
        field_name = token_to_field.get(token_name, None)
        if field_name is not None:
            mechanism_values[field_name] = listed_value

    return mechanism_values



def _make_experiment_code(experiment_suffix):
    # The human-facing code should stay short in Excel filters and pivot tables while still remaining readable.
    if experiment_suffix in PhaseCodeNames:
        return PhaseCodeNames[experiment_suffix]

    parts = experiment_suffix.split("__")
    phase_name = parts[0]
    mechanism_tokens = parts[1:]
    phase_code = PhaseCodeNames.get(phase_name, phase_name)
    token_to_short_name = {
        token_name: MechanismShortNames[field_name]
        for field_name, token_name in MechanismTokenNames.items()
    }
    short_names = []
    for token_name in mechanism_tokens:
        if token_name in token_to_short_name:
            short_names.append(token_to_short_name[token_name])
        elif token_name.startswith("without_") and token_name[len("without_"):] in token_to_short_name:
            short_names.append("without_{}".format(token_to_short_name[token_name[len("without_"):]]))
    if len(short_names) == 0:
        return phase_code
    return "{}_{}".format(phase_code, "_".join(short_names))



def write_metric_summary_csv(metric_rows, path, output_dir):
    # This CSV is intentionally compact: averages come first for easy sorting in Excel, while minima and maxima are
    # kept at the end so they remain available without crowding the main comparison columns.
    aggregate = aggregate_metric_rows(metric_rows)
    task_name, experiment_suffix = _extract_task_and_suffix(output_dir)
    mechanism_values = _decode_experiment_suffix(experiment_suffix)

    field_names = ["ExperimentCode", "Task"] + [MechanismShortNames[field_name] for field_name in MechanismFieldNames]
    for field_name in MetricFieldNames:
        field_names.append("{}Avg".format(MetricShortNames[field_name]))
    for field_name in MetricFieldNames:
        field_names.append("{}Min".format(MetricShortNames[field_name]))
        field_names.append("{}Max".format(MetricShortNames[field_name]))

    row = {
        "ExperimentCode": _make_experiment_code(experiment_suffix),
        "Task": task_name,
    }
    for field_name in MechanismFieldNames:
        row[MechanismShortNames[field_name]] = mechanism_values[field_name]
    for field_name in MetricFieldNames:
        row["{}Avg".format(MetricShortNames[field_name])] = _format_metric_value(aggregate[field_name]["average"])
    for field_name in MetricFieldNames:
        row["{}Min".format(MetricShortNames[field_name])] = _format_metric_value(aggregate[field_name]["minimum"])
        row["{}Max".format(MetricShortNames[field_name])] = _format_metric_value(aggregate[field_name]["maximum"])

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(row)


def write_metric_reports(output_dir, title, metric_rows):
    # This wrapper writes all structure-metric report files to the requested output directory.
    # The text reports and compact CSV are generated together for each metric collection.
    os.makedirs(output_dir, exist_ok=True)
    write_metric_definitions(os.path.join(output_dir, "structure_metric_definitions.txt"), title)
    write_metric_run_report(metric_rows, os.path.join(output_dir, "structure_metrics_runs.txt"), title)
    write_metric_summary_report(metric_rows, os.path.join(output_dir, "structure_metrics_summary.txt"), title)
    write_metric_summary_csv(metric_rows, os.path.join(output_dir, "structure_metrics_summary_compact.csv"), output_dir)
