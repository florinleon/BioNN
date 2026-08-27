import itertools
import os
import time
import run_sparse_parity_metrics
import run_xor_metrics


RunXor = True
RunSparseParity = True

# Stage switches control which ablation families are included in the batch.
RunOnOff = False   # Stage 1: all_off and all_on.
RunOne = True     # Stage 2: one_on and one_off.
RunTwo = True     # Stage 3: two_on and two_off.
RunThree = False   # Stage 4: the four selected three_on leave-one-out experiments.
RunFour = False    # Stage 5: the selected four_on experiment.

SkipCompletedExperiments = True

MechanismNames = [
    "UseInputGate",
    "UseGainModulation",
    "UseThresholdModulation",
    "UseLateralInhibition",
    "UseHomeostasis",
    "UseStructuralPlasticity",
    "UseActivationDecorrelation",
]

MechanismAliases = {
    "UseInputGate": "input_gate",
    "UseGainModulation": "gain",
    "UseThresholdModulation": "threshold",
    "UseLateralInhibition": "inhibition",
    "UseHomeostasis": "homeostasis",
    "UseStructuralPlasticity": "structural",
    "UseActivationDecorrelation": "decorrelation",
}

RelevantMechanisms = [
    "UseInputGate",
    "UseLateralInhibition",
    "UseHomeostasis",
    "UseStructuralPlasticity",
]

IrrelevantMechanisms = [
    "UseGainModulation",
    "UseThresholdModulation",
    "UseActivationDecorrelation",
]

XorResultsTemplate = "bionn_xor_{}_results"
SparseParityResultsTemplate = "bionn_sparse_parity_{}_output"


def make_all_false_settings():
    # A fresh dictionary for every experiment prevents one run from leaking state into the next when a later experiment
    # flips some mechanisms. The order follows MechanismNames so printed summaries stay stable across reruns.
    settings = {}
    for mechanism_name in MechanismNames:
        settings[mechanism_name] = False
    return settings


def make_all_true_settings():
    # This constructor defines the fully enabled baseline explicitly instead of deriving it from
    # an all-false dictionary.
    settings = {}
    for mechanism_name in MechanismNames:
        settings[mechanism_name] = True
    return settings


def make_one_on_settings(enabled_mechanism):
    # In the one-on family, exactly one mechanism is active while every other switch stays at the plain baseline.
    settings = make_all_false_settings()
    settings[enabled_mechanism] = True
    return settings


def make_one_off_settings(disabled_mechanism):
    # In the one-off family, the system starts fully enabled and removes exactly one mechanism.
    settings = make_all_true_settings()
    settings[disabled_mechanism] = False
    return settings


def make_two_on_settings(enabled_mechanisms):
    # In the two-on family, only the listed mechanisms are active. Every other switch stays at the plain MLP baseline.
    settings = make_all_false_settings()
    for mechanism_name in enabled_mechanisms:
        settings[mechanism_name] = True
    return settings


def make_two_off_settings(disabled_mechanisms):
    # In the two-off family, we start from the fully enabled system and selectively remove exactly two mechanisms.
    settings = make_all_true_settings()
    for mechanism_name in disabled_mechanisms:
        settings[mechanism_name] = False
    return settings


def make_four_on_settings():
    # The reduced four-on reference keeps gain modulation, threshold modulation, and activation decorrelation
    # disabled while input gate, lateral inhibition, homeostasis, and structural plasticity remain enabled.
    settings = make_all_false_settings()
    for mechanism_name in RelevantMechanisms:
        settings[mechanism_name] = True
    return settings


def make_three_on_settings(disabled_relevant_mechanism):
    # The three-on configuration keeps gain modulation, threshold modulation, and activation decorrelation
    # disabled while exactly one of the four retained mechanisms is removed from the reduced four-on reference.
    settings = make_four_on_settings()
    settings[disabled_relevant_mechanism] = False
    return settings


def make_experiment_suffix(family_name, mechanism_names=None):
    # Result-directory suffixes include only the mechanisms that define each family member, which keeps names
    # concise and readable without relying on binary codes or index numbers.
    if mechanism_names is None:
        return family_name

    alias_parts = []
    for mechanism_name in mechanism_names:
        alias_parts.append(MechanismAliases[mechanism_name])
    return "{}__{}".format(family_name, "__".join(alias_parts))


def make_three_on_suffix(disabled_relevant_mechanism):
    # The suffix records which of the four retained mechanisms is turned off in the three-on family.
    alias = MechanismAliases[disabled_relevant_mechanism]
    return "three_on__without_{}".format(alias)


def iter_stage_one_experiments():
    # Stage 1: the two extreme references.
    if not RunOnOff:
        return

    yield make_experiment_suffix("all_off"), make_all_false_settings()
    yield make_experiment_suffix("all_on"), make_all_true_settings()


def iter_stage_two_experiments():
    # Stage 2: a single mechanism on, then a single mechanism off.
    if not RunOne:
        return

    for enabled_mechanism in MechanismNames:
        suffix = make_experiment_suffix("one_on", [enabled_mechanism])
        yield suffix, make_one_on_settings(enabled_mechanism)

    for disabled_mechanism in MechanismNames:
        suffix = make_experiment_suffix("one_off", [disabled_mechanism])
        yield suffix, make_one_off_settings(disabled_mechanism)


def iter_stage_three_experiments():
    # Stage 3: two mechanisms on, then two mechanisms off.
    if not RunTwo:
        return

    for enabled_mechanisms in itertools.combinations(MechanismNames, 2):
        suffix = make_experiment_suffix("two_on", enabled_mechanisms)
        yield suffix, make_two_on_settings(enabled_mechanisms)

    for disabled_mechanisms in itertools.combinations(MechanismNames, 2):
        suffix = make_experiment_suffix("two_off", disabled_mechanisms)
        yield suffix, make_two_off_settings(disabled_mechanisms)


def iter_stage_four_experiments():
    # Stage 4: four three-on leave-one-out experiments over the retained mechanisms.
    if not RunThree:
        return

    for disabled_relevant_mechanism in RelevantMechanisms:
        suffix = make_three_on_suffix(disabled_relevant_mechanism)
        yield suffix, make_three_on_settings(disabled_relevant_mechanism)


def iter_stage_five_experiments():
    # Stage 5: the reduced four-on reference.
    if not RunFour:
        return

    yield make_experiment_suffix("four_on"), make_four_on_settings()


def iter_experiments():
    # The complete batch follows the five stages in order.
    for experiment in iter_stage_one_experiments():
        yield experiment

    for experiment in iter_stage_two_experiments():
        yield experiment

    for experiment in iter_stage_three_experiments():
        yield experiment

    for experiment in iter_stage_four_experiments():
        yield experiment

    for experiment in iter_stage_five_experiments():
        yield experiment


def expected_output_directories(results_suffix):
    # Result directories use the task-specific naming templates defined above. This helper
    # lets the batch optionally skip combinations that have already produced result directories on disk.
    directories = []
    if RunXor:
        directories.append(XorResultsTemplate.format(results_suffix))
    if RunSparseParity:
        directories.append(SparseParityResultsTemplate.format(results_suffix))
    return directories


def experiment_already_completed(results_suffix):
    # A run counts as completed only if every requested task has already written its own top-level output directory.
    # This conservative rule avoids skipping combinations after a partially finished or interrupted batch.
    output_directories = expected_output_directories(results_suffix)
    if len(output_directories) == 0:
        return False
    for output_directory in output_directories:
        if not os.path.isdir(output_directory):
            return False
    return True


def run_one_experiment(results_suffix, mechanism_settings):
    # Each task runner expands one mechanism dictionary into its configured seeded runs, plots, and
    # metric files. This wrapper supplies the mechanism settings and result suffix for each batch experiment.
    if RunXor:
        run_xor_metrics.run_experiment(mechanism_settings, results_suffix)
    if RunSparseParity:
        run_sparse_parity_metrics.run_experiment(mechanism_settings, results_suffix)


def print_experiment_header(experiment_index, experiment_count, results_suffix, mechanism_settings):
    # A compact header marks where one experiment combination ends and the next begins in serial batch logs,
    # which makes later log inspection easier.
    print("=" * 100)
    print("Experiment {}/{}: {}".format(experiment_index, experiment_count, results_suffix))
    for mechanism_name in MechanismNames:
        print("  {} = {}".format(mechanism_name, mechanism_settings[mechanism_name]))


def count_stage_experiments():
    return {
        "Stage 1 - all on/off": len(list(iter_stage_one_experiments())),
        "Stage 2 - one on/off": len(list(iter_stage_two_experiments())),
        "Stage 3 - two on/off": len(list(iter_stage_three_experiments())),
        "Stage 4 - three on": len(list(iter_stage_four_experiments())),
        "Stage 5 - four on": len(list(iter_stage_five_experiments())),
    }


def print_batch_settings(experiment_count):
    print("Prepared {} experiment combinations.".format(experiment_count))
    print("RunXor = {}".format(RunXor))
    print("RunSparseParity = {}".format(RunSparseParity))
    print("RunOnOff = {}".format(RunOnOff))
    print("RunOne = {}".format(RunOne))
    print("RunTwo = {}".format(RunTwo))
    print("RunThree = {}".format(RunThree))
    print("RunFour = {}".format(RunFour))
    print("SkipCompletedExperiments = {}".format(SkipCompletedExperiments))
    print("Stage counts:")
    for stage_name, stage_count in count_stage_experiments().items():
        print("  {}: {}".format(stage_name, stage_count))

    if RunThree:
        print("Stage 4: three_on runs use IG, LI, HO, SP as the relevant set and leave out one of them.")
    if RunFour:
        print("Stage 5: four_on uses IG, LI, HO, SP. GM, TM, and AD are fixed off.")


def main():
    experiments = list(iter_experiments())
    batch_start_time = time.perf_counter()

    print_batch_settings(len(experiments))

    completed_experiment_count = 0
    skipped_experiment_count = 0
    for experiment_index, experiment in enumerate(experiments, start=1):
        results_suffix, mechanism_settings = experiment
        print_experiment_header(experiment_index, len(experiments), results_suffix, mechanism_settings)

        if SkipCompletedExperiments and experiment_already_completed(results_suffix):
            skipped_experiment_count += 1
            print("Skipping existing outputs for {}.".format(results_suffix))
            continue

        experiment_start_time = time.perf_counter()
        run_one_experiment(results_suffix, mechanism_settings)
        experiment_wall_clock_seconds = time.perf_counter() - experiment_start_time
        completed_experiment_count += 1
        print("Finished {} in {:.6f} seconds.".format(results_suffix, experiment_wall_clock_seconds))

    batch_wall_clock_seconds = time.perf_counter() - batch_start_time
    print("=" * 100)
    message = "Completed {} experiments and skipped {} experiments in {:.6f} seconds."
    print(message.format(completed_experiment_count, skipped_experiment_count, batch_wall_clock_seconds))


if __name__ == "__main__":
    main()
