import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

results_dir = "./"
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)
network_types = ["mixed"]


def extract_values(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    fit_matches = re.findall(r"fit:\s*([\d.e+-]+)", content)
    nrmse_matches = re.findall(r"NRMSE:\s*([\d.e+-]+)", content)
    fit = []
    nrmse = []
    for i, f in enumerate(fit_matches):
        if i % 2 == 1:
            try:
                fit.append(float(f))
            except ValueError:
                pass
    for i, n in enumerate(nrmse_matches):
        if i % 2 == 1:
            try:
                nrmse.append(float(n))
            except ValueError:
                pass
    if len(fit) == 0:
        fit = [0]
    if len(nrmse) == 0:
        nrmse = [0]
    return fit, nrmse


def get_experiment_type(log_path):
    with open(log_path, "r") as file:
        content = file.read()
    multi_harmonic = re.findall(r"validationOnMultiHarmonic:\s*(True|False)", content)
    reset_every_match = re.findall(r"reset every:\s*(-?\d+)", content)
    reset_every = reset_every_match if reset_every_match else []
    return (multi_harmonic, reset_every)


def merge_experiment_data(experiment_prefix):
    """Merge data from experiments following the naming convention (0_0 to 1_0)"""
    all_fit = {}
    all_nrmse = {}
    x_labels = {}
    result = {}

    for i in range(10):  # 0_0 to 0_9
        exp_name = f"{experiment_prefix}0_{i}"
        exp_path = os.path.join(results_dir, network_types[0], exp_name)
        if os.path.isdir(exp_path):
            log_path = os.path.join(exp_path, "log.txt")
            if os.path.exists(log_path):
                fit, nrmse = extract_values(log_path)
                all_fit[i] = fit
                all_nrmse[i] = nrmse
                multi_harmonic, reset_every = get_experiment_type(log_path)
                x_labels[i] = [
                    f"Multi Harmonic: {m}\nReset Every: {r}"
                    for m, r in zip(multi_harmonic, reset_every)
                ]
                result[i] = (all_fit[i], all_nrmse[i], x_labels[i])

    # Add 1_0 experiment if it exists
    exp_name = f"{experiment_prefix}1_0"
    exp_path = os.path.join(results_dir, network_types[0], exp_name)
    if os.path.isdir(exp_path):
        log_path = os.path.join(exp_path, "log.txt")
        if os.path.exists(log_path):
            fit, nrmse = extract_values(log_path)
            all_fit[10] = fit
            all_nrmse[10] = nrmse
            multi_harmonic, reset_every = get_experiment_type(log_path)
            x_labels[10] = [
                f"Multi Harmonic: {m}\nReset Every: {r}"
                for m, r in zip(multi_harmonic, reset_every)
            ]
            result[10] = (all_fit[10], all_nrmse[10], x_labels[10])

    return result


all_fit_data = {network: [] for network in network_types}
all_nrmse_data = {network: [] for network in network_types}

# First process merged experiments (those following the naming convention)
steps = ["0_0"]
experiment_prefix = None
for experiment in os.listdir(os.path.join(results_dir, network_types[0])):
    if re.match(r".*_\d_\d$", experiment):  # Matches patterns like "exp_0_0"
        experiment_prefix = (
            experiment.rsplit("_", 2)[0] + "_"
        )  # Gets "exp_" from "exp_0_0"
if experiment_prefix:
    for step in steps:
        experiment = experiment_prefix + step
        print(f"Processing merged experiment: {experiment}")
        component_data = merge_experiment_data(experiment_prefix)

        # Create a figure for each component (0-10)
        for component in range(len(component_data[0][0])):  # for all the elements
            print(component)
            print(len(component_data))
            print(component_data[0][0])
            fit, nrmse, x_labels = [], [], []
            for k in range(len(component_data)):  # 0 to 10
                # print(component_data[k])
                # print(component_data[k][0])
                # print(component, component_data[k][0][component])
                try:
                    a = component_data[k][0][component]
                    b = component_data[k][1][component]
                    c = component_data[k][2][component]
                    fit.append(a)
                    nrmse.append(b)
                    x_labels.append(c)
                except:
                    pass

            fit = list(reversed(fit))
            nrmse = list(reversed(nrmse))
            x_labels = list(reversed(x_labels))

            # print(fit)

            fig, axes = plt.subplots(1, 2, figsize=(25, 8))
            fig.suptitle(
                f"{experiment_prefix[:-1]} (Component {component})", fontsize=14
            )

            colors = ["blue", "red", "green", "purple", "orange", "cyan"]
            best_fit_overall = -float("inf")
            best_nrmse_overall = float("inf")
            best_fit_network = None
            best_nrmse_network = None
            best_fit_index_overall = -1
            best_nrmse_index_overall = -1

            # Find best overall fit and NRMSE for this component
            if len(fit) > 0:
                best_fit_overall = max(fit)
                best_fit_index_overall = fit.index(best_fit_overall)
                best_nrmse_overall = min(nrmse)
                best_nrmse_index_overall = nrmse.index(best_nrmse_overall)

            # Plot fit values
            for i, v in enumerate(fit):
                color = colors[0]
                bar = axes[0].bar(
                    i,
                    v,
                    color=color,
                    width=0.1,
                    label=str(float(i / 10)),
                )
                axes[0].text(
                    i,
                    v + 0.02 * max(fit),
                    str(round(v, 3)),
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white",
                        alpha=1,
                        edgecolor="none",
                        boxstyle="round,pad=0.2",
                    ),
                )

            # Plot NRMSE values
            for i, v in enumerate(nrmse):
                color = colors[0]
                bar = axes[1].bar(
                    i,
                    v,
                    color=color,
                    width=0.1,
                    label=str(float(i / 10)),
                )
                axes[1].text(
                    i,
                    v + 0.02 * max(nrmse),
                    str(round(v, 3)),
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white",
                        alpha=1,
                        edgecolor="none",
                        boxstyle="round,pad=0.2",
                    ),
                )

            # Add vertical dashed line after all bars for this experiment
            x_pos = len(fit) - 0.5  # Position after last bar
            axes[0].axvline(x=x_pos, color="black", linestyle="--", linewidth=0.8)
            axes[1].axvline(x=x_pos, color="black", linestyle="--", linewidth=0.8)

            # Set x-axis labels
            # if len(x_labels) > 0:
            #     axes[0].set_xticks(range(len(x_labels)))
            #     axes[0].set_xticklabels(x_labels, rotation=0, ha="center")
            #     axes[1].set_xticks(range(len(x_labels)))
            #     axes[1].set_xticklabels(x_labels, rotation=0, ha="center")

            labels = []
            for k in range(len(x_labels)):
                labels.append(str(float(k / 10)))

            axes[0].set_xticks(range(len(x_labels)))
            axes[0].set_xticklabels(labels, rotation=0, ha="center")
            axes[1].set_xticks(range(len(x_labels)))
            axes[1].set_xticklabels(labels, rotation=0, ha="center")

            axes[0].set_title("Fit Values")
            axes[0].set_ylabel("Fit Value")
            axes[0].set_xlabel("Alpha Value")
            axes[1].set_title("NRMSE Values")
            axes[1].set_ylabel("NRMSE Value")
            axes[1].set_xlabel("Alpha Value")
            # axes[0].legend(loc="lower right")
            # axes[1].legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{experiment_prefix[:-1]}_alpha_0_{component}.png",
                )
            )
            plt.close()

            # Add to overall data
            all_fit_data[network_types[0]].extend(fit)
            all_nrmse_data[network_types[0]].extend(nrmse)

        # Also create a combined plot with all components
        fig, axes = plt.subplots(1, 2, figsize=(25, 8))
        fig.suptitle(f"{experiment_prefix[:-1]} (All Components Combined)", fontsize=14)

        colors = [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "cyan",
            "magenta",
            "brown",
            "pink",
            "gray",
            "olive",
        ]
        all_fits = []
        all_nrmses = []
        component_labels = []

        # Collect all data and create labels
        for component in range(11):
            if component in component_data:
                fit, nrmse, x_labels = component_data[component]
                all_fits.extend(fit)
                all_nrmses.extend(nrmse)
                component_labels.extend([f"Comp {component}" for _ in range(len(fit))])

        # Plot combined fit values
        for i, v in enumerate(all_fits):
            component = i % 11  # Cycle through colors based on component
            color = colors[component % len(colors)]
            bar = axes[0].bar(
                i,
                v,
                color=color,
                width=0.8,
                label=(
                    f"Component {component}" if i < 11 else ""
                ),  # Only label first occurrence of each component
            )
            axes[0].text(
                i,
                v + 0.02 * max(all_fits),
                str(round(v, 3)),
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    alpha=1,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )

        # Plot combined NRMSE values
        for i, v in enumerate(all_nrmses):
            component = i % 11  # Cycle through colors based on component
            color = colors[component % len(colors)]
            bar = axes[1].bar(
                i,
                v,
                color=color,
                width=0.8,
                label=(
                    f"Component {component}" if i < 11 else ""
                ),  # Only label first occurrence of each component
            )
            axes[1].text(
                i,
                v + 0.02 * max(all_nrmses),
                str(round(v, 3)),
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    alpha=1,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )

        # Set x-axis labels for combined plot
        if len(component_labels) > 0:
            axes[0].set_xticks(range(len(component_labels)))
            axes[0].set_xticklabels(
                component_labels, rotation=45, ha="right", fontsize=8
            )
            axes[1].set_xticks(range(len(component_labels)))
            axes[1].set_xticklabels(
                component_labels, rotation=45, ha="right", fontsize=8
            )

        axes[0].set_title("Combined Fit Values")
        axes[0].set_ylabel("Fit Value")
        axes[1].set_title("Combined NRMSE Values")
        axes[1].set_ylabel("NRMSE Value")

        # Create custom legend with one entry per component
        handles, labels = axes[0].get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        seen_labels = set()

        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                seen_labels.add(label)
                unique_labels.append(label)
                unique_handles.append(handle)

        axes[0].legend(
            unique_handles,
            unique_labels,
            loc="upper right",
            bbox_to_anchor=(1.2, 1),
        )
        axes[1].legend(
            unique_handles,
            unique_labels,
            loc="upper right",
            bbox_to_anchor=(1.2, 1),
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"{experiment_prefix[:-1]}_all_components_combined.png"
            )
        )
        plt.close()

# Then process individual experiments (those not following the naming convention)
for experiment in os.listdir(os.path.join(results_dir, network_types[0])):
    if not re.match(r".*_\d_\d$", experiment):  # Skip merged experiments
        experiment_path = os.path.join(results_dir, network_types[0], experiment)
        if os.path.isdir(experiment_path):
            print(f"Processing individual experiment: {experiment}")
            log_path = os.path.join(experiment_path, "log.txt")
            if os.path.exists(log_path):
                multi_harmonic, reset_every = get_experiment_type(log_path)
                fig, axes = plt.subplots(1, 2, figsize=(25, 8))
                fig.suptitle(f"{experiment}", fontsize=14)
                colors = ["blue", "red", "green", "purple", "orange", "cyan"]
                best_fit_overall = -float("inf")
                best_nrmse_overall = float("inf")
                best_fit_network = None
                best_nrmse_network = None
                best_fit_index_overall = -1
                best_nrmse_index_overall = -1

                # First pass: find best overall fit and NRMSE
                for idx, network in enumerate(network_types):
                    log_path = os.path.join(results_dir, network, experiment, "log.txt")
                    if os.path.exists(log_path):
                        fit, nrmse = extract_values(log_path)
                        all_fit_data[network].extend(fit)
                        all_nrmse_data[network].extend(nrmse)
                        # Find best fit and NRMSE for this network
                        best_fit_index = fit.index(max(fit))
                        best_nrmse_index = nrmse.index(min(nrmse))
                        # Update best overall fit and NRMSE
                        if max(fit) > best_fit_overall:
                            best_fit_overall = max(fit)
                            best_fit_network = network
                            best_fit_index_overall = best_fit_index
                        if min(nrmse) < best_nrmse_overall:
                            best_nrmse_overall = min(nrmse)
                            best_nrmse_network = network
                            best_nrmse_index_overall = best_nrmse_index

                # Second pass: plot bars
                for idx, network in enumerate(network_types):
                    log_path = os.path.join(results_dir, network, experiment, "log.txt")
                    if os.path.exists(log_path):
                        fit, nrmse = extract_values(log_path)
                        # Plot fit values
                        for i, v in enumerate(fit):
                            color = colors[idx % len(colors)]
                            bar = axes[0].bar(
                                i + idx * (1 / 6) - 3 / 6,
                                v,
                                color=color,
                                width=0.1,
                                label=f"{network}" if i == 0 else "",
                            )
                            if (
                                i == best_fit_index_overall
                                and network == best_fit_network
                            ):
                                bar[0].set_edgecolor("black")
                                bar[0].set_linewidth(3)
                            axes[0].text(
                                i + idx * (1 / 6) - 3 / 6,
                                v + 0.02 * max(fit),
                                str(round(v, 3)),
                                ha="center",
                                va="top",
                                fontsize=8,
                                bbox=dict(
                                    facecolor="white",
                                    alpha=1,
                                    edgecolor="none",
                                    boxstyle="round,pad=0.2",
                                ),
                            )
                        # Plot NRMSE values
                        for i, v in enumerate(nrmse):
                            color = colors[idx % len(colors)]
                            bar = axes[1].bar(
                                i + idx * (1 / 6) - 3 / 6,
                                v,
                                color=color,
                                width=0.1,
                                label=f"{network}" if i == 0 else "",
                            )
                            if (
                                i == best_nrmse_index_overall
                                and network == best_nrmse_network
                            ):
                                bar[0].set_edgecolor("black")
                                bar[0].set_linewidth(3)
                            axes[1].text(
                                i + idx * (1 / 6) - 3 / 6,
                                v + 0.02 * max(nrmse),
                                str(round(v, 3)),
                                ha="center",
                                va="top",
                                fontsize=8,
                                bbox=dict(
                                    facecolor="white",
                                    alpha=1,
                                    edgecolor="none",
                                    boxstyle="round,pad=0.2",
                                ),
                            )
                        # Add vertical dashed line after all bars for this experiment
                        x_pos = (
                            idx + 0.5 - 0.1 + 0.1 / 4
                        )  # Position between last and next experiment
                        axes[0].axvline(
                            x=x_pos, color="black", linestyle="--", linewidth=0.8
                        )
                        axes[1].axvline(
                            x=x_pos, color="black", linestyle="--", linewidth=0.8
                        )

                # Set x-axis labels based on multi_harmonic and reset_every
                try:
                    log_path = os.path.join(
                        results_dir, network_types[0], experiment, "log.txt"
                    )
                    if os.path.exists(log_path):
                        fit, nrmse = extract_values(log_path)
                        multi_harmonic, reset_every = get_experiment_type(log_path)
                        x_labels = [
                            f"Multi Harmonic: {multi_harmonic[i]}\nReset Every: {reset_every[i]}"
                            for i in range(len(fit))
                        ]
                        axes[0].set_xticks(range(len(fit)))
                        axes[0].set_xticklabels(x_labels, rotation=0, ha="center")
                        axes[1].set_xticks(range(len(nrmse)))
                        axes[1].set_xticklabels(x_labels, rotation=0, ha="center")
                except:
                    pass

                axes[0].set_title("Fit Values")
                axes[0].set_ylabel("Fit Value")
                axes[1].set_title("NRMSE Values")
                axes[1].set_ylabel("NRMSE Value")
                axes[0].legend(loc="lower right")
                axes[1].legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{experiment}_comparison.png"))
                plt.close()

# Create final summary plot with std bars
x = np.arange(len(network_types))
width = 0.35
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

mean_fit = [np.mean(all_fit_data[network]) for network in network_types]
std_fit = [np.std(all_fit_data[network]) for network in network_types]

bars_fit = axes[0].bar(x, mean_fit, width, label="Mean Fit", color="blue", capsize=5)
axes[0].errorbar(x, mean_fit, yerr=std_fit, fmt="none", color="black", capsize=5)

for bar in bars_fit:
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

axes[0].set_title("Mean and Std of Fit Values")
axes[0].set_xticks(x)
axes[0].set_xticklabels(network_types)
axes[0].legend(loc="lower right")

mean_nrmse = [np.mean(all_nrmse_data[network]) for network in network_types]
std_nrmse = [np.std(all_nrmse_data[network]) for network in network_types]

bars_nrmse = axes[1].bar(
    x, mean_nrmse, width, label="Mean NRMSE", color="orange", capsize=5
)
axes[1].errorbar(x, mean_nrmse, yerr=std_nrmse, fmt="none", color="black", capsize=5)

for bar in bars_nrmse:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

best_fit_index = mean_fit.index(max(mean_fit))
bars_fit[best_fit_index].set_edgecolor("black")
bars_fit[best_fit_index].set_linewidth(2)
best_nrmse_index = mean_nrmse.index(min(mean_nrmse))
bars_nrmse[best_nrmse_index].set_edgecolor("black")
bars_nrmse[best_nrmse_index].set_linewidth(2)

axes[1].set_title("Mean and Std of NRMSE Values")
axes[1].set_xticks(x)
axes[1].set_xticklabels(network_types)
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_std_comparison_with_error_bars.png"))
plt.close()

print(
    "Individual experiment comparison images, merged experiment images, and mean/std summary image with error bars generated and saved in the 'graphs' folder."
)
