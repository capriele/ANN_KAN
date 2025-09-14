import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

results_dir = "results"
output_dir = "comparative_graphs"
os.makedirs(output_dir, exist_ok=True)
network_types = ["ann", "kan", "kan_koopman", "koopman", "mamba"]

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

all_fit_data = {network: [] for network in network_types}
all_nrmse_data = {network: [] for network in network_types}

for experiment in os.listdir(os.path.join(results_dir, network_types[0])):
    print(experiment)
    experiment_path = os.path.join(results_dir, network_types[0], experiment)
    if os.path.isdir(experiment_path):
        log_path = os.path.join(experiment_path, "log.txt")
        if os.path.exists(log_path):
            multi_harmonic, reset_every = get_experiment_type(log_path)
            fig, axes = plt.subplots(1, 2, figsize=(25, 8))
            fig.suptitle(f"{experiment}", fontsize=14)
            colors = ["blue", "red", "green", "purple", "orange"]
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
                            i + idx * (1 / 5) - 2 / 5,
                            v,
                            color=color,
                            width=0.15,
                            label=f"{network}" if i == 0 else "",
                        )
                        if i == best_fit_index_overall and network == best_fit_network:
                            bar[0].set_edgecolor("black")
                            bar[0].set_linewidth(3)
                        axes[0].text(
                            i + idx * (1 / 5) - 2 / 5,
                            v + 0.02 * max(fit),
                            str(round(v, 3)),
                            ha="center",
                            va="top",
                            fontsize=8,
                        )
                    # Plot NRMSE values
                    for i, v in enumerate(nrmse):
                        color = colors[idx % len(colors)]
                        bar = axes[1].bar(
                            i + idx * (1 / 5) - 2 / 5,
                            v,
                            color=color,
                            width=0.15,
                            label=f"{network}" if i == 0 else "",
                        )
                        if (
                            i == best_nrmse_index_overall
                            and network == best_nrmse_network
                        ):
                            bar[0].set_edgecolor("black")
                            bar[0].set_linewidth(3)
                        axes[1].text(
                            i + idx * (1 / 5) - 2 / 5,
                            v + 0.02 * max(nrmse),
                            str(round(v, 3)),
                            ha="center",
                            va="top",
                            fontsize=8,
                        )
                    # Add vertical dashed line after all bars for this experiment
                    x_pos = (idx + 1) - 0.5  # Position between last and next experiment
                    axes[0].axvline(
                        x=x_pos, color="black", linestyle="--", linewidth=0.8
                    )
                    axes[1].axvline(
                        x=x_pos, color="black", linestyle="--", linewidth=0.8
                    )
            # Set x-axis labels based on multi_harmonic and reset_every
            try:
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

x = np.arange(len(network_types))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

mean_fit = [np.mean(all_fit_data[network]) for network in network_types]
std_fit = [np.std(all_fit_data[network]) for network in network_types]

bars_fit = axes[0].bar(x, mean_fit, width, label="Mean Fit", color="blue")
axes[0].errorbar(
    x + width / 2, mean_fit, yerr=std_fit, fmt="none", color="black", capsize=5
)
for bar in bars_fit:
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )
axes[0].set_title("Mean and Std of Fit Values")
axes[0].set_xticks(x)
axes[0].set_xticklabels(network_types)
axes[0].legend(loc="lower right")

mean_nrmse = [np.mean(all_nrmse_data[network]) for network in network_types]
std_nrmse = [np.std(all_nrmse_data[network]) for network in network_types]

bars_nrmse = axes[1].bar(x, mean_nrmse, width, label="Mean NRMSE", color="orange")
axes[1].errorbar(
    x + width / 2, mean_nrmse, yerr=std_nrmse, fmt="none", color="black", capsize=5
)
for bar in bars_nrmse:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
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
plt.savefig(os.path.join(output_dir, "mean_std_comparison.png"))
plt.close()
print(
    "Individual experiment comparison images and mean/std summary image generated and saved in the 'comparative_graphs' folder."
)
