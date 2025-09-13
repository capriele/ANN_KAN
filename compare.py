import os
import re
import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"
output_dir = "comparative_graphs"

os.makedirs(output_dir, exist_ok=True)

network_types = ["ann", "kan", "kan_koopman", "koopman", "mamba"]


def extract_values(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        fit_matches = re.findall(r"fit:\s*([\d.]+)", content)
        nrmse_matches = re.findall(r"NRMSE:\s*([\d.]+)", content)
        fit = float(fit_matches[-1]) if fit_matches else 0
        nrmse = float(nrmse_matches[-1]) if nrmse_matches else 0
        return fit, nrmse


mean_fit = {network: [] for network in network_types}
mean_nrmse = {network: [] for network in network_types}

for network in network_types:
    for experiment in os.listdir(os.path.join(results_dir, network)):
        if os.path.isdir(os.path.join(results_dir, network, experiment)):
            log_path = os.path.join(results_dir, network, experiment, "log.txt")
            if os.path.exists(log_path):
                fit, nrmse = extract_values(log_path)
                mean_fit[network].append(fit)
                mean_nrmse[network].append(nrmse)

mean_fit_values = [round(np.mean(mean_fit[network]), 3) for network in network_types]
mean_nrmse_values = [
    round(np.mean(mean_nrmse[network]), 3) for network in network_types
]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].bar(network_types, mean_fit_values, color="blue")
ax[0].set_title("Mean Fit Comparison")
ax[0].set_ylabel("Mean Fit Value")
for i, v in enumerate(mean_fit_values):
    ax[0].text(i, v, str(v), ha="center", va="bottom")

ax[1].bar(network_types, mean_nrmse_values, color="orange")
ax[1].set_title("Mean NRMSE Comparison")
ax[1].set_ylabel("Mean NRMSE Value")
for i, v in enumerate(mean_nrmse_values):
    ax[1].text(i, v, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_comparison.png"))
plt.close()

for experiment in os.listdir(os.path.join(results_dir, network_types[0])):
    if os.path.isdir(os.path.join(results_dir, network_types[0], experiment)):
        experiment_data = {network: {"fit": 0, "nrmse": 0} for network in network_types}

        print(experiment)

        for network in network_types:
            log_path = os.path.join(results_dir, network, experiment, "log.txt")
            if os.path.exists(log_path):
                fit, nrmse = extract_values(log_path)
                experiment_data[network]["fit"] = fit
                experiment_data[network]["nrmse"] = nrmse

        networks = list(experiment_data.keys())
        fit_values = [round(experiment_data[network]["fit"], 3) for network in networks]
        nrmse_values = [
            round(experiment_data[network]["nrmse"], 3) for network in networks
        ]

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].bar(networks, fit_values, color="blue")
        ax[0].set_title(f"Fit Comparison for {experiment}")
        ax[0].set_ylabel("Fit Value")
        for i, v in enumerate(fit_values):
            ax[0].text(i, v, str(v), ha="center", va="bottom")

        ax[1].bar(networks, nrmse_values, color="orange")
        ax[1].set_title(f"NRMSE Comparison for {experiment}")
        ax[1].set_ylabel("NRMSE Value")
        for i, v in enumerate(nrmse_values):
            ax[1].text(i, v, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{experiment}_comparison.png"))
        plt.close()
print(
    "Comparative bar graphs and mean comparison graph generated and saved in the 'comparative_graphs' folder."
)
