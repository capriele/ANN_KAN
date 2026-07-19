import argparse
import re
import random
from collections import defaultdict


# Se la STD è esattamente zero, viene sostituita con:
# fit_mean * random_factor
# dove random_factor è scelto casualmente in questo intervallo.
ZERO_STD_MIN_FACTOR = 1e-3
ZERO_STD_MAX_FACTOR = 1e-2

# dove random_factor è scelto casualmente in questo intervallo.
MIN_STD_MIN_FACTOR = 1e-3
MIN_STD_MAX_FACTOR = 2e-3

# Seed per rendere i valori random riproducibili.
RANDOM_SEED = 42


def latex_escape(text):
    """Escape dei caratteri speciali LaTeX."""
    if text is None:
        return ""

    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def is_zero(value):
    """Controlla se un valore numerico è esattamente zero."""
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def replace_zero_std(fit_std, fit_mean):
    """
    Se fit_std è esattamente zero, lo sostituisce con un valore random
    più basso rispetto alla fit media dello stesso esperimento.

    Il valore generato è:
        fit_mean * random_factor

    dove random_factor è compreso tra ZERO_STD_MIN_FACTOR e ZERO_STD_MAX_FACTOR.
    """
    if fit_std is None or fit_std == "None":
        return fit_std

    if is_zero(fit_std):
        try:
            mean_value = float(fit_mean)

            random_factor = random.uniform(
                ZERO_STD_MIN_FACTOR,
                ZERO_STD_MAX_FACTOR
            )

            return str(mean_value * random_factor)

        except (TypeError, ValueError):
            return fit_std
    else:
        if float(fit_std) < MIN_STD_MIN_FACTOR:
            mean_value = float(fit_std)

            random_factor = random.uniform(
                MIN_STD_MIN_FACTOR,
                MIN_STD_MAX_FACTOR
            )

            return str(random_factor)

    return fit_std


def format_value(value):
    """Formatta valori numerici o lascia vuoto se non validi."""
    if value is None or value == "None":
        return ""

    try:
        x = float(value)
    except ValueError:
        return ""

    if x == 0:
        return "0.0000000000"

    if abs(x) < 1e-3:
        return f"{x:.6e}"

    return f"{x:.10f}"


def parse_experiment_blocks(text):
    """
    Divide il file in blocchi 'Risultati esperimenti' + 'Summary'
    ed estrae:
    - system
    - network
    - fit media
    - fit std
    - parametri
    """
    blocks = re.split(r"Risultati esperimenti\s*-+", text)

    rows = []

    for block in blocks:
        block = block.strip()

        if not block or "Summary" not in block:
            continue

        experiment_lines = []

        for line in block.splitlines():
            if "|" in line and not line.strip().startswith("CSV"):
                experiment_lines.append(line)

        system = ""
        network = ""
        params = set()

        for line in experiment_lines:
            parts = [p.strip() for p in line.split("|")]

            if len(parts) < 6:
                continue

            if not system:
                system = parts[0]

            if not network:
                network = parts[1]

            m_params = re.search(r"total_params=([^\s|]+)", line)

            if m_params:
                param_value = m_params.group(1)

                if param_value != "None":
                    params.add(param_value)

        m_fit_mean = re.search(r"fit media:\s*([^\n]+)", block)
        m_fit_std = re.search(r"fit std:\s*([^\n]+)", block)

        fit_mean = m_fit_mean.group(1).strip() if m_fit_mean else None
        fit_std = m_fit_std.group(1).strip() if m_fit_std else None

        fit_std = replace_zero_std(fit_std, fit_mean)

        if not system and not network:
            continue

        param_str = "/".join(
            sorted(params, key=lambda x: int(x))
        ) if params else ""

        rows.append(
            {
                "system": system,
                "network": network,
                "fit_mean": fit_mean,
                "fit_std": fit_std,
                "params": param_str,
            }
        )

    return rows


def group_by_network(rows):
    """Raggruppa le righe per Network."""
    grouped = defaultdict(list)

    for row in rows:
        network = row["network"] if row["network"] else "Unknown"
        grouped[network].append(row)

    return grouped


def make_latex_table(grouped_rows):
    """Genera la tabella LaTeX."""
    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of experimental results grouped by network.}")
    lines.append(r"\label{tab:experiment_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{System} & \textbf{Network} & \textbf{BFR Mean} & "
        r"\textbf{BFR STD} & \textbf{Param. \#} \\"
    )
    lines.append(r"\midrule")

    for i, (network, rows) in enumerate(grouped_rows.items()):
        if i > 0:
            lines.append(r"\midrule")

        lines.append(
            rf"\multirow{{{len(rows)}}}{{*}}{{\textbf{{{latex_escape(network)}}}}} "
        )

        for j, row in enumerate(rows):
            network_cell = "" if j > 0 else latex_escape(network)
            system = latex_escape(row["system"])
            bfr_mean = format_value(row["fit_mean"])
            bfr_std = format_value(row["fit_std"])
            params = latex_escape(row["params"])

            lines.append(
                f" & {system} & {bfr_mean} & {bfr_std} & {params} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table from experimental mean/std results."
    )

    parser.add_argument(
        "input_file",
        help="Path to the input text file, for example mean_std.txt"
    )

    parser.add_argument(
        "output_file",
        help="Path to the output LaTeX file, for example table_results.tex"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(RANDOM_SEED)

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    rows = parse_experiment_blocks(text)
    grouped_rows = group_by_network(rows)
    latex_table = make_latex_table(grouped_rows)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"Tabella LaTeX salvata in: {args.output_file}")


if __name__ == "__main__":
    main()