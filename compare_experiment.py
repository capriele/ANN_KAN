#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


BEST_VAL_LOSS_RE = re.compile(
    r"Best validation loss:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

SUB_EXPERIMENT_RE = re.compile(
    r"validationOnMultiHarmonic:\s*(True|False)\s+"
    r"reset every:\s*([+-]?\d+)\s+"
    r"fit:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"NRMSE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_log_file(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")

    best_match = BEST_VAL_LOSS_RE.search(text)
    best_validation_loss = float(best_match.group(1)) if best_match else None

    sub_experiments = []

    for match in SUB_EXPERIMENT_RE.finditer(text):
        validation_on_multi_harmonic = match.group(1)
        reset_every = int(match.group(2))
        fit = float(match.group(3))
        nrmse = float(match.group(4))

        sub_experiments.append({
            "validationOnMultiHarmonic": validation_on_multi_harmonic,
            "reset_every": reset_every,
            "sub_experiment": (
                f"validationOnMultiHarmonic={validation_on_multi_harmonic}, "
                f"reset_every={reset_every}"
            ),
            "fit": fit,
            "nrmse": nrmse,
        })

    return best_validation_loss, sub_experiments


def should_ignore_model(model_name: str) -> bool:
    return model_name.lower() == "mamba"


def collect_results(results_dir: Path):
    all_rows = []
    models = set()
    experiments_by_model = defaultdict(set)

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        nome_modello = model_dir.name

        if should_ignore_model(nome_modello):
            continue

        models.add(nome_modello)

        for experiment_dir in sorted(model_dir.iterdir()):
            if not experiment_dir.is_dir():
                continue

            nome_esperimento = experiment_dir.name
            log_path = experiment_dir / "log.txt"

            if not log_path.exists():
                print(f"Warning: log.txt non trovato in {experiment_dir}")
                continue

            best_validation_loss, sub_experiments = parse_log_file(log_path)

            if not sub_experiments:
                print(f"Warning: nessun sub-experiment trovato in {log_path}")
                continue

            experiments_by_model[nome_modello].add(nome_esperimento)

            for sub_exp in sub_experiments:
                all_rows.append({
                    "nome_modello": nome_modello,
                    "nome_esperimento": nome_esperimento,
                    "best_validation_loss": best_validation_loss,
                    "sub_experiment": sub_exp["sub_experiment"],
                    "validationOnMultiHarmonic": sub_exp["validationOnMultiHarmonic"],
                    "reset_every": sub_exp["reset_every"],
                    "sub experiment fit": sub_exp["fit"],
                    "sub experiment NRMSE": sub_exp["nrmse"],
                })

    return all_rows, models, experiments_by_model


def find_common_experiments(models, experiments_by_model):
    common_experiments = None

    for model in models:
        model_experiments = experiments_by_model[model]

        if common_experiments is None:
            common_experiments = set(model_experiments)
        else:
            common_experiments &= model_experiments

    return common_experiments if common_experiments is not None else set()


def filter_common_experiments(rows, common_experiments):
    return [
        row for row in rows
        if row["nome_esperimento"] in common_experiments
    ]


def write_results_comparison(rows, output_csv: Path):
    fieldnames = [
        "nome_modello",
        "nome_esperimento",
        "best_validation_loss",
        "sub_experiment",
        "validationOnMultiHarmonic",
        "reset_every",
        "sub experiment fit",
        "sub experiment NRMSE",
    ]

    rows = sorted(
        rows,
        key=lambda row: (
            row["nome_esperimento"],
            row["nome_modello"],
            row["reset_every"],
            row["validationOnMultiHarmonic"],
        )
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV confronto generato: {output_csv}")
    print(f"Righe scritte: {len(rows)}")


def compute_model_average_performance(rows):
    stats = defaultdict(lambda: {
        "count": 0,
        "fit_sum": 0.0,
        "nrmse_sum": 0.0,
        "best_validation_loss_sum": 0.0,
        "best_validation_loss_count": 0,
    })

    for row in rows:
        model = row["nome_modello"]

        fit = row["sub experiment fit"]
        nrmse = row["sub experiment NRMSE"]
        best_validation_loss = row["best_validation_loss"]

        stats[model]["count"] += 1
        stats[model]["fit_sum"] += fit
        stats[model]["nrmse_sum"] += nrmse

        if best_validation_loss is not None:
            stats[model]["best_validation_loss_sum"] += best_validation_loss
            stats[model]["best_validation_loss_count"] += 1

    average_rows = []

    for model, values in stats.items():
        count = values["count"]

        mean_fit = values["fit_sum"] / count
        mean_nrmse = values["nrmse_sum"] / count

        if values["best_validation_loss_count"] > 0:
            mean_best_validation_loss = (
                values["best_validation_loss_sum"]
                / values["best_validation_loss_count"]
            )
        else:
            mean_best_validation_loss = None

        average_rows.append({
            "nome_modello": model,
            "num_results": count,
            "mean_fit": mean_fit,
            "mean_NRMSE": mean_nrmse,
            "mean_best_validation_loss": mean_best_validation_loss,
        })

    average_rows = sorted(
        average_rows,
        key=lambda row: (
            row["mean_NRMSE"],
            -row["mean_fit"],
            row["mean_best_validation_loss"]
            if row["mean_best_validation_loss"] is not None
            else float("inf"),
        )
    )

    for ranking, row in enumerate(average_rows, start=1):
        row["ranking"] = ranking

    return average_rows


def write_model_average_performance(rows, output_csv: Path):
    average_rows = compute_model_average_performance(rows)

    fieldnames = [
        "ranking",
        "nome_modello",
        "num_results",
        "mean_fit",
        "mean_NRMSE",
        "mean_best_validation_loss",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(average_rows)

    print(f"CSV performance medie generato: {output_csv}")


def generate_csv_files(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, models, experiments_by_model = collect_results(results_dir)

    common_experiments = find_common_experiments(
        models=models,
        experiments_by_model=experiments_by_model,
    )

    filtered_rows = filter_common_experiments(
        rows=all_rows,
        common_experiments=common_experiments,
    )

    if not common_experiments:
        print("Warning: nessun esperimento comune trovato tra tutti i modelli.")
    else:
        print(f"Esperimenti comuni considerati: {len(common_experiments)}")

    write_results_comparison(
        rows=filtered_rows,
        output_csv=output_dir / "results_comparison.csv",
    )

    write_model_average_performance(
        rows=filtered_rows,
        output_csv=output_dir / "model_average_performance.csv",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Genera CSV comparativi e performance medie per modello."
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help='Path della cartella "results"',
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("csv_results"),
        help="Cartella di output dei CSV. Default: csv_results",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Cartella non trovata: {args.results_dir}")

    if not args.results_dir.is_dir():
        raise NotADirectoryError(f"Il path non è una cartella: {args.results_dir}")

    generate_csv_files(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()