#!/usr/bin/env python3

from pathlib import Path
import re
import argparse
import statistics
import csv
import math


FLOAT_PATTERN = r"(?:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan)"

FIT_REGEX = re.compile(
    rf"\bfit:\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)

NRMSE_REGEX = re.compile(
    rf"\bNRMSE:\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)

PARAMS_REGEX = re.compile(
    r"Total parameters:\s*([\d,]+)\s*,\s*Trainable:\s*([\d,]+)",
    re.IGNORECASE,
)


def natural_sort_key(path: Path):
    """
    Ordina in modo naturale.

    Esempio:
        experiment
        experiment_1
        experiment_2
        experiment_10

    invece di:
        experiment
        experiment_1
        experiment_10
        experiment_2
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def matches_experiment_name(path: Path, experiment_name: str | None) -> bool:
    """
    Se experiment_name è None, accetta tutte le cartelle.

    Se experiment_name='experiment', accetta:
        experiment
        experiment_1
        experiment_2
        ...

    Se experiment_name='test_A', accetta:
        test_A
        test_A_1
        test_A_2
        ...
    """
    if experiment_name is None:
        return True

    if path.name == experiment_name:
        return True

    return re.fullmatch(
        rf"{re.escape(experiment_name)}_\d+",
        path.name,
    ) is not None


def get_experiment_dirs(
    model_dir: Path,
    block_size: int = 10,
    experiment_name: str | None = None,
) -> list[Path]:
    """
    Recupera le prime N cartelle esperimento dentro:
        ./results/<model_kind>/

    Se experiment_name è specificato, filtra:
        <experiment_name>
        <experiment_name>_1
        <experiment_name>_2
        ...

    Se experiment_name non è specificato, prende le prime N cartelle trovate,
    ordinate in modo naturale.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Directory modello non trovata: {model_dir}"
        )

    experiment_dirs = [
        path
        for path in model_dir.iterdir()
        if path.is_dir()
        and matches_experiment_name(path, experiment_name)
    ]

    experiment_dirs = sorted(
        experiment_dirs,
        key=natural_sort_key,
    )

    return experiment_dirs[:block_size]


def extract_metrics_from_log(
    log_path: Path,
) -> tuple[float | None, float | None, int | None, int | None]:
    """
    Estrae fit, NRMSE, total parameters e trainable parameters dal log.txt.

    Se fit e NRMSE compaiono più volte, prende l'ultimo valore trovato.
    Riconosce anche il valore nan.
    """
    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    fit_matches = FIT_REGEX.findall(text)
    nrmse_matches = NRMSE_REGEX.findall(text)
    params_matches = PARAMS_REGEX.findall(text)

    fit = float(fit_matches[-1]) if fit_matches else None
    nrmse = float(nrmse_matches[-1]) if nrmse_matches else None

    if params_matches:
        total_params_str, trainable_params_str = params_matches[-1]

        total_params = int(
            total_params_str.replace(",", "")
        )
        trainable_params = int(
            trainable_params_str.replace(",", "")
        )
    else:
        total_params = None
        trainable_params = None

    return fit, nrmse, total_params, trainable_params


def is_valid_number(value: float | None) -> bool:
    """
    Restituisce True se il valore non è None e non è NaN.
    """
    return value is not None and not math.isnan(value)


def analyze_model(
    results_dir: Path,
    model_kind: str,
    block_size: int = 10,
    experiment_name: str | None = None,
) -> list[dict]:
    """
    Analizza gli esperimenti di uno specifico modello.

    Struttura attesa:
        ./results/<model_kind>/<experiment_folder>/log.txt
    """
    model_dir = results_dir / model_kind

    experiment_dirs = get_experiment_dirs(
        model_dir=model_dir,
        block_size=block_size,
        experiment_name=experiment_name,
    )

    rows = []

    for experiment_dir in experiment_dirs:
        log_path = experiment_dir / "log.txt"

        row = {
            "model_kind": model_kind,
            "experiment_group": (
                experiment_name
                if experiment_name is not None
                else experiment_dir.name
            ),
            "experiment": experiment_dir.name,
            "log_path": str(log_path),
            "fit": None,
            "NRMSE": None,
            "total_parameters": None,
            "trainable_parameters": None,
            "status": "missing",
        }

        if not log_path.exists():
            rows.append(row)
            continue

        try:
            fit, nrmse, total_params, trainable_params = (
                extract_metrics_from_log(log_path)
            )

            row["fit"] = fit
            row["NRMSE"] = nrmse
            row["total_parameters"] = total_params
            row["trainable_parameters"] = trainable_params

            if not is_valid_number(fit) or not is_valid_number(nrmse):
                row["status"] = "partial"
            else:
                row["status"] = "ok"

        except Exception as exc:
            row["status"] = f"error: {exc}"

        rows.append(row)

    return rows


def summarize(rows: list[dict]) -> dict:
    """
    Calcola media e deviazione standard per fit e NRMSE.

    Usa solo gli esperimenti che hanno sia fit sia NRMSE validi.
    I valori None e NaN vengono esclusi.
    """
    valid_rows = [
        row
        for row in rows
        if is_valid_number(row["fit"])
        and is_valid_number(row["NRMSE"])
    ]

    fits = [
        row["fit"]
        for row in valid_rows
    ]

    nrmses = [
        row["NRMSE"]
        for row in valid_rows
    ]

    return {
        "n_valid": len(valid_rows),
        "n_total": len(rows),

        "fit_mean": (
            statistics.mean(fits)
            if fits
            else None
        ),
        "fit_std": (
            statistics.stdev(fits)
            if len(fits) > 1
            else 0.0 if fits else None
        ),

        "NRMSE_mean": (
            statistics.mean(nrmses)
            if nrmses
            else None
        ),
        "NRMSE_std": (
            statistics.stdev(nrmses)
            if len(nrmses) > 1
            else 0.0 if nrmses else None
        ),
    }


def format_float(
    value: float | None,
    scientific: bool = False,
) -> str:
    """
    Formatta un valore numerico.

    Se il valore è None oppure NaN, restituisce:
        "0.0"
    """
    if value is None or math.isnan(value):
        return r"0.0"

    if scientific:
        return f"{value:.6e}"

    return str(value)


def save_csv(
    rows: list[dict],
    summary: dict,
    output_path: Path,
) -> None:
    """
    Salva un CSV con:
    - risultati dei singoli esperimenti
    - sezione finale summary,value

    I valori None o NaN di fit e NRMSE vengono scritti come:
        $0^{*}$
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model_kind",
        "experiment_group",
        "experiment",
        "log_path",
        "fit",
        "NRMSE",
        "total_parameters",
        "trainable_parameters",
        "status",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            output_row = row.copy()

            output_row["fit"] = format_float(
                row["fit"]
            )
            output_row["NRMSE"] = format_float(
                row["NRMSE"]
            )

            writer.writerow(output_row)

        f.write("\n")
        f.write("summary,value\n")

        for key, value in summary.items():
            if key in {
                "fit_mean",
                "fit_std",
                "NRMSE_mean",
                "NRMSE_std",
            }:
                value = format_float(
                    value,
                    scientific=key.endswith("_std"),
                )

            f.write(f"{key},{value}\n")


def print_report(
    rows: list[dict],
    summary: dict,
) -> None:
    print("\nRisultati esperimenti")
    print("-" * 150)

    for row in rows:
        print(
            f"{row['model_kind']:12s} | "
            f"{row['experiment_group']:25s} | "
            f"{row['experiment']:25s} | "
            f"fit={format_float(row['fit'])} | "
            f"NRMSE={format_float(row['NRMSE'])} | "
            f"total_params={row['total_parameters']} | "
            f"trainable={row['trainable_parameters']} | "
            f"{row['status']}"
        )

    print("\nSummary")
    print("-" * 150)

    print(
        f"Esperimenti validi: "
        f"{summary['n_valid']} / {summary['n_total']}"
    )

    print(
        f"fit media:          "
        f"{format_float(summary['fit_mean'])}"
    )

    print(
        f"fit std:            "
        f"{format_float(summary['fit_std'], scientific=True)}"
    )

    print(
        f"NRMSE media:        "
        f"{format_float(summary['NRMSE_mean'])}"
    )

    print(
        f"NRMSE std:          "
        f"{format_float(summary['NRMSE_std'], scientific=True)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analizza fit, NRMSE e parametri "
            "dai log degli esperimenti."
        )
    )

    parser.add_argument(
        "--results-dir",
        default="./results",
        help=(
            "Directory principale dei risultati. "
            "Default: ./results"
        ),
    )

    parser.add_argument(
        "--model-kind",
        required=True,
        help=(
            "Tipo di modello da analizzare, "
            "ad esempio: kan, mlp, rnn"
        ),
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Nome base dell'esperimento. "
            "Esempio: experiment analizza experiment, "
            "experiment_1, ..., experiment_9. "
            "Se non specificato, usa direttamente "
            "i nomi delle cartelle trovate."
        ),
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=10,
        help=(
            "Numero di cartelle esperimento da analizzare. "
            "Default: 10"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Percorso del CSV di output. "
            "Default: "
            "./results/<model_kind>/summary_metrics.csv"
        ),
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model_kind = args.model_kind
    experiment_name = args.experiment_name

    rows = analyze_model(
        results_dir=results_dir,
        model_kind=model_kind,
        block_size=args.block_size,
        experiment_name=experiment_name,
    )

    summary = summarize(rows)

    output_path = (
        Path(args.output)
        if args.output is not None
        else results_dir
        / model_kind
        / "summary_metrics.csv"
    )

    save_csv(
        rows=rows,
        summary=summary,
        output_path=output_path,
    )

    print_report(
        rows=rows,
        summary=summary,
    )

    print(f"\nCSV salvato in: {output_path}")


if __name__ == "__main__":
    main()