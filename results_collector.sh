#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Uso:
  ./raccogli_risultati.sh FILE_COMANDI [CARTELLA_OUTPUT]

Esempio:
  ./raccogli_risultati.sh analisi.sh risultati_raccolti

Lo script legge le righe contenenti comandi del tipo:
  python3 analyze_result.py \
    --results-dir ./results_auv_model \
    --model-kind chebyshev_kan \
    --experiment-name AUV

Per ogni comando cerca e copia, quando esistono:
  <results-dir>/<model-kind>/<experiment-name>
  <results-dir>/<model-kind>/<experiment-name>_1
  ...
  <results-dir>/<model-kind>/<experiment-name>_9

La destinazione avra' la struttura:
  <cartella-output>/<nome-results-dir>/<model-kind>/<experiment-name[_N]>

Alla fine viene creato:
  <cartella-output>.zip
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

COMMAND_FILE=$1
OUTPUT_DIR=${2:-cartella_nuova}

if [[ ! -f "$COMMAND_FILE" ]]; then
    printf 'Errore: il file dei comandi non esiste: %s\n' "$COMMAND_FILE" >&2
    exit 1
fi

# Rende assoluto il percorso del file, cosi' i --results-dir relativi vengono
# risolti rispetto alla cartella in cui si trova lo script dei comandi.
COMMAND_FILE=$(realpath "$COMMAND_FILE")
COMMAND_BASE_DIR=$(dirname "$COMMAND_FILE")

# Mantiene la destinazione relativa alla directory corrente dell'utente.
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
ZIP_FILE="${OUTPUT_DIR%/}.zip"

rm -rf -- "$OUTPUT_DIR"
rm -f -- "$ZIP_FILE"
mkdir -p -- "$OUTPUT_DIR"

copied=0
missing=0
commands=0

declare -A seen

extract_option() {
    local line=$1
    local option=$2

    # Supporta sia "--opzione valore" sia "--opzione=valore".
    sed -nE \
        -e "s/.*${option}[[:space:]]*=[[:space:]]*([^[:space:]]+).*/\\1/p" \
        -e "s/.*${option}[[:space:]]+([^[:space:]]+).*/\\1/p" \
        <<< "$line" | head -n 1
}

# Unisce eventuali comandi Bash spezzati su piu' righe con '\'.
while IFS= read -r command_line; do
    [[ "$command_line" =~ ^[[:space:]]*# ]] && continue
    [[ "$command_line" != *"python3"* ]] && continue
    [[ "$command_line" != *"--results-dir"* ]] && continue
    [[ "$command_line" != *"--model-kind"* ]] && continue
    [[ "$command_line" != *"--experiment-name"* ]] && continue

    results_dir=$(extract_option "$command_line" '--results-dir')
    model_kind=$(extract_option "$command_line" '--model-kind')
    experiment_name=$(extract_option "$command_line" '--experiment-name')

    if [[ -z "$results_dir" || -z "$model_kind" || -z "$experiment_name" ]]; then
        printf 'Avviso: comando non riconosciuto, ignorato:\n  %s\n' "$command_line" >&2
        continue
    fi

    ((commands += 1))

    # Rimuove eventuali virgolette semplici o doppie attorno ai valori.
    results_dir=${results_dir#\"}; results_dir=${results_dir%\"}
    results_dir=${results_dir#\'}; results_dir=${results_dir%\'}
    model_kind=${model_kind#\"}; model_kind=${model_kind%\"}
    model_kind=${model_kind#\'}; model_kind=${model_kind%\'}
    experiment_name=${experiment_name#\"}; experiment_name=${experiment_name%\"}
    experiment_name=${experiment_name#\'}; experiment_name=${experiment_name%\'}

    if [[ "$results_dir" = /* ]]; then
        source_base=$results_dir
    else
        source_base="$COMMAND_BASE_DIR/$results_dir"
    fi
    source_base=$(realpath -m "$source_base")

    base_folder=$(basename "$source_base")

    for suffix in '' _{1..9}; do
        experiment_dir="${experiment_name}${suffix}"
        source_dir="$source_base/$model_kind/$experiment_dir"
        destination_dir="$OUTPUT_DIR/$base_folder/$model_kind/$experiment_dir"
        key="$source_dir|$destination_dir"

        # Evita copie duplicate quando piu' righe indicano la stessa sorgente.
        [[ -n "${seen[$key]+x}" ]] && continue
        seen[$key]=1

        if [[ -d "$source_dir" ]]; then
            mkdir -p -- "$(dirname "$destination_dir")"
            cp -a -- "$source_dir" "$destination_dir"
            printf 'Copiato: %s -> %s\n' "$source_dir" "$destination_dir"
            ((copied += 1))
        else
            printf 'Non trovato: %s\n' "$source_dir" >&2
            ((missing += 1))
        fi
    done
done < <(
    awk '
        {
            sub(/\r$/, "")
            if (buffer == "") {
                buffer = $0
            } else {
                buffer = buffer " " $0
            }

            if (buffer ~ /\\[[:space:]]*$/) {
                sub(/\\[[:space:]]*$/, "", buffer)
                next
            }

            print buffer
            buffer = ""
        }
        END {
            if (buffer != "") print buffer
        }
    ' "$COMMAND_FILE"
)

if (( commands == 0 )); then
    printf 'Errore: nessun comando compatibile trovato in %s\n' "$COMMAND_FILE" >&2
    rm -rf -- "$OUTPUT_DIR"
    exit 1
fi

# Crea lo ZIP senza includere percorsi assoluti.
output_parent=$(dirname "$OUTPUT_DIR")
output_name=$(basename "$OUTPUT_DIR")
(
    cd "$output_parent"
    zip -qr "$ZIP_FILE" "$output_name"
)

printf '\nCompletato.\n'
printf 'Comandi analizzati: %d\n' "$commands"
printf 'Cartelle copiate:  %d\n' "$copied"
printf 'Cartelle mancanti: %d\n' "$missing"
printf 'Cartella finale:   %s\n' "$OUTPUT_DIR"
printf 'Archivio ZIP:      %s\n' "$ZIP_FILE"