import argparse
import os
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem
from agentic_pdm_data_cleaning.utils.utility import load_config
from pathlib import Path
import csv
import gzip
import bz2
import lzma


def count_jsonl_entries(path, ignore_blank=True, ignore_comments=True):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    with p.open("r", encoding="utf-8") as f:
        if ignore_blank or ignore_comments:
            return sum(
                1
                for line in f
                if (not ignore_blank or line.strip())
                and (not ignore_comments or not line.lstrip().startswith("#"))
            )
        else:
            return sum(1 for _ in f)


def _open_text_auto(path, encoding: str = "utf-8"):
    """
    Open plain or compressed text transparently with newline='' for csv.
    Supports .gz, .bz2, .xz; otherwise opens as a normal text file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding=encoding, newline="")
    if p.suffix in (".bz2", ".bzip2"):
        return bz2.open(p, "rt", encoding=encoding, newline="")
    if p.suffix in (".xz", ".lzma"):
        return lzma.open(p, "rt", encoding=encoding, newline="")
    return p.open("r", encoding=encoding, newline="")


def count_csv_records(path: str,
                      has_header: bool = True,
                      comment_char: str | None = None,
                      encoding: str = "utf-8") -> int:
    """
    Robustly count rows in a CSV using the csv module (handles embedded newlines).

    Args:
        path: Path to the CSV file (supports .gz/.bz2/.xz).
        has_header: If True, skips the first non-empty, non-comment row.
        comment_char: If set, skips rows whose first cell starts with this char.
        encoding: File encoding.

    Returns:
        int: Number of data records (rows).
    """
    count = 0
    header_skipped = not has_header
    with _open_text_auto(path, encoding=encoding) as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip blank rows
            if not row or all((c is None or (isinstance(c, str) and c.strip() == "")) for c in row):
                continue
            # Skip comment rows (e.g., lines where first cell starts with '#')
            if comment_char and isinstance(row[0], str) and row[0].lstrip().startswith(comment_char):
                continue
            if not header_skipped:
                header_skipped = True
                continue
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check benchmark results.")
    parser.add_argument("benchmark_id", type=str,
                        help="Id of the benchmark.")

    args = parser.parse_args()

    benchmark_id = args.benchmark_id
    fs = FileSystem()
    benchmark_config_path = fs.benchmark_config_path(benchmark_id)

    benchmark_conf = load_config(benchmark_config_path)
    models = benchmark_conf.get("models", [])
    experiments = benchmark_conf.get("experiments", {})
    # Get the first experiment name
    experiment_name = list(experiments.keys())[0]
    runs = benchmark_conf.get("runs", 1)

    print(
        f"Checking benchmark '{benchmark_id}' with models {models} and experiment '{experiment_name}'")

    for model in models:
        for run_id in range(runs):
            fs = FileSystem(context={'benchmark_id': benchmark_id,
                                     'fleet_id': benchmark_conf.get('fleet_id', 'default_fleet'),
                            'experiment_name': experiment_name,
                                     'model_name': model,
                                     'run_id': run_id})

            output_directory = Path(fs.model_output_dir(model_name=model, benchmark_id=benchmark_id,
                                                        experiment_name=experiment_name))
            if not output_directory.exists():
                exit(1)

            eval_report_path = fs.experiment_results_file(
                benchmark_id=benchmark_id, experiment_name=experiment_name, model_name=model)

            if not eval_report_path.exists():
                print(
                    f"Evaluation report for model '{model}' does not exist: {eval_report_path}")
                exit(1)

            num_elaborated_entries = count_jsonl_entries(eval_report_path)
            num_noisy_records = count_csv_records(
                fs.noisy_table_file("maintenance_log"))

            if num_elaborated_entries < num_noisy_records:
                print(
                    f"Model '{model}' has fewer elaborated entries ({num_elaborated_entries}) than noisy records ({num_noisy_records}).")
                exit(1)

    exit(0)
