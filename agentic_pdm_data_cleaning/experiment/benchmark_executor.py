from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from agentic_pdm_data_cleaning.benchmark import EXPERIMENT_RUNNERS
from agentic_pdm_data_cleaning.utils import FileSystem, LoggerFactory
from agentic_pdm_data_cleaning.utils.check_benchmark import (
    count_csv_records, count_jsonl_entries)
from agentic_pdm_data_cleaning.utils.config_parser import ConfigSingleton


class BenchmarkExecutor:
    """
    Execute a benchmark defined by a YAML config file.

    Usage (programmatic):
        executor = BenchmarkExecutor("config/benchmarks/benchmark_functional_output.yaml")
        results = executor.run(only=["experiment_a"], skip=["experiment_b"])

    Usage (CLI):
        python -m agentic_pdm_data_cleaning.benchmark_executor \
            config/benchmarks/benchmark_functional_output.yaml --only experiment_a
    """

    def __init__(
        self,
        benchmark_config={},
        logger_factory: Optional[LoggerFactory] = None,
        seed: int = 42,
        run_id: int = -1,
    ) -> None:
        # Ensure any global config is initialized (mirrors your script)
        ConfigSingleton.get_instance()

        self.benchmark_config = benchmark_config
        if run_id >= 0:
            # Override run if provided
            self.benchmark_config['run_id'] = run_id
        print(f"Benchmark config: {self.benchmark_config}")

        self.benchmark_id: str = self.benchmark_config.get(
            "benchmark_id", "default_benchmark_id")
        self.benchmark_models: List[Dict[str, Any]
                                    ] = self.benchmark_config.get("models", [])

        self.logger_factory = logger_factory or LoggerFactory()
        self.fs = FileSystem(
            context=self.benchmark_config)

        # Paths from your FileSystem helpers
        self.output_results_path = Path(
            self.fs.benchmark_output_dir(self.benchmark_id))
        self.log_file_path = Path(
            self.fs.benchmark_log_file(self.benchmark_id))

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r") as f:
            return yaml.safe_load(f) or {}

    def iter_experiments(self) -> Iterable[tuple[str, Dict[str, Any]]]:
        return (self.benchmark_config.get("experiments") or {}).items()

    def run(
        self,
        only: Optional[Iterable[str]] = None,
        skip: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all (or a filtered subset of) experiments.

        Args:
            only: experiment names to include (if provided, others are ignored)
            skip: experiment names to exclude

        Returns:
            Dict mapping experiment_name -> runner.run() return value (if any)
        """
        only_set = set(only) if only else None
        skip_set = set(skip) if skip else set()

        print(f"Running benchmark: {self.benchmark_id}")

        results: Dict[str, Any] = {}
        for experiment_name, experiment_config in self.iter_experiments():
            if only_set is not None and experiment_name not in only_set:
                continue
            if experiment_name in skip_set:
                continue

            model_name = self.benchmark_config.get('models', [])[0]
            eval_report_path = self.fs.experiment_results_file(
                benchmark_id=self.benchmark_id, experiment_name=experiment_name, model_name=model_name)

            if eval_report_path.exists():
                num_elaborated_entries = count_jsonl_entries(eval_report_path)
                num_noisy_records = count_csv_records(
                    self.fs.noisy_table_file("maintenance_log"))
                if num_elaborated_entries >= num_noisy_records:
                    print(
                        f"Skipping experiment '{experiment_name}' for run {self.benchmark_config['run_id']} as it has already been run with sufficient entries.")
                    continue

            experiment_type = experiment_config.get("experiment_type")
            runner_cls = EXPERIMENT_RUNNERS.get(experiment_type)
            if runner_cls is None:
                raise ValueError(
                    f"Unknown experiment_type {experiment_type} for experiment {experiment_name}"
                )

            logger = self.logger_factory.get_logger_with_file(
                benchmark_id=self.benchmark_id,
                experiment_name=experiment_name,
                log_file_path=str(self.log_file_path),
            )

            print(
                f"\tRunning experiment {experiment_name} of type {experiment_type}")
            logger.info(
                f"Running experiment {experiment_name} of type {experiment_type}")

            runner = runner_cls(
                benchmark_config=self.benchmark_config,
                benchmark_id=self.benchmark_id,
                experiment_name=experiment_name,
                experiment_config=experiment_config,
                logger=logger,
                benchmark_models=self.benchmark_models,
            )

            result = runner.run()
            results[experiment_name] = result

            logger.info(
                f"Finished experiment {experiment_name} of type {experiment_type}")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run agentic cleaning benchmark")
    parser.add_argument(
        "path_benchmark_config",
        type=str,
        nargs="?",
        default="config/benchmarks/benchmark_functional_output.yaml",
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these experiment names",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Skip these experiment names",
    )
    args = parser.parse_args()

    executor = BenchmarkExecutor(args.path_benchmark_config)
    executor.run(only=args.only, skip=args.skip)


if __name__ == "__main__":
    main()
