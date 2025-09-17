import argparse
import os
import sys
import traceback

from agentic_pdm_data_cleaning.evaluation.evaluator import Evaluator
from agentic_pdm_data_cleaning.experiment.benchmark_executor import \
    BenchmarkExecutor
from agentic_pdm_data_cleaning.experiment.environment_generator import \
    EnvironmentDataGeneration
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem
from agentic_pdm_data_cleaning.utils.utility import load_config

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Set up experiment environment with synthetic data, database, preprocessing, and RAG index")

    # 'benchmark_config_path' is optional; if not provided, 'config/benchmarks/benchmark_1.yaml' will be used as the default
    argparser.add_argument(
        "--benchmark_config_path",
        type=str,
        default="config/benchmarks/benchmark_ubix.yaml",
        help="Path to benchmark configuration YAML file."
    )

    argparser.add_argument(
        "--skip_data_generation",
        action="store_true",
        help="Skip data generation step."
    )

    argparser.add_argument(
        "--skip_benchmark_execution",
        action="store_true",
        help="Skip benchmark execution step."
    )

    argparser.add_argument(
        "--execute_experiment_evaluation",
        action="store_true",
        default=False,
        help="Execute experiment evaluation step."
    )

    args = argparser.parse_args()

    # Load benchmark config
    try:
        benchmark_config = load_config(args.benchmark_config_path)
    except Exception as e:
        print(f"Error loading benchmark config: {e}")
        traceback.print_exc()
        sys.exit(1)

    fleet_id = benchmark_config.get(
        "global_params", {}).get("fleet_id", benchmark_config.get('fleet_id', 'default_fleet'))

    fs = FileSystem()
    try:
        fleet_config = load_config(fs.fleet_config_file(fleet_id))
    except Exception as e:
        print(f"Error loading fleet config: {e}")
        traceback.print_exc()
        sys.exit(1)

    base_seed = benchmark_config.get("seed", 42)

    base_runs = max(benchmark_config.get("runs", 3), 1)
    benchmark_id = benchmark_config.get('benchmark_id', 'default_benchmark_id')
    benchmark_models = benchmark_config.get('models', [])
    log_file_path = fs.benchmark_log_file(benchmark_id)

    if not args.skip_data_generation:
        for run in range(base_runs):
            print(
                f"Setting up environment for benchmark {benchmark_id} run {run}...")
            EnvironmentDataGeneration(
                benchmark_id=benchmark_id,
                fleet_config_path=fs.fleet_config_file(fleet_id),
                seed=base_seed + run,
                run=run,
                use_llm_generation=False,
                llm_config_param_space_path=fs.hyperparameters_space_config_file(
                    benchmark_config.get('hyperparameters_space', 'space_1')),
            ).run_pipeline(
                make_db=True,
                preprocess=True,
                write_manifest=True,
                write_llm_params=True,
            )

    if not args.skip_benchmark_execution:
        for run in range(base_runs):
            print(f"Running benchmark executor for run {run}...")
            benchmark_config['run_id'] = run
            BenchmarkExecutor(
                benchmark_config,
                seed=base_seed + run,
                run_id=run,
            ).run(
                only=None,
                skip=None,
            )

    if args.execute_experiment_evaluation:
        experiments: dict = benchmark_config.get('experiments', {})
        print(experiments)
        for experiment_name, experiment_config in experiments.items():
            report_path = fs.get_eval_report_path(
                benchmark_id=benchmark_id, experiment_name=experiment_name)
            os.remove(report_path) if os.path.exists(report_path) else None
            benchmark_config['experiment_name'] = experiment_name
            evaluator = Evaluator(benchmark_config)
            print(f"Evaluating experiment : {experiment_name}...")
            runs = experiment_config.get('runs', base_runs)
            print(f"Number of runs to evaluate: {runs}")
            for run in range(runs):
                for model in benchmark_models:
                    evaluator.evaluate_run(
                        benchmark_id, experiment_name, model, run)
            evaluator.merge_run_reports(
                benchmark_id, experiment_name, benchmark_models)
            evaluator.compute_aggregate_eval_metrics(
                input_jsonl=fs.get_common_eval_report_path(benchmark_id=benchmark_id, experiment_name=experiment_name), save_csv=fs.get_aggregate_eval_metrics_path(benchmark_id=benchmark_id, experiment_name=experiment_name))
            evaluator.compute_aggregate_llm_usage(
                input_jsonl=fs.get_common_eval_report_path(benchmark_id=benchmark_id, experiment_name=experiment_name), save_csv=fs.get_aggregate_usage_metrics_path(benchmark_id=benchmark_id, experiment_name=experiment_name))
