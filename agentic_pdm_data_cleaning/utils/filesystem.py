"""
filesystem.py

This module provides the `FileSystem` class to manage file paths related to 
synthetic datasets, maintenance logs, and signal data for a vehicle fleet.

It relies on a central configuration file (`app_config.yaml`) and 
automatically constructs relevant directories and filenames.
"""

import os
from pathlib import Path

from agentic_pdm_data_cleaning.utils.config_parser import ConfigSingleton
from .utility import load_config, sanitize_model_name


class FileSystem:
    """
    Provides an interface for managing paths and files used in synthetic data 
    generation, maintenance log storage, benchmarks, and experiments.

    The FileSystem class centralizes all path management for the application,
    including data storage, configuration files, benchmark results, and experiment outputs.

    Example usage:
        # Basic usage for data generation
        fs = FileSystem(context={'fleet_id': 'fleet_1'})
        devices_file = fs.clean_devices_file()

        # Benchmark and experiment usage
        fs = FileSystem(context={
            'benchmark_id': 'my_benchmark',
            'experiment_name': 'experiment_1',
            'fleet_id': 'fleet_1'
        })
        results_file = fs.experiment_results_file('qwen3:8b')
        log_file = fs.benchmark_log_file()

    Attributes:
        config (dict): Configuration loaded from app_config.yaml.
        data_dir (str): Base directory for data storage.
        app_config (str): Path to the configuration file.
        context (dict): Execution context (e.g., contains 'fleet_id', 'benchmark_id').
        fleet_id (str): Fleet identifier extracted from context.
        experiment_name (str): Experiment name extracted from context.
        experiment_type (str): Experiment type extracted from context.
        banchmark_name (str): Benchmark name extracted from context.
    """

    def __init__(self, context={}, app_config: str = 'config/app_config.yaml'):
        """
        Initializes the FileSystem.

        Args:
            context (dict): Context containing fleet_id or other runtime settings.
            app_config (str): Path to the application configuration YAML file.
        """
        self.config = ConfigSingleton.get_instance(path=app_config)
        self.data_dir = self.config.get('data', {}).get('dir_path', 'data')
        self.log_dir = self.config.get('log', {}).get('dir_path', 'logs')
        self.resources_dir = self.config.get('resources_dir', 'resources')
        self.app_config = app_config
        self.context = context
        self.fleet_id = context.get('global_params', {}).get(
            'fleet_id', context.get('fleet_id', 'default_fleet'))
        self.run_id = context.get('global_params', {}).get(
            'run_id', context.get('run_id', 0))
        self.experiment_name = context.get(
            'experiment_name', 'default_experiment')
        self.experiment_type = context.get(
            'experiment_type', 'default_experiment_type')
        self.benchmark_id = context.get(
            'benchmark_id', 'default_benchmark')
        self.model_name = context.get('model_name', 'default_model')

    @property
    def synthetic_dir(self):
        """Returns the path to the synthetic data directory."""
        return Path(self.data_dir) / 'synthetic'

    @property
    def common_synthetic_fleets_dir(self, benchmark_id=None):
        benchmark_id = benchmark_id if benchmark_id is not None else self.benchmark_id
        return self.synthetic_dir / benchmark_id / 'fleets'

    @property
    def synthetic_fleet_dir(self, run_id=None):
        """Returns the path to the current fleet's directory."""
        run_id = run_id if run_id is not None else self.run_id
        return self.common_synthetic_fleets_dir / self.fleet_id / f"run_{run_id}"

    @property
    def preprocessing_dir(self):
        """Returns the path to the preprocessing data directory."""
        return Path(self.data_dir) / 'preprocessing'

    @property
    def preprocessing_fleet_dir(self):
        """Returns the path to the preprocessing data directory."""
        return self.synthetic_fleet_dir

    @property
    def fleet_registry_dir(self):
        """Returns the path to the fleet_registry directory for the fleet."""
        return self.synthetic_fleet_dir / 'fleet_registry'

    @property
    def maintenance_dir(self):
        """Returns the path to the maintenance log directory."""
        return self.synthetic_fleet_dir / 'maintenance_log'

    @property
    def signals_dir(self):
        """Returns the path to the signals directory."""
        return self.synthetic_fleet_dir / 'signals'

    @property
    def results_dir(self):
        """Returns the path to the results data directory."""
        return Path(self.data_dir) / 'results'

    def get_signal_dir(self, signal_type: str):
        """
        Returns the path to the directory for a specific signal type.

        Args:
            signal_type (str): Type of signal (e.g., 'odometer').

        Returns:
        """
        return self.signals_dir / signal_type

    def get_banchmark_results_dir(self, benchmark_id: str):
        """
        Returns the path to the benchmark results directory.

        Args:
            benchmark_id (str): Name of the benchmark.

        Returns:
        """
        return self.results_dir / benchmark_id

    @property
    def rag_index_dir(self):
        """Returns the path to the RAG index storage directory."""
        return Path(self.data_dir) / 'rag_index_storage'

    @property
    def db_docs_dir(self):
        """Returns the path to the directory containing documents for RAG indexing."""
        return Path(self.resources_dir) / 'db_docs'

    def preprocessing_discarded_records_file(self):
        """Returns the path to the discarded records file in preprocessing."""
        os.makedirs(self.preprocessing_fleet_dir, exist_ok=True)
        return self.preprocessing_fleet_dir / 'discarded_records.json'

    def preprocessing_json_records(self):
        """Returns the path to the JSON records file in preprocessing."""
        os.makedirs(self.preprocessing_fleet_dir, exist_ok=True)
        return self.preprocessing_fleet_dir / 'json_records.json'

    def preprocessing_retained_records_file(self):
        """Returns the path to the retained records file in preprocessing."""
        os.makedirs(self.preprocessing_fleet_dir, exist_ok=True)
        return self.preprocessing_fleet_dir / 'retained_records.json'

    def clean_table_file(self, table_name: str):
        os.makedirs(self.synthetic_fleet_dir / table_name, exist_ok=True)
        return self.synthetic_fleet_dir / table_name / f"{table_name}_clean.csv"

    def noisy_table_file(self, table_name: str):
        os.makedirs(self.synthetic_fleet_dir / table_name, exist_ok=True)
        return self.synthetic_fleet_dir / table_name / f"{table_name}_noisy.csv"

    def get_manifest_path(self):
        """Ensures the synthetic fleet directory exists and returns the path to the manifest file for a given run."""
        os.makedirs(self.synthetic_fleet_dir, exist_ok=True)
        return self.synthetic_fleet_dir / f"synthetic_manifest.json"

    def clean_signal(self, vehicle_id: str, signal_type: str = 'odometer'):
        """
        Ensures the signal directory exists and returns the path to the signal file for a given vehicle.

        Args:
            vehicle_id (str): The vehicle identifier.
            signal_type (str): Type of signal (e.g., 'odometer').

        Returns:
            Path: Path to the CSV file for the specified vehicle and signal type.
        """
        os.makedirs(self.signals_dir / signal_type, exist_ok=True)
        return self.signals_dir / signal_type / f"{vehicle_id}_{signal_type}.csv"

    def get_fleet_db_file(self,):
        os.makedirs(self.synthetic_fleet_dir, exist_ok=True)
        return self.synthetic_fleet_dir / f"{self.fleet_id}.db"

    @property
    def config_dir(self):
        """Returns the path to the configuration directory."""
        return Path('config')

    @property
    def benchmarks_config_dir(self):
        """Returns the path to the benchmarks configuration directory."""
        return self.config_dir / 'benchmarks'

    @property
    def models_config_dir(self):
        """Returns the path to the models configuration directory."""
        return self.config_dir / 'models'

    @property
    def fleets_config_dir(self):
        """Returns the path to the fleet configuration directory."""
        return self.config_dir / 'fleets'

    def get_service_catalog_file_path(self):
        """
        Returns the path to the service catalog file used for RAG indexing.

        Returns:
        """
        return Path(self.synthetic_dir) / 'service_catalog.csv'

    def benchmark_config_file(self, benchmark_id: str):
        """
        Returns the path to a specific benchmark configuration file.

        Args:
            benchmark_id (str): Name of the benchmark configuration file.

        Returns:
            Path: Path to the benchmark configuration YAML file.
        """
        return self.benchmarks_config_dir / f"{benchmark_id}.yaml"

    def model_config_file(self, model_name: str):
        """
        Returns the path to a specific model configuration file.

        Args:
            model_name (str): Name of the model configuration file.

        Returns:
            Path: Path to the model configuration YAML file.
        """
        return self.models_config_dir / f"{model_name}.yaml"

    def fleet_config_file(self, fleet_name: str):
        """
        Returns the path to a specific fleet configuration file.

        Args:
            fleet_name (str): Name of the fleet configuration file.

        Returns:
            Path: Path to the fleet configuration YAML file.
        """
        return self.fleets_config_dir / f"{fleet_name}.yaml"

    def hyperparameters_space_config_file(self, space_name: str):
        """
        Returns the path to a specific hyperparameters space configuration file.

        Args:
            space_name (str): Name of the hyperparameters space configuration file.

        Returns:
            Path: Path to the hyperparameters space configuration YAML file.
        """
        return self.config_dir / 'hyperparameters_space' / f"{space_name}.yaml"

    def benchmark_output_dir(self, benchmark_id: str = None):
        """
        Returns the path to the output directory for a specific benchmark.

        Args:
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.

        Returns:
            Path: Path to the benchmark output directory.
        """
        benchmark_id = benchmark_id or self.benchmark_id
        benchmark_dir = self.results_dir / \
            benchmark_id
        os.makedirs(benchmark_dir, exist_ok=True)
        return benchmark_dir

    def experiment_output_dir(self, benchmark_id: str = None, experiment_name: str = None):
        """
        Returns the path to the output directory for a specific experiment within a benchmark.

        Args:
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.
            experiment_name (str, optional): Name of the experiment. Uses instance experiment_name if not provided.

        Returns:
            Path: Path to the experiment output directory.
        """
        experiment_name = experiment_name or self.experiment_name
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_dir = self.benchmark_output_dir(
            benchmark_id) / experiment_name
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    def model_output_dir(self, model_name, benchmark_id: str = None, experiment_name: str = None):
        """
        Returns the path to the output directory for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            Path: Path to the model output directory.
        """
        sanitized_model_name = sanitize_model_name(model_name)
        model_dir = self.experiment_output_dir(benchmark_id=benchmark_id, experiment_name=experiment_name) / sanitized_model_name / \
            f'run_{self.run_id}'
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def benchmark_log_file(self, benchmark_id: str = None):
        """
        Returns the path to the log file for a specific benchmark.

        Args:
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.

        Returns:
            Path: Path to the benchmark log file.
        """
        benchmark_id = benchmark_id or self.benchmark_id
        log_file = Path(self.log_dir) / f"{benchmark_id}.log"
        os.makedirs(log_file.parent, exist_ok=True)
        return log_file

    def experiment_results_file(self, model_name: str, benchmark_id: str = None, experiment_name: str = None):
        """
        Returns the path to the results file for a specific model in an experiment.

        Args:
            model_name (str): Name of the model.
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.
            experiment_name (str, optional): Name of the experiment. Uses instance experiment_name if not provided.

        Returns:
            Path: Path to the experiment results JSON file.
        """
        return self.model_output_dir(model_name=model_name, benchmark_id=benchmark_id, experiment_name=experiment_name) / "run_info.jsonl"

    @property
    def evaluations_dir(self):
        """Returns the path to the evaluation results directory."""
        return Path(self.data_dir) / 'evaluations'

    def evaluation_output_dir(self, benchmark_id: str = None):
        """
        Returns the path to the evaluation output directory for a specific benchmark.

        Args:
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.

        Returns:
            Path: Path to the evaluation output directory.
        """
        benchmark_id = benchmark_id or self.benchmark_id
        eval_dir = self.evaluations_dir / benchmark_id
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir

    def evaluation_results_file(self, experiment_name: str, model_name: str, benchmark_id: str = None):
        """
        Returns the path to the evaluation output file.

        Args:
            experiment_name (str): Name of the experiment.
            model_name (str): Name of the model.
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.

        Returns:
            Path: Path to the evaluation results JSON file.
        """

        # Sanitize model name for filename
        sanitized_model_name = sanitize_model_name(model_name)
        return self.evaluation_output_dir(benchmark_id) / f"{experiment_name}_{sanitized_model_name}_evaluation.json"

    # ========================================
    # INDEX AND STORAGE PATH METHODS
    # ========================================

    def setup_index_storage_dir(self):
        """Ensures the index storage directory exists and returns its path."""
        os.makedirs(self.rag_index_dir, exist_ok=True)
        return self.rag_index_dir

    # ========================================
    # UTILITY METHODS
    # ========================================

    def ensure_directory_exists(self, path: Path):
        """
        Ensures a directory exists, creating it if necessary.

        Args:
            path (Path): Path to the directory.

        Returns:
            Path: The same path that was passed in.
        """
        os.makedirs(path, exist_ok=True)
        return path

    def get_experiment_context(self, benchmark_id: str, experiment_name: str, fleet_id: str = None):
        """
        Creates a context dictionary for experiment execution.

        Args:
            benchmark_id (str): Name of the benchmark.
            experiment_name (str): Name of the experiment.
            fleet_id (str, optional): Fleet identifier. Uses instance fleet_id if not provided.

        Returns:
            dict: Context dictionary containing all relevant identifiers.
        """
        return {
            'benchmark_id': benchmark_id,
            'experiment_name': experiment_name,
            'fleet_id': fleet_id or self.fleet_id,
            'experiment_type': self.experiment_type
        }

    # ========================================
    # EVALUATOR SPECIFIC PATH METHODS
    # ========================================

    def benchmark_config_path(self, benchmark_id: str):
        """
        Returns the path to a benchmark configuration file.

        Args:
            benchmark_id (str): Name of the benchmark.

        Returns:
            Path: Path to the benchmark configuration YAML file.
        """
        return self.benchmarks_config_dir / f"{benchmark_id}.yaml"

    def model_config_path(self, model_name: str):
        """
        Returns the path to a model configuration file.

        Args:
            model_name (str): Name of the model (can include colons).

        Returns:
            Path: Path to the model configuration YAML file.
        """
        return self.models_config_dir / f"{model_name}.yaml"

    def evaluation_dataset_file(self, experiment_name: str, model_name: str, benchmark_id: str = None):
        """
        Returns the path to the evaluation dataset file.
        Args:
            experiment_name (str): Name of the experiment.
            model_name (str): Name of the model.
            benchmark_id (str, optional): Name of the benchmark. Uses instance benchmark_id if not provided.            
        Returns:    
        """
        benchmark_id = benchmark_id or self.benchmark_id
        sanitized_model_name = sanitize_model_name(model_name)
        return self.model_output_dir(model_name) / f"{experiment_name}_{sanitized_model_name}_evaluation_dataset.csv"

    def cleaned_maintenance_log(self, model_name, benchmark_id=None, experiment_name=None, run_id=None):
        """Returns the path to the cleaned maintenance log file."""
        model_dir = self.model_output_dir(
            model_name=model_name, benchmark_id=benchmark_id, experiment_name=experiment_name)
        return model_dir / f"maintenance_log_cleaned.csv"

    def llm_params(self):
        """Returns the path to the LLM params file."""
        return self.synthetic_fleet_dir / f"llm_params.json"

    def get_eval_report_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the evaluation report file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name / model_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'evaluation_report.jsonl'

    def get_family_report_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the family report file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name / model_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / f'family_report_{model_name}.jsonl'

    def get_noise_report_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the noise report file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name / model_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / f'noise_report_{model_name}.jsonl'

    def get_common_eval_report_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the evaluation report file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'evaluation_report.jsonl'

    def get_common_family_report_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the family report file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'family_report.csv'

    def get_common_noise_report_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the noise report file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'noise_report.csv'

    def get_common_per_noise_table_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the per-noise-type metrics table file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'per_noise_type_metrics.csv'

    def get_aggregate_eval_metrics_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the aggregate evaluation metrics file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'evaluation_metrics_aggregate.csv'

    def get_aggregate_usage_metrics_path(self, benchmark_id: str = None, experiment_name: str = None, model_name: str = None):
        """Returns the path to the aggregate usage metrics file."""
        benchmark_id = benchmark_id or self.benchmark_id
        experiment_name = experiment_name or self.experiment_name
        model_name = model_name or self.model_name
        eval_dir = self.evaluations_dir / benchmark_id / experiment_name
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir / 'usage_metrics_aggregate.csv'
