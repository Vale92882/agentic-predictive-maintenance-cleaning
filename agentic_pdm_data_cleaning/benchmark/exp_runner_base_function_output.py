# benchmark_runner/base.py
import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict

import logfire
import pandas as pd
import yaml
from dotenv import load_dotenv
from pydantic_ai import (Agent, RunContext, UnexpectedModelBehavior,
                         UsageLimitExceeded, capture_run_messages)
from pydantic_ai.usage import Usage, UsageLimits
from sqlalchemy import create_engine

from agentic_pdm_data_cleaning.domain import (EvaluationEntry, LLM_Answer,
                                              load_records_state)
from agentic_pdm_data_cleaning.domain.dependencies import Dependencies
from agentic_pdm_data_cleaning.domain.llm import LLM_Answer
from agentic_pdm_data_cleaning.utils import (FileSystem, LoggerFactory,
                                             constants, get_model, utility)
from agentic_pdm_data_cleaning.utils.db_utils import (describe_table,
                                                      list_tables,
                                                      run_sql_query)
from agentic_pdm_data_cleaning.utils.utility import load_config

load_dotenv()

"""Base class for running and evaluating experiments on the benchmark.
Methods used for evaluation: parse_runner_results and get_evaluation_dataset.
Methods used for running the experiment: all the other methods."""


class ExperimentRunner(ABC):
    def __init__(self, benchmark_config, benchmark_id, experiment_name, experiment_config, logger, benchmark_models, fleet_id='fleet_default'):
        self.benchmark_config = benchmark_config
        self.benchmark_id = benchmark_id
        self.experiment_name = experiment_name
        self.experiment_config = experiment_config
        self.logger = logger if logger else LoggerFactory.get_logger(
            self.__class__.__name__)
        self.benchmark_models = benchmark_models
        self.fleet_id = benchmark_config.get('fleet_id', 'fleet_default')
        self.num_records = experiment_config.get('num_records', -1)
        self.research_space = benchmark_config.get('research_space', 'space_1')
        self.run_id = benchmark_config.get('run_id', 0)

    @abstractmethod
    def get_system_prompt(self) -> Agent:
        pass

    @abstractmethod
    def get_user_prompt_template(self) -> str:
        "Template for the user prompt to be used by the agent."
        "The template must be paramenterized with the record."
        pass

    @abstractmethod
    def get_instructions(self) -> str:
        """Instructions for the agent to follow."""
        pass

    @abstractmethod
    def get_record_label_to_class_map(self):
        """Returns the mapping record_label to class for the experiment."""
        pass

    @abstractmethod
    def get_class_to_experiment_label_map(self):
        """Returns the mapping class to experiment_label for the experiment."""
        pass

    @abstractmethod
    def get_experiment_type(self):
        """Returns the type of the experiment."""
        pass

    def get_preprocessed_records(self, fs: FileSystem):
        """Returns the output type for the agent."""

        parsed_data = self.load_json_records(fs)
        return parsed_data

    def load_json_records(self, fs):
        file_json_records = fs.preprocessing_json_records()

        # Convert each JSON string into a dictionary
        parsed_data = utility.load_json_records(file_json_records)
        return parsed_data

    def filter_preprocessed_records(self, records):
        """Filters records based on the experiment configuration."""
        return records

    def build_runner_result(self, record_id, usage, total_time, output=None, error=None):
        """Builds the EvaluationEntry for the given record."""

        if error:
            llm_run_completed = False
            output_dic = {'human_error': False,
                          'repaired_json_record': None,
                          'details': "UnexpectedModelBehavior: " + str(error)}
        else:
            llm_run_completed = True
            output_dic = {
                'human_error': output.human_error,
                'repaired_json_record': output.repaired_json_record,
                'details': output.details
            }

        return EvaluationEntry(
            id=record_id,
            llm_run_completed=llm_run_completed,
            human_error=output_dic.get('human_error', False),
            repaired_json_record=output_dic.get('repaired_json_record', None),
            details=output_dic.get('details', 'No details provided.'),
            num_requests=usage.requests,
            request_tokens=usage.request_tokens,
            response_tokens=usage.response_tokens,
            total_tokens=usage.total_tokens,
            time=total_time
        )

    def parse_experiment_report(self, experiment_report_path):
        """
        Parse repaired records from a JSON lines result file.
        Returns: df_results, df_records_repaired
        """
        data = []
        repaired_records = []
        logger = self.logger
        with open(experiment_report_path, 'r') as f:
            for line in f:
                repaired_record = json.loads(line)
                data.append(repaired_record)
                repaired = repaired_record.get('repaired_json_record')
                if repaired not in [None, "None"]:
                    try:
                        repaired_json = json.loads(repaired) if isinstance(
                            repaired, str) else repaired
                        repaired_records.append(repaired_json)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not parse repaired_json_record for record_id {repaired_record.get('records_id')}")

        df_classification_results = pd.DataFrame(data)
        df_repaired_records = pd.DataFrame(
            repaired_records) if repaired_records else pd.DataFrame()
        logger.info(
            f"Parsed {len(df_classification_results)} result records with {len(df_repaired_records)} repaired records.")

        return df_classification_results, df_repaired_records

    def get_evaluation_dataset(self, ground_truth, df_classification_results):
        """
        Returns the evaluation dataset for the experiment.
        This is used to evaluate the classification performance of the agent on the benchmark.
        """
        ground_truth['class'] = ground_truth['noise_label'].map(
            self.get_record_label_to_class_map()).fillna(-1).astype(int)
        ground_truth['work_order_number'] = ground_truth['work_order_number'].astype(
            str)
        df_classification_results['predicted_class'] = df_classification_results['human_error'].apply(
            lambda x: 1 if x else 0)
        df_evaluation = pd.merge(
            df_classification_results, ground_truth, left_on="work_order_number", right_on='work_order_number', how='inner')
        return df_evaluation

    def get_agent(self, model_config, output_type, llm_params={}) -> Agent:
        """Returns the initialized agent."""

        logger = self.logger

        # Load LLM config
        llm_section = model_config.get("llm_config", {})

        # Define the model
        model = get_model(model_config.get("model_name", "llama3.1:latest"))
        if model is None:
            raise ValueError(
                f"Model configuration for {model_config.get('model_name')} not found.")

        # Load experiment-specific config
        experiment = model_config.get(
            self.get_experiment_type(), {})

        # Initialize the agent
        agent = Agent(
            model=model,
            name=experiment.get("agent_name", "Fleet Manager Assistant"),
            output_type=output_type,
            retries=experiment.get("retries", 4),
            output_retries=experiment.get("output_retries", 10),
            model_settings=llm_params
        )

        @agent.instructions
        def instructions(ctx: RunContext[Dependencies]) -> str:
            """Instructions for the agent to follow."""
            logger.debug("Instructions called")
            return self.get_instructions()

        @agent.system_prompt
        def system_prompt(ctx: RunContext[Dependencies]) -> str:
            """System prompt to set the context for the agent."""
            logger.debug("System prompt called")
            return self.get_system_prompt()

            # Tool: List tables
        @agent.tool
        def list_tables_tool(ctx: RunContext[Dependencies], db_name: str = "db") -> str:
            """Use this function to get a list of table names in the database."""
            logger.debug("Tool: list_tables called")
            return list_tables(ctx.deps.db_engine_ro)
        # Tool: Describe a table

        @agent.tool
        def describe_table_tool(ctx: RunContext[Dependencies], table_name: str) -> str:
            """Use this function to get a description of a table in the database.
            Args:
                ctx (RunContext[Dependencies]): The context containing the database engine.
                table_name (str): The name of the table to describe.
            Returns:
                str: A description of the table, including its columns and types."""
            logger.debug(f"Tool: describe_table called on table {table_name}")
            return describe_table(ctx.deps.db_engine_ro, table_name)

        # Tool: Run SQL
        @agent.tool
        def run_sql_tool(ctx: RunContext[Dependencies], query: str, limit: int) -> str:
            """ run_sql_tool is a tool that allows the agent to run SQL queries on the database.
            It takes a query string and an optional limit for the number of rows to return.
            The function uses the SQLAlchemy engine provided in the dependencies to execute the query.
            Args:
                ctx (RunContext[Dependencies]): The context containing the database engine.
                query (str): The SQL query to run.
                limit (int, optional): The maximum number of rows to return. Defaults to 10 so no more than 10 rows are returned.
            Returns:
                str: The result of the query as a JSON string.
            """
            logger.debug(f"Tool: run_sql called with query: {query}")
            return run_sql_query(ctx.deps.db_engine_ro, query, limit)

        return agent

    @abstractmethod
    def get_result_entry(self, record_id, usage, total_time, json_record=None, experiment_execution_output=None, execution_error=None):
        pass

    def parse_experiment_report(self, result_path):
        """
        Parse repaired records from a JSON lines result file.
        Returns: df_results, df_records_repaired
        """
        data = []
        repaired_records = []
        logger = self.logger
        with open(result_path, 'r') as f:
            for line in f:
                repaired_record = json.loads(line)
                data.append(repaired_record)
                repaired = repaired_record.get('repaired_json_record')
                if repaired not in [None, "None"]:
                    try:
                        repaired_json = json.loads(repaired) if isinstance(
                            repaired, str) else repaired
                        repaired_records.append(repaired_json)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not parse repaired_json_record for record_id {repaired_record.get('records_id')}")

        df_classification_results = pd.DataFrame(data)
        df_repaired_records = pd.DataFrame(
            repaired_records) if repaired_records else pd.DataFrame()
        logger.info(
            f"Parsed {len(df_classification_results)} result records with {len(df_repaired_records)} repaired records.")

        return df_classification_results, df_repaired_records

    def run(self):
        benchmark_config = self.benchmark_config
        if benchmark_config.get('enable_logfire', False):
            logfire.configure()
            logfire.instrument_pydantic_ai()
        benchmark_id = self.benchmark_id
        experiment_config = self.experiment_config
        logger = self.logger
        benchmark_models = self.benchmark_models
        experiment_name = self.experiment_name
        fleet_id = self.fleet_id
        run = benchmark_config.get('run_id', 0)
        print(
            f"Running {self.get_experiment_type()} with config: {experiment_config}")
        experiment_models = experiment_config.get('models', benchmark_models)
        print(f"Experiment models: {experiment_models}")

        # Initialize FileSystem with complete context
        fs_context = {
            'fleet_id': fleet_id,
            'benchmark_id': benchmark_id,
            'experiment_name': experiment_name,
            'run_id': run,
        }

        fs = FileSystem(context=fs_context)
        try:
            llm_params = load_config(fs.llm_params())
        except Exception as e:
            logger.error(f"Failed to load LLM params: {e}")
            llm_params = {'temperature': 0.1, 'top_p': 0.9}  # default params

        # Use FileSystem for all path management
        output_results_path = fs.benchmark_output_dir()

        logger.debug(f"Results will be saved to: {output_results_path}")
        print(f"Results will be saved to: {output_results_path}")
        logger.info(f"Experiment name: {experiment_name}")

        records = self.get_preprocessed_records(fs)
        records = self.filter_preprocessed_records(records)
        num_records = self.num_records
        if num_records < 0:
            num_records = len(records)
        else:
            num_records = min(num_records, len(records))

        db_engine_ro = create_engine(
            f"duckdb:///{fs.get_fleet_db_file()}", connect_args={"read_only": True})

        dependencies = Dependencies(
            db_engine_ro=db_engine_ro,
        )

        for model in experiment_models:
            model_config_path = fs.model_config_file(model)
            print(f"Model config path: {model_config_path}")
            if not model_config_path.exists():
                logger.info(
                    f"Model configuration file {model_config_path} does not exist. Skipping model {model}.")
                continue
            else:
                logger.debug(
                    f"Loading model configuration from {model_config_path}")
                with open(model_config_path, 'r') as f:
                    model_config = yaml.safe_load(f)

            logger.info(
                f"Running experiment {experiment_name} with model {model}")

            # Use FileSystem for results file path
            results_file_path = fs.experiment_results_file(
                model, benchmark_id, experiment_name)

            path_cleaned_maintenance_log = fs.cleaned_maintenance_log(
                model, benchmark_id, experiment_name)

            utility.scratch_cleaned_log(
                fs.noisy_table_file("maintenance_log"), path_cleaned_maintenance_log)

            def accept(work_order_number: str):
                print(
                    f"Accepting record with work_order_number: {work_order_number}")
                cleaned_maintenance_log = pd.read_csv(
                    path_cleaned_maintenance_log)
                cleaned_maintenance_log.loc[cleaned_maintenance_log['work_order_number']
                                            == work_order_number, 'label'] = constants.LABEL_CLASS_ACCEPT
                cleaned_maintenance_log.to_csv(
                    path_cleaned_maintenance_log, index=False)

            def reject(work_order_number: str):
                print(
                    f"Rejecting record with work_order_number: {work_order_number}")
                cleaned_maintenance_log = pd.read_csv(
                    path_cleaned_maintenance_log)
                cleaned_maintenance_log.loc[cleaned_maintenance_log['work_order_number']
                                            == work_order_number, 'label'] = constants.LABEL_CLASS_REJECT
                cleaned_maintenance_log.to_csv(
                    path_cleaned_maintenance_log, index=False)

            def update(work_order_number: str, field: str, value: str):
                cleaned_maintenance_log = pd.read_csv(
                    path_cleaned_maintenance_log)
                cleaned_maintenance_log.loc[cleaned_maintenance_log['work_order_number']
                                            == work_order_number, 'label'] = constants.LABEL_CLASS_UPDATE
                cleaned_maintenance_log.loc[cleaned_maintenance_log['work_order_number']
                                            == work_order_number, field] = value
                cleaned_maintenance_log.to_csv(
                    path_cleaned_maintenance_log, index=False)

            output_type = [accept, reject, update]

            if experiment_config.get('reset', False) and results_file_path.exists():
                os.remove(results_file_path)

            records_to_process = self.remove_records_already_processed(
                records, results_file_path)

            with open(results_file_path, 'a') as f:
                count = 0
                for record in records_to_process:
                    json_record = record.json_record if hasattr(
                        record, 'json_record') else record
                    logger.debug(f"Processing record: {json_record}")
                    usage: Usage = Usage()
                    start_time = time.perf_counter()
                    with capture_run_messages() as messages:
                        try:
                            max_retries = experiment_config.get(
                                'max_retries', 3)
                            timeout_seconds = experiment_config.get(
                                'timeout_seconds', 300)  # 5 minutes

                            def run_agent_with_args():
                                return self.run_agent(record, dependencies, model_config, usage, output_type, llm_params)

                            last_exception = None
                            for attempt in range(max_retries):
                                try:
                                    with ThreadPoolExecutor(max_workers=1) as executor:
                                        print(
                                            f"Running experiment {experiment_name} with model {model} - attempt {attempt} of {max_retries}")
                                        future = executor.submit(
                                            run_agent_with_args)
                                        output = future.result(
                                            timeout=timeout_seconds)
                                        last_exception = None
                                    break  # Success, exit retry loop
                                except TimeoutError:
                                    logger.error(
                                        f"Timeout occurred for record {json_record['work_order_number']} on attempt {attempt}")
                                    last_exception = TimeoutError(
                                        f"Timeout after {timeout_seconds} seconds")
                                except Exception as e:
                                    logger.error(
                                        f"Exception occurred for record {json_record['work_order_number']} on attempt {attempt}: {e}")
                                    print(e)
                                    last_exception = e
                            if last_exception is not None:
                                # All retries failed — mark as failed, don't raise
                                end_time = time.perf_counter()
                                total_time = end_time - start_time
                                logger.error(
                                    f"All retries failed for record {json_record['work_order_number']}: {last_exception}"
                                )
                                result_entry = self.get_result_entry(
                                    record_id=str(
                                        json_record["work_order_number"]),
                                    usage=usage,
                                    total_time=total_time,
                                    execution_error=last_exception,
                                )

                                work_order_number = json_record["work_order_number"]

                                cleaned_maintenance_log = pd.read_csv(
                                    path_cleaned_maintenance_log)
                                cleaned_maintenance_log.loc[cleaned_maintenance_log['work_order_number']
                                                            == work_order_number, 'label'] = constants.LABEL_CLASS_FAILED
                                cleaned_maintenance_log.to_csv(
                                    path_cleaned_maintenance_log, index=False)

                            else:
                                end_time = time.perf_counter()
                                total_time = end_time - start_time
                                print(f"Final Output: {output}")
                                result_entry = self.get_result_entry(
                                    record_id=str(
                                        json_record["work_order_number"]),
                                    experiment_execution_output=output,
                                    usage=usage,
                                    total_time=total_time,
                                    json_record=json_record,
                                )
                        except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
                            end_time = time.perf_counter()
                            total_time = end_time - start_time
                            logger.error(f'An error occurred: {e}')
                            logger.error(f'messages:{messages}')
                            result_entry = self.get_result_entry(
                                record_id=f"{str(json_record["work_order_number"])}",
                                usage=usage,
                                total_time=total_time,
                                execution_error=e
                            )
                    json.dump(asdict(result_entry), f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    count += 1
                    if count % num_records == 0:
                        break
                logger.info(
                    f"Processed {count} records for model {model}. The log cleaned by the agent is saved at {path_cleaned_maintenance_log} whereas the run report is saved to {results_file_path}")

    def remove_records_already_processed(self, records, results_file_path):
        """
        Removes records that have already been processed based on the results file.
        Returns a list of records that need to be processed.
        """
        if not results_file_path.exists():
            return records

        processed_ids = set()
        with open(results_file_path, 'r') as f:
            for line in f:
                result = json.loads(line)
                processed_ids.add(result['id'])

        return [record for record in records if str(record['work_order_number']) not in processed_ids]

    def run_agent(self, record, deps, model_config, usage, output_type, llm_params={}) -> LLM_Answer:

        logger = self.logger
        user_prompt = self.build_user_prompt(
            record, self.get_user_prompt_template())
        print(f"User prompt: {user_prompt}")
        agent = self.get_agent(model_config, output_type, llm_params)
        json_record = record.json_record if hasattr(
            record, 'json_record') else record
        logger.info(
            f"Running agent based on model {model_config} on record ID: {json_record["work_order_number"]}")
        usage_limits: UsageLimits = UsageLimits(
            request_limit=50)
        output = agent.run_sync(user_prompt=user_prompt,
                                deps=deps, usage=usage, usage_limits=usage_limits)
        result = output.output
        return result

    def build_user_prompt(self, record, user_prompt_template) -> str:
        return (
            user_prompt_template.format(
                record=record) if record else "No record provided."
        )
