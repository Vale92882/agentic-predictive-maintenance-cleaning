import pandas as pd

from agentic_pdm_data_cleaning.domain.evaluation_entry import \
    ExperimentUsageReportEntry
from agentic_pdm_data_cleaning.utils import constants

from .exp_runner_base_function_output import ExperimentRunner


class ZeroShotFunctionalCleaning(ExperimentRunner):

    def get_experiment_type(self):
        """Returns the type of the experiment."""
        return constants.EXP_TYPE_SINGLE_SHOT_FUNCTIONAL

    def get_user_prompt_template(self) -> str:
        """
        Template for the user prompt to be used by the agent.
        The template must be parameterized with the record.
        """
        class_map_str = "".join(
            [f"{t[1]} - {t[2]}.\n" for t in self.get_class_to_experiment_label_map()]
        )
        return (
            "You are given one maintenance record:\n\n"
            "{record}\n\n"
            "Task. Decide exactly one action for this record and call the corresponding output function:\n"
            "   accept(work_order_number) — record is valid; no changes are needed.\n"
            "   reject(work_order_number) — record is invalid for this fleet.\n"
            "   update(work_order_number, field, value) — record is fixable with a single, concrete change (one field per call).\n"
            "Use the appropriate function to classify the record.\n"
        )

    def get_system_prompt(self) -> str:
        return (
            "You are a meticulous data curator focused on workshop maintenance logs.\n\n"
            "When correcting records, you access the resources in the DB. "
            "The DB contains: "
            " a fleet_registry table where all the vehicles of the target fleet are listed."
            " a service_catalog table that provides information about valid categories in maintenance records and their hierarchy."
            " a signal_odometer table that tracks the odometer readings of all vehicles in the fleet."
        )

    def get_instructions(self):
        return (
            "Spot inconsistencies, check DB tables and propose corrections when needed."
        )

    def get_record_label_to_class_map(self):
        return {
            constants.LABEL_MAIN_RECORD_CLEAN: 0,
            constants.LABEL_MAIN_RECORD_VEHICLE_NOT_IN_FLEET: 1,
            constants.LABEL_MAIN_RECORD_TEST: 1,
            constants.LABEL_MAIN_RECORD_WRONG_VEHICLE_ID: 2,
            constants.LABEL_MAIN_RECORD_WRONG_END_DATE: 2,
            constants.LABEL_FIELD_INTEGRITY_MISSING_VALUE: 2,
            constants.LABEL_FIELD_INTEGRITY_CATEGORICAL_TYPO: 2,
        }

    def get_class_to_experiment_label_map(self):
        """Returns the mapping record_label to class for the experiment."""
        return [
            (0, "accept", "record is valid; no changes are needed."),
            (1, "reject", "record is invalid for this fleet"),
            (2, "update", "record is fixable with a single, concrete change (one field per call)"),
        ]

    def get_evaluation_dataset(self, ground_truth, df_results):
        """
        Returns the evaluation dataset for the experiment.
        This is used to evaluate the performance of the agent on the benchmark.
        """
        ground_truth['class'] = ground_truth['noise_label'].map(
            self.get_record_label_to_class_map()).fillna(-1).astype(int)
        ground_truth['id'] = ground_truth['id'].astype(str)
        df_evaluation = pd.merge(
            df_results, ground_truth, left_on="id", right_on='id', how='inner')
        return df_evaluation

    def get_result_entry(self, record_id, usage, total_time, json_record=None, experiment_execution_output=None, execution_error=None):
        """Returns the result entry for the given record."""

        if execution_error:
            llm_run_completed = False
            error = "UnexpectedModelBehavior: " + str(execution_error)
        else:
            llm_run_completed = True
            error = None

        return ExperimentUsageReportEntry(
            id=record_id,
            llm_run_completed=llm_run_completed,
            error=error,
            num_requests=usage.requests,
            request_tokens=usage.request_tokens,
            response_tokens=usage.response_tokens,
            total_tokens=usage.total_tokens,
            time=total_time
        )
