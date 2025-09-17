from __future__ import annotations

import json
import os
import pathlib
import time
from importlib.resources import contents
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)

from agentic_pdm_data_cleaning.utils import constants
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem
from agentic_pdm_data_cleaning.utils.llm import CostManager


class Evaluator:
    def __init__(self, benchmark_config):
        self.benchmark_config = benchmark_config
        self.fleet_name = benchmark_config.get("fleet_name")
        self.benchmark_id = benchmark_config.get("benchmark_id")
        self.experiment_name = benchmark_config.get("experiment_name")
        self.model_name = benchmark_config.get("model_name")
        self.run_id = benchmark_config.get("run_id")
        self.cost_manager = CostManager()

    def aggregate_run_usage(self, model_name: str, df: pd.DataFrame) -> dict:
        """
        Expects columns:
          ['llm_run_completed','error','num_requests','request_tokens',
           'response_tokens','total_tokens','time']
        Returns aggregate usage metrics for the run.
        """
        df = df.copy()

        # Make sure numerics are numeric
        for c in ['num_requests', 'request_tokens', 'response_tokens', 'total_tokens', 'time']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Flags
        completed = df['llm_run_completed'] if 'llm_run_completed' in df.columns else pd.Series([
                                                                                                True]*len(df))
        error_flag = df['error'].notna(
        ) if 'error' in df.columns else pd.Series([False]*len(df))

        # Safe helper for means with NaNs
        def _mean(s): return float(pd.to_numeric(
            s, errors='coerce').mean(skipna=True))

        # Aggregate
        summary = {
            "n_records":      int(len(df)),
            "request_tokens_total":    int(df['request_tokens'].sum(skipna=True)),
            "response_tokens_total":   int(df['response_tokens'].sum(skipna=True)),
            "total_tokens_total":      int(df['total_tokens'].sum(skipna=True)),
            "total_time_seconds":      float(df['time'].sum(skipna=True)),
            "total_requests":          int(df['num_requests'].sum(skipna=True)),
            "total_cost":             self.cost_manager.get_model_costs(
                model_name,
                int(df['request_tokens'].sum(skipna=True)),
                int(df['response_tokens'].sum(skipna=True))
            ) if 'request_tokens' in df.columns and 'response_tokens' in df.columns else None,
            "completed":               int(completed.sum()),
            "completion_rate":         float(completed.mean()),
            "error_count":             int(error_flag.sum()),
        }
        return summary

    def evaluate_run(self, benchmark_id, experiment_name, model_name, run_id):
        print("Evaluation for:", benchmark_id,
              experiment_name, model_name, run_id)
        labels = [
            constants.LABEL_CLASS_ACCEPT, constants.LABEL_CLASS_REJECT, constants.LABEL_CLASS_UPDATE, constants.LABEL_CLASS_FAILED]
        self.benchmark_config['run_id'] = run_id
        self.fs = FileSystem(context=self.benchmark_config)
        self.report_path = self.fs.get_eval_report_path(
            benchmark_id=benchmark_id, experiment_name=experiment_name, model_name=model_name)
        predicted_log_path = self.fs.cleaned_maintenance_log(
            model_name, benchmark_id=benchmark_id, experiment_name=experiment_name, run_id=run_id)
        if not os.path.exists(predicted_log_path):
            print(
                f"Predicted log file not found for model {model_name}, experiment {experiment_name}, run {run_id}. Skipping.")
            return
        predicted_log = pd.read_csv(
            predicted_log_path)

        predicted_log.rename(
            columns={"label": "predicted_label"}, inplace=True)

        ground_truth = pd.read_csv(self.fs.noisy_table_file('maintenance_log'))
        clean_log = pd.read_csv(self.fs.clean_table_file('maintenance_log'))
        run_info = pd.read_json(self.fs.experiment_results_file(
            model_name=model_name, benchmark_id=benchmark_id, experiment_name=experiment_name), lines=True)
        run_info.rename(columns={'id': 'work_order_number'}, inplace=True)
        merged_log = pd.merge(predicted_log, run_info,
                              on='work_order_number', how='inner')
        _, report = evaluate_classification(
            ground_truth, predicted_log, labels=labels)
        model_usage_metrics = get_usage_metrics(
            ground_truth, merged_log, labels=labels)

        # === NEW: attach LLM usage summary to the report ===
        usage_cols = [
            'llm_run_completed', 'error', 'num_requests',
            'request_tokens', 'response_tokens', 'total_tokens', 'time'
        ]
        have_cols = [c for c in usage_cols if c in merged_log.columns]
        if len(have_cols) >= 4:  # need at least requests/tokens/time
            report['llm_usage'] = self.aggregate_run_usage(
                model_name, merged_log)
        report['noise_metrics'] = get_noise_metrics(
            ground_truth, merged_log, clean_log, labels=labels)
        report['model_metrics'] = model_usage_metrics

        family_report, noise_report = compute_family_and_noise_reports(
            ground_truth, predicted_log, clean_log,
            model_name=model_name, run_id=run_id
        )

        fam_path, noise_path = persist_family_and_noise_reports(
            family_report, noise_report,
            out_dir=self.fs.get_eval_report_path(
                benchmark_id=benchmark_id, experiment_name=experiment_name, model_name=model_name).parent,
            model_name=model_name,
            run_id=run_id,
            append=True
        )

        print("Saved:", fam_path, noise_path)

        save_run_report(report, self.report_path, exp_id=experiment_name,
                        run_id=run_id, model=model_name)

    def merge_family_reports(self, benchmark_id, experiment_name, models):

        paths = [
            self.fs.get_family_report_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            )
            for model in models
            if os.path.exists(self.fs.get_family_report_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            ))
        ]

        # Path for the merged output
        merged_path = self.fs.get_common_family_report_path(
            benchmark_id=benchmark_id,
            experiment_name=experiment_name,
        )

        dfs = []
        for path in paths:
            if os.path.exists(path):
                df = pd.read_json(path, lines=True)
                dfs.append(df)
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(merged_path, index=False)
        else:
            print("No family tables found to merge.")

    def merge_noise_reports(self, benchmark_id, experiment_name, models):
        import pandas as pd

        # Collect paths to all individual model family tables

        paths = [
            self.fs.get_noise_report_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            )
            for model in models
            if os.path.exists(self.fs.get_noise_report_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            ))
        ]

        # Path for the merged output
        merged_path = self.fs.get_common_noise_report_path(
            benchmark_id=benchmark_id,
            experiment_name=experiment_name,
        )

        # Merge contents
        dfs = []
        for path in paths:
            if os.path.exists(path):
                df = pd.read_json(path, lines=True)
                dfs.append(df)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(merged_path, index=False)
        else:
            print("No noise reports found to merge.")

    def merge_per_noise_type_tables(self, benchmark_id, experiment_name, models):
        import pandas as pd

        # Collect paths to all individual model per-noise-type tables
        paths = [
            self.fs.get_per_noise_table_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            )
            for model in models
        ]

        # Path for the merged output
        merged_path = self.fs.get_common_per_noise_table_path(
            benchmark_id=benchmark_id,
            experiment_name=experiment_name
        )

        # Merge contents
        dfs = []
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(merged_path, index=False)
        else:
            print("No per-noise-type tables found to merge.")

    def merge_run_reports(self, benchmark_id, experiment_name, models):
        import json

        # Collect paths to all individual model reports
        paths = [
            self.fs.get_eval_report_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            )
            for model in models
            if os.path.exists(self.fs.get_eval_report_path(
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                model_name=model
            ))
        ]

        if len(paths) == 0:
            return

        common_eval_report_path = self.fs.get_common_eval_report_path(
            benchmark_id=benchmark_id,
            experiment_name=experiment_name
        )

        # Merge contents
        with open(common_eval_report_path, 'w') as outfile:
            for path in paths:
                with open(path, 'r') as infile:
                    for line in infile:
                        outfile.write(line)

    def compute_aggregate_llm_usage(
            self,
        input_jsonl=None,
        group_by=("model",),  # e.g., ("model", "experiment_name")
        save_csv=None,
        save_json=None,
    ) -> pd.DataFrame:
        """
        Aggregate LLM usage metrics across records in a JSONL file.

        It looks for a dict at key 'llm_usage' (preferred) or 'usage' (fallback).
        All numeric fields inside that dict are aggregated per group using mean and std.

        Parameters
        ----------
        input_jsonl : str | Path
            Path to the evaluation report JSONL.
        group_by : Iterable[str]
            Columns to group by. Supported auto-extracted fields:
            - 'model'            (string)
            - 'run_id'           (string/int, if present)
            - 'experiment_name'  (string, if present)
            - 'benchmark_id'     (string, if present)
            - 'fleet'            (string, if present)
        save_csv : str | Path | None
            Where to save a CSV of the aggregated table. Pass None to skip saving.
        save_json : str | Path | None
            Where to save a JSON (records) of the aggregated table. Pass None to skip saving.

        Returns
        -------
        pd.DataFrame
            Aggregated table with mean and std columns per numeric usage field.
        """

        input_jsonl = Path(input_jsonl)

        # -------- helpers --------
        def _find_llm_usage(obj: Any) -> Optional[Dict[str, Any]]:
            """Recursively locate a dict under key 'llm_usage'."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "llm_usage" and isinstance(v, dict):
                        return v
                    found = _find_llm_usage(v)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = _find_llm_usage(item)
                    if found is not None:
                        return found
            return None

        def _get_nested(rec: Dict[str, Any], *path: str) -> Any:
            cur = rec
            for key in path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    return None
            return cur

        def _find_model(rec: Dict[str, Any]) -> str:
            for k in ("model", "model_name", "full_model_identifier"):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [
                ("report", "model"),
                ("report", "model_name"),
                ("report", "full_model_identifier"),
                ("metadata", "model"),
                ("metadata", "model_name"),
                ("model", "full_model_identifier"),
                ("config", "model_name"),
                ("run_config", "model_name"),
                ("benchmark", "model_name"),
            ]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return "UNKNOWN_MODEL"

        def _find_run_id(rec: Dict[str, Any]) -> Optional[Union[str, int]]:
            for k in ("run_id", "id", "uuid"):
                v = rec.get(k)
                if isinstance(v, (str, int)):
                    return v
            for p in [
                ("report", "run_id"),
                ("metadata", "run_id"),
                ("run", "id"),
                ("config", "run_id"),
            ]:
                v = _get_nested(rec, *p)
                if isinstance(v, (str, int)):
                    return v
            return None

        def _find_experiment_name(rec: Dict[str, Any]) -> Optional[str]:
            for k in ("experiment_name",):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [("report", "experiment_name"), ("metadata", "experiment_name"), ("config", "experiment_name")]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return None

        def _find_benchmark_id(rec: Dict[str, Any]) -> Optional[str]:
            for k in ("benchmark_id",):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [("report", "benchmark_id"), ("metadata", "benchmark_id"), ("config", "benchmark_id")]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return None

        def _find_fleet(rec: Dict[str, Any]) -> Optional[str]:
            for k in ("fleet", "fleet_name"):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [("report", "fleet"), ("metadata", "fleet"), ("config", "fleet")]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return None

        # -------- read & normalize rows --------
        if not input_jsonl.exists():
            raise FileNotFoundError(f"File not found: {input_jsonl}")

        rows: List[Dict[str, Any]] = []

        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                usage = _find_llm_usage(rec)
                if usage is None:
                    # common fallback layouts
                    if isinstance(rec.get("usage"), dict):
                        usage = rec["usage"]
                    elif isinstance(rec.get("llm"), dict) and isinstance(rec["llm"].get("usage"), dict):
                        usage = rec["llm"]["usage"]

                if not isinstance(usage, dict):
                    continue

                # keep numeric fields only
                numeric_usage = {k: v for k, v in usage.items(
                ) if isinstance(v, (int, float, np.number))}
                if not numeric_usage:
                    continue

                # attach grouping fields if available
                numeric_usage["model"] = _find_model(rec)
                maybe = _find_run_id(rec)
                if maybe is not None:
                    numeric_usage["run_id"] = maybe
                maybe = _find_experiment_name(rec)
                if maybe is not None:
                    numeric_usage["experiment_name"] = maybe
                maybe = _find_benchmark_id(rec)
                if maybe is not None:
                    numeric_usage["benchmark_id"] = maybe
                maybe = _find_fleet(rec)
                if maybe is not None:
                    numeric_usage["fleet"] = maybe

                rows.append(numeric_usage)

        if not rows:
            raise ValueError(
                "No numeric LLM usage fields found. Ensure records contain 'llm_usage' or 'usage' with numeric metrics."
            )

        df = pd.DataFrame(rows)

        # ensure all requested group_by columns exist
        missing = [g for g in group_by if g not in df.columns]
        if missing:
            raise KeyError(
                f"Requested group_by columns not found in extracted data: {missing}. "
                f"Available columns: {sorted(df.columns)}"
            )

        metric_cols = [c for c in df.columns if c not in group_by]

        # group and aggregate (mean & std)
        agg = df.groupby(list(group_by), dropna=False).agg(
            {
                **{m: ["mean", "std"] for m in metric_cols},
                metric_cols[0]: "count"
            }
        )

        agg.rename(
            columns={"n_records_count": "total_num_repetitions"}, inplace=True)

        # flatten MultiIndex columns: ('total_tokens','mean') -> 'total_tokens_mean'
        agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
        agg = agg.reset_index()

        # save outputs if requested
        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            agg.to_csv(save_csv, index=False)
        if save_json:
            Path(save_json).parent.mkdir(parents=True, exist_ok=True)
            agg.to_json(save_json, orient="records", indent=2)

        return agg

    def compute_aggregate_eval_metrics(self,
                                       input_jsonl=None,
                                       # e.g., ("model", "experiment_name")
                                       group_by=("model",),
                                       save_csv=None,
                                       save_json=None,
                                       ) -> pd.DataFrame:
        """
        Aggregate evaluation metrics (balanced accuracy, F1, precision, recall) across records in a JSONL file.

        The function is tolerant to different layouts:
          • 'classification_report' dict with 'macro avg' -> {'precision','recall','f1-score'}
          • flat or nested keys such as: 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'
          • generic 'metrics' blocks under e.g. 'report', 'metadata', etc.

        Grouping fields auto-extracted when available:
          - model, run_id, experiment_name, benchmark_id, fleet

        Parameters
        ----------
        input_jsonl : str | Path
            Path to the evaluation report JSONL.
        group_by : Iterable[str]
            Columns to group by (must exist in extracted rows, default: ('model',)).
        save_csv : str | Path | None
            Where to save CSV. Pass None to skip saving.
        save_json : str | Path | None
            Where to save JSON (records). Pass None to skip saving.

        Returns
        -------
        pd.DataFrame
            Aggregated table with mean and std for each metric per group.
        """

        input_jsonl = Path(input_jsonl)

        # -------- helpers --------
        def _get_nested(d: Dict[str, Any], *path: str) -> Any:
            cur = d
            for key in path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    return None
            return cur

        def _find_model(rec: Dict[str, Any]) -> str:
            for k in ("model", "model_name", "full_model_identifier"):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [
                ("report", "model"),
                ("report", "model_name"),
                ("report", "full_model_identifier"),
                ("metadata", "model"),
                ("metadata", "model_name"),
                ("model", "full_model_identifier"),
                ("config", "model_name"),
                ("run_config", "model_name"),
                ("benchmark", "model_name"),
            ]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return "UNKNOWN_MODEL"

        def _find_run_id(rec: Dict[str, Any]) -> Optional[Union[str, int]]:
            for k in ("run_id", "id", "uuid"):
                v = rec.get(k)
                if isinstance(v, (str, int)):
                    return v
            for p in [
                ("report", "run_id"),
                ("metadata", "run_id"),
                ("run", "id"),
                ("config", "run_id"),
            ]:
                v = _get_nested(rec, *p)
                if isinstance(v, (str, int)):
                    return v
            return None

        def _find_experiment_name(rec: Dict[str, Any]) -> Optional[str]:
            for k in ("experiment_name",):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [("report", "experiment_name"), ("metadata", "experiment_name"), ("config", "experiment_name")]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return None

        def _find_benchmark_id(rec: Dict[str, Any]) -> Optional[str]:
            for k in ("benchmark_id",):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [("report", "benchmark_id"), ("metadata", "benchmark_id"), ("config", "benchmark_id")]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return None

        def _find_fleet(rec: Dict[str, Any]) -> Optional[str]:
            for k in ("fleet", "fleet_name"):
                v = rec.get(k)
                if isinstance(v, str):
                    return v
            for p in [("report", "fleet"), ("metadata", "fleet"), ("config", "fleet")]:
                v = _get_nested(rec, *p)
                if isinstance(v, str):
                    return v
            return None

        # Priority lists (prefer macro if available, then weighted, then micro, then generic)
        BALANCED_ACC_KEYS = [
            "balanced_accuracy", "balanced_acc", "balanced_accuracy_score", "balanced_accuracy_macro",
            "bal_acc", "ba"
        ]
        F1_KEYS_PRIORITY = [
            "f1_macro", "macro_f1", "f1_macro_avg", "f1-score_macro", "f1_macro_score",
            "f1_weighted", "weighted_f1",
            "f1_micro", "micro_f1",
            "f1", "f1_score",
        ]
        PREC_KEYS_PRIORITY = [
            "precision_macro", "macro_precision", "precision_macro_avg", "precision-score_macro",
            "precision_weighted", "weighted_precision",
            "precision_micro", "micro_precision",
            "precision", "precision_score",
        ]
        REC_KEYS_PRIORITY = [
            "recall_macro", "macro_recall", "recall_macro_avg", "recall-score_macro",
            "recall_weighted", "weighted_recall",
            "recall_micro", "micro_recall",
            "recall", "recall_score",
        ]

        def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
            return {str(k).lower(): v for k, v in d.items()}

        def _recursive_number_for_any_key(obj: Any, candidates_lower: List[str]) -> Optional[float]:
            """
            DFS over dict/list; return first numeric value whose key (case-insensitive)
            matches any in candidates_lower.
            """
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lk = str(k).lower()
                    if lk in candidates_lower and isinstance(v, (int, float, np.number)):
                        return float(v)
                for v in obj.values():
                    val = _recursive_number_for_any_key(v, candidates_lower)
                    if val is not None:
                        return val
            elif isinstance(obj, list):
                for it in obj:
                    val = _recursive_number_for_any_key(it, candidates_lower)
                    if val is not None:
                        return val
            return None

        def _extract_macro_from_classification_report(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
            """
            Handle sklearn-like structure:
                classification_report['macro avg']['precision'|
                    'recall'|'f1-score']
            (may appear under 'report' or top-level).
            """
            # try some common paths
            candidates = [
                _get_nested(rec, "classification_report"),
                _get_nested(rec, "report", "classification_report"),
                _get_nested(rec, "metrics", "classification_report"),
                _get_nested(rec, "report", "metrics", "classification_report"),
            ]
            for rep in candidates:
                if isinstance(rep, dict) and "macro avg" in rep and isinstance(rep["macro avg"], dict):
                    macro = rep["macro avg"]
                    # keys vary: 'f1-score' vs 'f1', same for precision/recall
                    f1 = None
                    for k in ("f1-score", "f1", "f1_score"):
                        if k in macro and isinstance(macro[k], (int, float, np.number)):
                            f1 = float(macro[k])
                            break
                    prec = None
                    for k in ("precision", "precision_score"):
                        if k in macro and isinstance(macro[k], (int, float, np.number)):
                            prec = float(macro[k])
                            break
                    rec = None
                    for k in ("recall", "recall_score"):
                        if k in macro and isinstance(macro[k], (int, float, np.number)):
                            rec = float(macro[k])
                            break
                    return {"f1": f1, "precision": prec, "recall": rec}
            return {"f1": None, "precision": None, "recall": None}

        # -------- read & normalize rows --------
        if not input_jsonl.exists():
            raise FileNotFoundError(f"File not found: {input_jsonl}")

        rows: List[Dict[str, Any]] = []

        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row: Dict[str, Any] = {}
                # attach grouping fields if available
                row["model"] = _find_model(rec)
                rid = _find_run_id(rec)
                if rid is not None:
                    row["run_id"] = rid
                en = _find_experiment_name(rec)
                if en is not None:
                    row["experiment_name"] = en
                bid = _find_benchmark_id(rec)
                if bid is not None:
                    row["benchmark_id"] = bid
                fl = _find_fleet(rec)
                if fl is not None:
                    row["fleet"] = fl

                # 1) Try sklearn-like classification_report (macro avg)
                macro_triplet = _extract_macro_from_classification_report(rec)

                # 2) Balanced accuracy (search by common keys)
                ba = _recursive_number_for_any_key(
                    rec, [k.lower() for k in BALANCED_ACC_KEYS])

                # 3) F1 / Precision / Recall with priority lists (macro > weighted > micro > generic)
                f1 = macro_triplet["f1"] if macro_triplet["f1"] is not None else \
                    _recursive_number_for_any_key(
                        rec, [k.lower() for k in F1_KEYS_PRIORITY])

                precision = macro_triplet["precision"] if macro_triplet["precision"] is not None else \
                    _recursive_number_for_any_key(
                        rec, [k.lower() for k in PREC_KEYS_PRIORITY])

                recall = macro_triplet["recall"] if macro_triplet["recall"] is not None else \
                    _recursive_number_for_any_key(
                        rec, [k.lower() for k in REC_KEYS_PRIORITY])

                # Skip record if none of the metrics found
                if all(v is None for v in (ba, f1, precision, recall)):
                    continue

                if ba is not None:
                    row["balanced_accuracy"] = ba
                if f1 is not None:
                    row["f1"] = f1
                if precision is not None:
                    row["precision"] = precision
                if recall is not None:
                    row["recall"] = recall

                rows.append(row)

        if not rows:
            raise ValueError(
                "No evaluation metrics found. Ensure your JSONL contains either a 'classification_report' "
                "with 'macro avg' or keys like 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'."
            )

        df = pd.DataFrame(rows)

        # ensure requested group_by columns exist
        missing = [g for g in group_by if g not in df.columns]
        if missing:
            raise KeyError(
                f"Requested group_by columns not found: {missing}. "
                f"Available columns: {sorted(df.columns)}"
            )

        metric_cols = [c for c in (
            "balanced_accuracy", "f1", "precision", "recall") if c in df.columns]
        if not metric_cols:
            raise ValueError(
                "No numeric metric columns extracted (balanced_accuracy/f1/precision/recall).")

        agg = df.groupby(list(group_by), dropna=False).agg(
            {m: ["mean", "std"] for m in metric_cols})
        agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]  # flatten
        agg = agg.reset_index()

        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            agg.to_csv(save_csv, index=False)
        if save_json:
            Path(save_json).parent.mkdir(parents=True, exist_ok=True)
            agg.to_json(save_json, orient="records", indent=2)

        return agg


def get_usage_metrics(ground_truth, predicted_log, labels) -> dict:
    w_total_tokens = predicted_log['total_tokens'].fillna(
        0).to_numpy(dtype=float)
    w_num_requests = predicted_log['num_requests'].fillna(
        1).to_numpy(dtype=float)
    w_request_tokens = predicted_log['request_tokens'].fillna(
        0).to_numpy(dtype=float)
    w_response_tokens = predicted_log['response_tokens'].fillna(
        0).to_numpy(dtype=float)
    w_time = predicted_log['time'].fillna(0).to_numpy(dtype=float)
    cm_classification = confusion_matrix(
        ground_truth['label'], predicted_log['predicted_label'], labels=labels).tolist()
    cm_total_tokens = confusion_matrix(
        ground_truth['label'], predicted_log['predicted_label'], labels=labels, sample_weight=w_total_tokens).tolist()
    cm_num_requests = confusion_matrix(
        ground_truth['label'], predicted_log['predicted_label'], labels=labels, sample_weight=w_num_requests).tolist()
    cm_request_tokens = confusion_matrix(
        ground_truth['label'], predicted_log['predicted_label'], labels=labels, sample_weight=w_request_tokens).tolist()
    cm_response_tokens = confusion_matrix(
        ground_truth['label'], predicted_log['predicted_label'], labels=labels, sample_weight=w_response_tokens).tolist()
    cm_time = confusion_matrix(
        ground_truth['label'], predicted_log['predicted_label'], labels=labels, sample_weight=w_time).tolist()
    results = {
        "confusion_matrix": {
            "classification": cm_classification,
            "tokens": cm_total_tokens,
            "num_requests": cm_num_requests,
            "request_tokens": cm_request_tokens,
            "response_tokens": cm_response_tokens,
            "time": cm_time
        }
    }
    return results


def get_noise_metrics(ground_truth, predicted_log, clean_log, labels=None) -> dict:
    ground_truth['predicted_label'] = predicted_log['predicted_label']
    ground_truth['correct_prediction'] = ground_truth['label'] == ground_truth['predicted_label']
    predicted_log['correct_prediction'] = ground_truth['label'] == ground_truth['predicted_label']
    columns = ['id', 'work_order_number', 'start_date', 'end_date', 'license_plate',
               'system', 'subsystem', 'component', 'activity', 'work_description', 'work_order_type']
    correct_records = pd.merge(predicted_log[columns], clean_log[columns],
                               on='work_order_number', how='inner')
    ground_truth['correct_repair'] = ground_truth['work_order_number'].isin(
        correct_records['work_order_number'].tolist())
    s = ground_truth.groupby('noise_label')['predicted_label'].value_counts()
    df = s.rename('count').reset_index()
    nested = (df.groupby('noise_label')
                .apply(lambda d: dict(zip(d['predicted_label'], d['count'])))
                .to_dict())
    # 1) get the string
    json_str = json.dumps(nested, ensure_ascii=False, indent=2)
    return json_str


def compute_per_noise_table(
    ground_truth: pd.DataFrame,
    predicted_log: pd.DataFrame,
    clean_log: pd.DataFrame,
    model_name: str = "model",
    run_id: str | int = 0,
    labels: List[str] | None = None,
    *,
    id_col: str = "work_order_number",
    noise_col: str = "noise_label",
    y_true_col: str = "label",
    y_pred_col: str = "predicted_label",
    accept_label: str = "accept",
    target_action_map: dict | None = None,
) -> pd.DataFrame:
    """
    Returns a per-noise summary with:
      Noise | n | Acc | BalAcc | Macro-F1 | Target (Action) | Target-Recall | Hygiene | HygieneType

    HygieneType:
      - 'Over-Edit Rate' for clean/no-noise slices (e.g., 'none', 'absence_of_noise', 'clean')
      - 'Missed-Error Rate' for all other noises
    """
    ground_truth['predicted_label'] = predicted_log['predicted_label']
    ground_truth['correct_prediction'] = ground_truth['label'] == ground_truth['predicted_label']
    columns = ['id', 'work_order_number', 'start_date', 'end_date', 'license_plate',
               'system', 'subsystem', 'component', 'activity', 'work_description', 'work_order_type']
    correct_records = pd.merge(predicted_log[columns], clean_log[columns])
    ground_truth['correct_repair'] = ground_truth['work_order_number'].isin(
        correct_records['work_order_number'].tolist())

    # Merge by id to avoid relying on row order
    cols_gt = [id_col, noise_col, y_true_col,
               'correct_prediction', 'correct_repair']
    cols_pred = [id_col, y_pred_col]
    df = pd.merge(
        ground_truth[cols_gt].copy(),
        predicted_log[cols_pred].copy(),
        on=id_col,
        how="inner",
    )

    # Default mapping for typical noise → intended action
    default_map = {
        # clean slice
        "none": "accept",
        "absence_of_noise": "accept",
        "no_noise": "accept",
        "clean": "accept",
        # reject-type
        "vehicle_not_in_fleet": "reject",
        "digital_system_test": "reject",
        # update-type
        "wrong_vehicle_id": "update",
        "missing_value": "update",
        "wrong_end_date": "update",
        "wrong_start_date": "update",
        "categorical_typo": "update",
        "field_typo": "update",
    }
    if target_action_map:
        default_map.update(target_action_map)

    # Sets to decide hygiene type
    clean_like = {"none", "absence_of_noise", "no_noise", "clean"}

    rows = []
    for noise, g in df.groupby(noise_col):
        y_true = g[y_true_col].to_numpy()
        y_pred = g[y_pred_col].to_numpy()
        count_correct_predictions = g['correct_prediction'].sum()
        count_correct_repairs = g['correct_repair'].sum()
        n = len(g)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        # mean of recalls over classes present in y_true
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro",
                            zero_division=0, labels=labels)

        # Target action (fallback to majority true class if unknown noise)
        target = default_map.get(noise, None)
        if target is None:
            # choose majority of y_true as the intended action for that noise
            target = pd.Series(y_true).mode().iloc[0]

        # Target recall
        support_target = (y_true == target).sum()
        if support_target > 0:
            tp_target = ((y_true == target) & (y_pred == target)).sum()
            target_recall = tp_target / support_target
        else:
            target_recall = np.nan  # undefined if no true examples of target in this noise

        # Hygiene metric
        if noise in clean_like:
            # Over-Edit Rate: predicted not-accept when ground truth is accept
            over_edit = ((y_true == accept_label) & (
                y_pred != accept_label)).sum() / n
            hygiene_val = over_edit
            hygiene_name = "Over-Edit Rate"
        else:
            # Missed-Error Rate: predicted accept when there is actually an error
            missed_err = ((y_true != accept_label) & (
                y_pred == accept_label)).sum() / n
        rows.append({
            "round_id": run_id,
            "model": model_name,
            "noise_type": noise,
            "n": int(n),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "target_action": target,
            "target_recall": float(target_recall) if not np.isnan(target_recall) else np.nan,
            "correct_predictions": int(count_correct_predictions),
            "correct_repairs": int(count_correct_repairs),
            "error_detected_rate": (count_correct_predictions / n),
            "error_correction_rate": (count_correct_repairs / n),
        })

    out = pd.DataFrame(rows).sort_values("noise_type").reset_index(drop=True)
    return out


def evaluate_classification(ground_truth, predicted_log, labels=None, zero_division=0):
    """
    ground_truth: 1D array-like of true labels
    predicted_log: 1D labels OR 2D scores/logits/probs [n_samples, n_classes]
    labels: optional explicit label order; if None, uses union of y_true and y_pred
    """
    y_true = ground_truth['label']
    y_in = predicted_log['predicted_label']

    # Convert logits/probs to hard predictions if needed
    y_pred = y_in.argmax(axis=1) if y_in.ndim == 2 else y_in

    # Label order
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    # Confusion matrix & supports
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)  # samples per true class

    # Per-class accuracy == recall for that class
    per_class_acc = np.divide(
        np.diag(cm), support, out=np.zeros_like(support, dtype=float), where=support > 0
    )

    # Precision/Recall/F1 per class
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=zero_division,
    )

    # Macro & weighted aggregates
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=zero_division, labels=labels
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=zero_division, labels=labels
    )

    # Specificity (TNR) and G-mean per class (useful for imbalance)
    # One-vs-rest decomposition from cm
    total = cm.sum()
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = total - (tp + fp + fn)
    specificity = np.divide(
        tn, tn + fp, out=np.zeros_like(tp, dtype=float), where=(tn + fp) > 0
    )
    gmean = np.sqrt(rec * specificity)

    # Core metrics
    results = {
        "labels": list(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro": {"precision": float(prec_macro), "recall": float(rec_macro), "f1": float(f1_macro)},
        "weighted": {"precision": float(prec_weighted), "recall": float(rec_weighted), "f1": float(f1_weighted)},
        "per_class": {
            str(lbl): {
                "support": int(support[i]),
                "accuracy": float(per_class_acc[i]),  # == recall for class lbl
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "specificity": float(specificity[i]),
                "gmean": float(gmean[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),  # row=true, col=pred
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, digits=3, zero_division=zero_division
        ),
    }

    return y_pred, results


# metrics_agg.py

Z95 = 1.96  # normal approx for 95% CI


def save_noise_table(df: pd.DataFrame, filepath: str):
    # Check if the file already exists
    file_exists = os.path.isfile(filepath)
    # If it exists, append without writing the header again
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)


def save_run_report(
    report: Dict[str, Any],
    jsonl_path: str | pathlib.Path,
    *,
    exp_id: str,
    run_id: Optional[int] = None,
    model: str,
    split: Optional[str] = None,
    notes: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a single run to a JSONL log. 'report' is the dict returned by your evaluate_classification().
    """
    row = {
        "timestamp": time.time(),
        "exp_id": exp_id,
        "run_id": run_id,
        "model": model,
        "split": split,
        "notes": notes,
        "report": report,
    }
    if extra_meta:
        row.update(extra_meta)
    jsonl_path = pathlib.Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def create_run_report(fleet_name, benchmark_id, experiment_name, model_name, run_id):
    context = {'fleet': fleet_name, 'benchmark_id': benchmark_id,
               'experiment_name': experiment_name, 'model_name': model_name, 'run_id': run_id}
    fs = FileSystem(context)
    fs.make_dirs(fs.eval_dir)
    report_path = fs.get_eval_report_path()
    predicted_log = pd.read_csv(fs.cleaned_maintenance_log())
    ground_truth = pd.read_csv(fs.noisy_table_file('maintenance_log'))
    _, report = evaluate_classification(ground_truth, predicted_log, labels=[
        constants.LABEL_CLASS_ACCEPT, constants.LABEL_CLASS_REJECT, constants.LABEL_CLASS_UPDATE, constants.LABEL_CLASS_FAILED])
    save_run_report(report, report_path, exp_id=experiment_name,
                    run_id=run_id, model=model_name)


def _collect_runs(jsonl_path: str | pathlib.Path, exp_id: Optional[str]) -> List[Dict[str, Any]]:
    runs = []
    print(f"Loading runs from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if exp_id is None or obj.get("exp_id") == exp_id:
                runs.append(obj)
    if not runs:
        raise ValueError("No runs found for the given criteria.")
    return runs


def _flatten_scalar_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    """
    Pull out scalar metrics that are directly comparable across runs.
    """
    out = {}
    out["accuracy"] = float(report["accuracy"])
    out["balanced_accuracy"] = float(report["balanced_accuracy"])
    out["matthews_corrcoef"] = float(report["matthews_corrcoef"])
    out["cohen_kappa"] = float(report["cohen_kappa"])
    out["macro.precision"] = float(report["macro"]["precision"])
    out["macro.recall"] = float(report["macro"]["recall"])
    out["macro.f1"] = float(report["macro"]["f1"])
    out["weighted.precision"] = float(report["weighted"]["precision"])
    out["weighted.recall"] = float(report["weighted"]["recall"])
    out["weighted.f1"] = float(report["weighted"]["f1"])
    return out


def _per_class_df(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Return per-class metrics as a tidy DataFrame with index=label and columns for metrics.
    """
    rows = []
    for lbl, m in report["per_class"].items():
        rows.append({
            "label": lbl,
            "support": int(m["support"]),
            "accuracy": float(m["accuracy"]),      # == recall for that class
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "specificity": float(m["specificity"]),
            "gmean": float(m["gmean"]),
        })
    return pd.DataFrame(rows).set_index("label").sort_index()


def _sum_conf_mats(reports: List[Dict[str, Any]]) -> (np.ndarray, List[str]):
    """
    Build a unified label order and sum confusion matrices element-wise.
    """
    # Union of labels across runs (preserve as strings to match your report)
    all_labels = [constants.LABEL_CLASS_ACCEPT, constants.LABEL_CLASS_REJECT,
                  constants.LABEL_CLASS_UPDATE, constants.LABEL_CLASS_FAILED]
    k = len(all_labels)
    agg = np.zeros((k, k), dtype=int)

    index_of = {lbl: i for i, lbl in enumerate(all_labels)}
    for r in reports:
        lbls = [str(x) for x in r["labels"]]
        cm = np.array(r["confusion_matrix"], dtype=int)
        # Map to unified order
        idx = [index_of[l] for l in lbls]
        # Place cm into agg using outer indices
        for i_t, ii in enumerate(idx):
            for j_p, jj in enumerate(idx):
                agg[ii, jj] += cm[i_t, j_p]
    return agg, all_labels


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """
    Compute micro/macro summaries from an aggregated confusion matrix.
    """
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (tp + fp + fn)

    # Per-class recall/specificity
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)

    accuracy = tp.sum() / total if total > 0 else 0.0
    balanced_accuracy = np.mean(rec) if len(rec) else 0.0
    gmean = np.sqrt(rec * spec)
    macro = {
        "recall": float(np.mean(rec)) if rec.size else 0.0,
        "specificity": float(np.mean(spec)) if spec.size else 0.0,
        "gmean": float(np.mean(gmean)) if gmean.size else 0.0,
    }
    return {
        "micro_accuracy": float(accuracy),
        "macro_balanced_accuracy": float(balanced_accuracy),
        "macro_recall_from_cm": macro["recall"],
        "macro_specificity_from_cm": macro["specificity"],
        "macro_gmean_from_cm": macro["gmean"],
    }


def _save_jsonl(df, path: str, append: bool = True) -> None:
    """
    Save a DataFrame as JSON Lines.
    - append=True will append to an existing file (one record per line).
    - append=False will atomically overwrite the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = df.to_json(orient="records", lines=True,
                         force_ascii=False, date_format="iso")
    if not payload.strip():
        return  # nothing to write (empty df)

    if append:
        # append with a trailing newline to separate future appends
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)
            if not payload.endswith("\n"):
                f.write("\n")
    else:
        # atomic replace to avoid partial writes
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            if not payload.endswith("\n"):
                f.write("\n")
        os.replace(tmp, path)


def persist_family_and_noise_reports(
    family_report,
    noise_report,
    out_dir: str,
    model_name: str,
    *,
    run_id=None,
    append: bool = True,
) -> tuple[str, str]:
    """
    Persist two JSONL files:
      - {out_dir}/family_report.{model}.jsonl
      - {out_dir}/noise_report.{model}.jsonl

    If you prefer per-run files, include run_id in the filename.
    """
    if run_id is None:
        fam_path = os.path.join(out_dir, f"family_report.{model_name}.jsonl")
        noise_path = os.path.join(out_dir, f"noise_report.{model_name}.jsonl")
    else:
        fam_path = os.path.join(
            out_dir, f"family_report_{model_name}.jsonl")
        noise_path = os.path.join(
            out_dir, f"noise_report_{model_name}.jsonl")

    _save_jsonl(family_report, fam_path, append=append)
    _save_jsonl(noise_report, noise_path, append=append)
    return fam_path, noise_path


def compute_family_and_noise_reports(
    ground_truth: pd.DataFrame,
    predicted_log: pd.DataFrame,
    clean_log: pd.DataFrame,
    model_name: str = "model",
    run_id: str | int = 0,
    *,
    id_col: str = "work_order_number",
    noise_col: str = "noise_label",
    y_true_col: str = "label",
    y_pred_col: str = "predicted_label",
    target_action_map: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      family_report: one row per noise family with action-precision/recall
      noise_report : one row per noise type with % correctly classified and % correctly corrected (only for 'generative/update')

    Families & actions:
      - absence_of_noise → action 'accept'
      - corruptive      → action 'reject'
      - generative      → action 'update'
    """

    # --- Copy to avoid side-effects
    gt = ground_truth.copy()
    pl = predicted_log.copy()
    cl = clean_log.copy()

    # --- Merge GT + Predictions on id to align rows safely
    df = pd.merge(
        gt[[id_col, noise_col, y_true_col]],
        pl[[id_col, y_pred_col]],
        on=id_col,
        how="inner",
        validate="one_to_one",
    )
    df["correct_prediction"] = (df[y_true_col] == df[y_pred_col])

    # --- Build "correct update" detector: predicted record exactly equals clean record
    #     (we only count it if the prediction was 'update'; others are ignored)
    eq_cols = [
        'id', 'work_order_number', 'start_date', 'end_date', 'license_plate',
        'system', 'subsystem', 'component', 'activity', 'work_description', 'work_order_type'
    ]
    # keep only predicted updates for the equality join
    pl_updates = pl.loc[pl[y_pred_col] == "update", eq_cols]
    correct_update_rows = pd.merge(
        pl_updates, cl[eq_cols], on=eq_cols, how="inner")
    correct_update_ids = set(
        correct_update_rows[id_col]) if id_col in correct_update_rows.columns else set()
    df["correct_update"] = df[id_col].isin(correct_update_ids)

    # --- Map each noise to its intended action (target) and then to a family
    default_map = {
        # absence-of-noise slice
        "none": "accept", "absence_of_noise": "accept", "no_noise": "accept", "clean": "accept",
        # corruptive
        "vehicle_not_in_fleet": "reject", "digital_system_test": "reject",
        # generative
        "wrong_vehicle_id": "update", "missing_value": "update", "wrong_end_date": "update",
        "wrong_start_date": "update", "categorical_typo": "update", "field_typo": "update",
    }
    if target_action_map:
        default_map.update(target_action_map)

    def noise_target_action(noise: str, g: pd.DataFrame) -> str:
        act = default_map.get(noise)
        if act is not None:
            return act
        # fallback: majority true class inside that noise
        return g[y_true_col].mode().iloc[0]

    def action_to_family(action: str) -> str:
        return (
            "absence_of_noise" if action == "accept"
            else "corruptive" if action == "reject"
            else "generative" if action == "update"
            else "other"
        )

    # Precompute per-noise target action and family
    noise_meta = []
    for noise, g in df.groupby(noise_col):
        act = noise_target_action(noise, g)
        fam = action_to_family(act)
        noise_meta.append((noise, act, fam))
    noise_to_action = {n: a for n, a, _ in noise_meta}
    noise_to_family = {n: f for n, _, f in noise_meta}

    df["target_action"] = df[noise_col].map(noise_to_action)
    df["noise_family"] = df[noise_col].map(noise_to_family)

    # --- 1) FAMILY REPORT: action precision & recall per family
    family_rows = []
    for fam, fam_df in df.groupby("noise_family"):
        # Skip 'other' if it happens
        if fam not in {"absence_of_noise", "corruptive", "generative"}:
            continue

        action = {"absence_of_noise": "accept",
                  "corruptive": "reject", "generative": "update"}[fam]
        n = len(fam_df)

        # True/Pred flags for the family's action
        y_true_is_A = (fam_df[y_true_col] == action)
        y_pred_is_A = (fam_df[y_pred_col] == action)

        tp = int((y_true_is_A & y_pred_is_A).sum())
        fp = int((~y_true_is_A & y_pred_is_A).sum())
        fn = int((y_true_is_A & ~y_pred_is_A).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        family_rows.append({
            "round_id": run_id,
            "model": model_name,
            "noise_family": fam,
            "action": action,
            "n": n,
            "action_precision": float(prec) if not np.isnan(prec) else np.nan,
            "action_recall": float(rec) if not np.isnan(rec) else np.nan,
            "tp": tp, "fp": fp, "fn": fn,
        })
    family_report = pd.DataFrame(family_rows).sort_values(
        "noise_family").reset_index(drop=True)

    # --- 2) NOISE REPORT: per-noise correctness
    noise_rows = []
    for noise, g in df.groupby(noise_col):
        n = len(g)
        correctly_classified_rate = g["correct_prediction"].mean(
        ) if n > 0 else np.nan

        fam = noise_to_family[noise]
        # "correctly corrected" is meaningful for generative/update only
        if fam == "generative":
            correctly_corrected_rate = g["correct_update"].mean(
            ) if n > 0 else np.nan
        else:
            correctly_corrected_rate = np.nan

        noise_rows.append({
            "round_id": run_id,
            "model": model_name,
            "noise_type": noise,
            "noise_family": fam,
            "n": int(n),
            "correctly_classified_rate": float(correctly_classified_rate) if n > 0 else np.nan,
            "correctly_corrected_rate": float(correctly_corrected_rate) if n > 0 else np.nan,
        })
    noise_report = pd.DataFrame(noise_rows).sort_values(
        ["noise_family", "noise_type"]).reset_index(drop=True)

    return family_report, noise_report
