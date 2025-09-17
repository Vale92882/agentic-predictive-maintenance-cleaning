
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agentic_pdm_data_cleaning.utils import FileSystem

# Import your existing pipeline and helpers
from .fleet_data_generator import (FleetDataGeneration, _derive_seed,
                                   _load_config)


@dataclass
class EnvironmentDataGeneration(FleetDataGeneration):
    """
    Extension of ExperimentDataGenerationPipeline that, for each run,
    samples RANDOM LLM parameters (temperature, top_p) from a config-defined
    search space and persists them to disk.

    Expected YAML config section (at top-level of your existing YAML):
    ------------------------------------------------------------------
    llm_search_space:
      temperature:
        # Either uniform range OR discrete choices
        # uniform: [0.0, 1.0]
        choices: [0.0, 0.2, 0.4, 0.7]
      top_p:
        uniform: [0.7, 1.0]
        # choices: [0.8, 0.9, 1.0]

    Notes:
    - Only RANDOM sampling is supported (one sample per pipeline run).
    - Sampling is deterministic per (base_seed, "llm_params", run).
    """

    llm_config_param_space_path: str = 'config/research_space/space_1.yaml'

    llm_param_space_config: Dict[str, Any] = field(
        init=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.llm_param_space_config = _load_config(
            self.llm_config_param_space_path)

    # ------------ LLM PARAMS: READ SPACE, SAMPLE, PERSIST ---------------------

    def _get_llm_search_space(self) -> Dict[str, Any]:
        space = self.llm_param_space_config.get("llm_search_space", {})
        if not isinstance(space, dict) or not space:
            raise ValueError(
                "Missing or invalid 'llm_search_space' in YAML. "
                "Provide temperature/top_p with either 'uniform' or 'choices'."
            )
        if "temperature" not in space or "top_p" not in space:
            raise ValueError(
                "llm_search_space must define both 'temperature' and 'top_p'.")
        return space

    def _sample_one(self, dim_cfg: Dict[str, Any], rng: random.Random, name: str) -> float:
        if "uniform" in dim_cfg:
            low, high = dim_cfg["uniform"]
            val = rng.uniform(float(low), float(high))
            return round(float(val), 6)
        if "choices" in dim_cfg:
            if not dim_cfg["choices"]:
                raise ValueError(f"llm_search_space.{name}.choices is empty")
            val = rng.choice(dim_cfg["choices"])
            return round(float(val), 6)
        raise ValueError(
            f"llm_search_space.{name} must have 'uniform' or 'choices'")

    def sample_llm_params(self) -> Dict[str, Any]:
        space = self._get_llm_search_space()
        seed_llm = _derive_seed(self.base_seed, "llm_params", self.run)
        rng = random.Random(seed_llm)

        temperature = self._sample_one(
            space["temperature"], rng, "temperature")
        top_p = self._sample_one(space["top_p"], rng, "top_p")

        params = {
            "temperature": temperature,
            "top_p": top_p,
        }
        return params

    def write_llm_params(self, params: Dict[str, Any], out_path: Optional[str] = None) -> str:
        if out_path is None:
            # Place next to the manifest by default
            fs = FileSystem(context=self.fleet_config)
            manifest_path = fs.get_manifest_path()
            base_dir = os.path.dirname(manifest_path)
            out_path = os.path.join(base_dir, f"llm_params.json")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        return out_path

    # ----------------------- OVERRIDE RUN PIPELINE ----------------------------

    def run_pipeline(
        self,
        make_db: bool = True,
        preprocess: bool = True,
        write_manifest: bool = True,
        write_llm_params: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the base data pipeline; additionally, sample and persist RANDOM LLM params.
        """
        summary = super().run_pipeline(
            make_db=make_db, preprocess=preprocess, write_manifest=write_manifest
        )

        params = self.sample_llm_params()
        self.write_llm_params(params)

        return summary
