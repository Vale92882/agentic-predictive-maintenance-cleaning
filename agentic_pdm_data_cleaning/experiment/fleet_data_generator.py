import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Reuse your project functions (these imports will resolve in your environment)
from agentic_pdm_data_cleaning.preprocessor.maintenance_record_preprocessor import \
    generate_preprocessed_records
from agentic_pdm_data_cleaning.synthetic_generator.fleet_registry import \
    generate_fleet_registry
from agentic_pdm_data_cleaning.synthetic_generator.maintenance_log import \
    generate_maintenance_log
from agentic_pdm_data_cleaning.synthetic_generator.signals import \
    generate_signals
from agentic_pdm_data_cleaning.utils import FileSystem, generate_db


def _get_pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "n/a"


def _derive_seed(base_seed: int, *scope) -> int:
    """Deterministically derive a sub-seed from base_seed and a scope tuple (32-bit)."""
    s = str(base_seed) + "::" + "::".join(map(str, scope))
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:8], 16)


def _seed_environment(base_seed: int):
    """Best-effort process-wide determinism (prefer local RNGs inside generators)."""
    os.environ["PYTHONHASHSEED"] = str(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)
    # If you use Faker here, you can also seed it:
    # from faker import Faker; Faker.seed(base_seed)


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _count_num_items(fleet_config: Dict[str, Any], subsection: str = "fleet_registry") -> int:
    generators = fleet_config.get(subsection, {}).get("generators", [])
    num = 0
    for g in generators:
        num += g.get("params", {}).get("count", 0)
    return num


@dataclass
class FleetDataGeneration:
    """
    Programmatic pipeline to generate a synthetic fleet dataset, load it into a DB,
    and optionally produce preprocessed records and a reproducibility manifest.

    Basic usage:
        pipe = FleetDataGenerationPipeline("config/fleet/fleet_3.yaml", seed=123, run=0, repro_mode=True)
        summary = pipe.run(make_db=True, preprocess=True, write_manifest=True)

    The returned `summary` dict includes seeds, sizes, versions, and platform info.
    """
    benchmark_id: Optional[str] = 'default_benchmark_id'
    fleet_config_path: str = 'config/fleet/default_fleet.yaml'
    seed: Optional[int] = None
    run: int = 0
    use_llm_generation: bool = False
    # If True, enforce that fleet has >= #maintenance_records (like the CLI script)
    strict_vehicle_count_check: bool = True

    # Internal state
    fleet_config: Dict[str, Any] = field(init=False, default_factory=dict)
    base_seed: int = field(init=False, default=42)

    def __post_init__(self):
        self.fleet_config = _load_config(self.fleet_config_path)
        # Resolve seed (explicit > config > default)
        self.base_seed = (
            int(self.seed)
            if self.seed is not None
            else int(self.fleet_config.get("global_params", {}).get("seed", 42))
        )
        self._prepare_config()

    def _prepare_config(self):
        # Global seeding
        _seed_environment(self.base_seed)

        # Inject global params back into config for downstream consumers
        gp = self.fleet_config.setdefault("global_params", {})
        gp["seed"] = int(self.base_seed)
        gp["use_llm_generation"] = bool(self.use_llm_generation)
        gp['run_id'] = int(self.run)

        # Ensure sub-sections exist
        self.fleet_config.setdefault("fleet_registry", {})
        self.fleet_config.setdefault("maintenance_log", {})
        self.fleet_config.setdefault("signals", {})

        self.fleet_config['benchmark_id'] = self.benchmark_id

        # Derive deterministic sub-seeds
        self.fleet_config["fleet_registry"]["seed"] = _derive_seed(
            self.base_seed, "fleet_registry")
        self.fleet_config["maintenance_log"]["seed"] = _derive_seed(
            self.base_seed, "maintenance_log")
        self.fleet_config["signals"]["seed"] = _derive_seed(
            self.base_seed, "signals")

        if self.strict_vehicle_count_check:
            n_fleet = _count_num_items(self.fleet_config, "fleet_registry")
            n_log = _count_num_items(self.fleet_config, "maintenance_log")
            if n_fleet < n_log:
                raise ValueError(
                    f"Invalid config: #vehicles ({n_fleet}) must be >= #maintenance_records ({n_log})."
                )

    # ---- Individual steps (call what you need) --------------------------------

    def generate_fleet_registry(self):
        return generate_fleet_registry(self.benchmark_id, self.fleet_config)

    def generate_maintenance_log(self):
        return generate_maintenance_log(self.benchmark_id, self.fleet_config)

    def generate_signals(self):
        return generate_signals(self.benchmark_id, self.fleet_config)

    def load_database(self):
        return generate_db(self.benchmark_id, fleet_config=self.fleet_config)

    def preprocess_records(self):
        return generate_preprocessed_records(self.benchmark_id, fleet_config=self.fleet_config)

    # ---- Manifest -------------------------------------------------------------

    def _make_manifest(self) -> Dict[str, Any]:
        n_fleet = _count_num_items(self.fleet_config, "fleet_registry")
        n_log = _count_num_items(self.fleet_config, "maintenance_log")
        manifest = {
            "seed": int(self.base_seed),
            "derived_seeds": {
                "fleet_registry": int(self.fleet_config["fleet_registry"]["seed"]),
                "maintenance_log": int(self.fleet_config["maintenance_log"]["seed"]),
                "signals": int(self.fleet_config["signals"]["seed"]),
            },
            "use_llm_generation": bool(self.use_llm_generation),
            "config_path": str(self.fleet_config_path),
            "sizes": {
                "num_fleet_vehicles": int(n_fleet),
                "num_maintenance_records": int(n_log),
            },
            "versions": {
                "python": sys.version,
                "numpy": _get_pkg_version("numpy"),
                "pandas": _get_pkg_version("pandas"),
                "faker": _get_pkg_version("faker"),
            },
            "platform": platform.platform(),
        }
        return manifest

    def write_manifest(self, manifest: Optional[Dict[str, Any]] = None, manifest_path: Optional[str] = None) -> str:
        if manifest is None:
            manifest = self._make_manifest()

        # If a custom path is not provided, prefer the project's FileSystem path
        if manifest_path is None:
            fs = FileSystem(context=self.fleet_config)
            manifest_path = fs.get_manifest_path()

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        print(json.dump(manifest, f, indent=2, sort_keys=True))
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return manifest_path

    # ---- Orchestrated run -----------------------------------------------------

    def run_pipeline(self, make_db: bool = True, preprocess: bool = True, write_manifest: bool = True) -> Dict[str, Any]:
        """
        Execute the full pipeline:
          1) Generate fleet registry, maintenance log, and signals
          2) (optional) Load data into DB
          3) (optional) Preprocess records
          4) (optional) Write reproducibility manifest
        Returns a summary dict (seeds, sizes, versions, platform).
        """
        # 1) Generators
        self.generate_fleet_registry()
        self.generate_maintenance_log()
        self.generate_signals()

        # 2) DB
        if make_db:
            self.load_database()

        # 3) Preprocessing
        if preprocess:
            self.preprocess_records()

        # 4) Manifest
        summary = self._make_manifest()
        if write_manifest:
            try:
                path = self.write_manifest(summary)
                summary["manifest_written_to"] = str(path)
            except Exception as e:
                # Don't fail the entire run just because manifest write failed
                summary["manifest_error"] = str(e)

        return summary


if __name__ == "__main__":
    pipe = FleetDataGeneration(
        "config/fleet/fleet_mixt.yaml",
        seed=123,           # optional; falls back to config.global_params.seed
        run=1,              # stored in config.global_params.run
        repro_mode=True,    # toggles reproducibility knobs for downstream
    )

    summary = pipe.run_pipeline(
        make_db=True,       # load generated data into DB
        preprocess=True,    # create preprocessed records
        write_manifest=True  # write reproducibility manifest
    )
