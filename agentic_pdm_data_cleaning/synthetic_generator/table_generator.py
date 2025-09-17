# tabular_generator.py

import os
from typing import List

import pandas as pd

from agentic_pdm_data_cleaning.synthetic_generator.base_generator import \
    BaseDataClassGenerator
from agentic_pdm_data_cleaning.utils import FileSystem

from .seeds import SeedBundle, derive_seed


class TableGenerator:
    def __init__(self, table_name: str, table_records_generator, class_records_generator: List[BaseDataClassGenerator], global_params: dict):
        self.table_name = table_name
        self.table_records_generator = table_records_generator
        self.class_records_generators = class_records_generator
        self.global_params = global_params
        base_seed = self.global_params.get("seed", 42)
        # also allow a per-table seed if present in the table section
        per_table_seed = (
            self.global_params.get(self.table_name, {}).get("seed", base_seed)
            if isinstance(self.global_params.get(self.table_name, {}), dict) else None
        )
        seed = per_table_seed if per_table_seed is not None else base_seed
        self.seed_bundle = SeedBundle(seed, self.table_name, "shuffle")

    def generate_dataset(self):
        clean_records, noisy_records = [], []
        for generator in self.class_records_generators:
            clean = generator.generate_clean_entries(
                self.table_records_generator, self.global_params)
            clean_records.extend(clean)

        for generator in self.class_records_generators:
            noisy = generator.generate_noisy_entries(
                self.table_records_generator, self.global_params)
            noisy_records.extend(noisy)

        self.seed_bundle.py.shuffle(noisy_records)
        df_noisy = pd.DataFrame(noisy_records)
        # Reorder df_clean to match the order of df_noisy by index
        df_clean = pd.DataFrame(clean_records)
        self.persist_dataset(df_clean, df_noisy)
        return df_clean, df_noisy

    def persist_dataset(self, cleaned_df, noisy_df):
        filesystem = FileSystem(context=self.global_params)
        os.makedirs(filesystem.common_synthetic_fleets_dir, exist_ok=True)
        cleaned_df.to_csv(filesystem.clean_table_file(
            table_name=self.table_name), index=False)
        noisy_df.to_csv(filesystem.noisy_table_file(
            table_name=self.table_name), index=False)
