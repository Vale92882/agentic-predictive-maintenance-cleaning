# base_generator.py
from abc import ABC, abstractmethod
from typing import List


class BaseDataClassGenerator(ABC):
    def __init__(self, global_params: dict = {}):
        self.count = global_params.get('count', 1)
        self.noise_label = global_params.get('noise_label', "none")
        self.distribution = global_params.get('distribution', "uniform")

    @abstractmethod
    def generate_clean_entries(self, table_records_generator, global_params: dict) -> List[dict]:
        pass

    @abstractmethod
    def generate_noisy_entries(self, table_records_generator, global_params: dict) -> List[dict]:
        pass
