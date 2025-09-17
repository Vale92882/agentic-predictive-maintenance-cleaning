# fleet_agent_benchmark/records/discarded_record.py

import json
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class MaintenanceRecordState:
    """
    A structured container that holds a maintenance record along with
    contextual metadata related to its processing state.

    This class is used throughout the data cleaning and enrichment pipeline.
    It wraps the normalized JSON record and tracks metadata such as the
    current processing step, fault status, and failure reason.

    Attributes:
        json_record (str): The normalized maintenance record in JSON format.
        processing_step (str): A label indicating the current step of processing
            (e.g., 'normalized', 'validated', 'repaired').
        is_faulty (bool): Flag indicating whether the record is considered faulty.
        failure_reason (Optional[str]): Optional description of why the record was marked faulty.
    """
    json_record: str  # the normalized JSON
    processing_step: str
    is_faulty: bool
    failure_reason: Optional[str] = None


def save_records_state(path: str, records: List[MaintenanceRecordState]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in records],
                  f, ensure_ascii=False, indent=2)


def load_records_state(path: str) -> List[MaintenanceRecordState]:
    print(f"Loading records from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        raw_list = json.load(f)
    return [MaintenanceRecordState(**item) for item in raw_list]
