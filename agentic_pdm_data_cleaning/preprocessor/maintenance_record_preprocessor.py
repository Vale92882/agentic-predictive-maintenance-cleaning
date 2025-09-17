import json
import re
from dataclasses import asdict
from importlib.resources import path

from dateutil import parser

from agentic_pdm_data_cleaning.domain import (MaintenanceRecordState,
                                              save_records_state)
from agentic_pdm_data_cleaning.preprocessor.service_catalog import \
    ServiceCatalog
from agentic_pdm_data_cleaning.utils import BatchToStream
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem


def is_lux_license_plate(plate: str) -> bool:
    # pattern_standard = r'^[A-Z]{3}\d{3}$'      # ABC123
    pattern_standard = r'^[A-Z]{2}\d{4}$'
    return bool(re.match(pattern_standard, plate.upper()))


class MaintenanceRecordProcessor:
    def __init__(self, fleet_config={}):
        self.config = fleet_config
        self.workshop_service_catalog = ServiceCatalog()

    def normalize_record(self, record_container: MaintenanceRecordState) -> MaintenanceRecordState:
        record_container.processing_step = "normalization"
        try:
            record = json.loads(record_container.json_record)
        except json.JSONDecodeError:
            record_container.is_faulty = True
            record_container.failure_reason = "Maintenance Record is expressed into an invalid JSON format"
            return record_container

        normalized = {}

        # Required ID
        normalized['id'] = record.get('id')

        # Optional string fields to normalize
        def normalize_str_field(field_name):
            value = record.get(field_name, '')
            if value is None:
                value = ''
            return str(value).strip().upper()

        # normalized['work_order_number'] = record.get('work_order_number', '').strip()
        normalized['license_plate'] = normalize_str_field('license_plate')
        normalized['system'] = normalize_str_field('system')
        normalized['subsystem'] = normalize_str_field('subsystem')
        normalized['component'] = normalize_str_field('component')
        normalized['activity'] = normalize_str_field('activity')
        normalized['work_description'] = normalize_str_field(
            'work_description')
        normalized['work_order_type'] = normalize_str_field('work_order_type')
        # normalized['workshop_location'] = normalize_str_field('workshop_location')

        # Normalize date fields
        def normalize_date(date_str):
            try:
                parsed_date = parser.parse(
                    date_str.strip(), dayfirst=False, yearfirst=True)
                return parsed_date.strftime('%Y-%m-%d')
            except (ValueError, TypeError, AttributeError):
                raise ValueError("Invalid date format")

        try:
            normalized['start_date'] = normalize_date(record.get('start_date'))
            normalized['end_date'] = normalize_date(record.get('end_date'))
        except ValueError as e:
            record_container.is_faulty = True
            record_container.failure_reason = str(e)
            return record_container

        record_container.json_record = json.dumps(
            normalized, ensure_ascii=False)
        return record_container


def generate_preprocessed_records(benchmark_id, fleet_config={}):

    fs = FileSystem(context=fleet_config)

    streaming_adapter = BatchToStream(fleet_config=fleet_config)
    discarded_records = []
    kept_records = []
    json_records = []

    with open(fs.preprocessing_json_records(), 'w', encoding='utf-8') as f:
        json.dump([r for r in streaming_adapter],
                  f, ensure_ascii=False, indent=2)

    print("Starting serialization of records.")

    for json_record in streaming_adapter:
        maintenance_record_id = json.loads(json_record).get("id", "N/A")
        # print(f"Processing record with id: {maintenance_record_id}")
        maintenance_record_container = MaintenanceRecordState(
            json_record=json_record, processing_step="initial", is_faulty=False)
        # print(f"Created MaintenanceRecordContainer: {maintenance_record_container}")
        normalized_record = MaintenanceRecordProcessor(fleet_config=fleet_config
                                                       ).normalize_record(maintenance_record_container)
        if normalized_record.is_faulty:
            discarded_records.append(normalized_record)
            continue

        filtered_record = MaintenanceRecordProcessor(
            fleet_config=fleet_config).filter_record(normalized_record)
        if filtered_record.is_faulty:
            discarded_records.append(filtered_record)
            continue

        kept_records.append(normalized_record)

    # define output path

    if discarded_records:
        print(
            f"Num. discarded records: {len(discarded_records)}/{len(kept_records) + len(discarded_records)}")

        # Persist discarded records
        save_records_state(
            fs.preprocessing_discarded_records_file(), discarded_records)
        print(
            f"Discarded records saved to {fs.preprocessing_discarded_records_file()}")

    # Persist kept records
    if kept_records:
        print(
            f"Num. kept records: {len(kept_records)}/{len(kept_records) + len(discarded_records)}")

        # Persist kept records
        save_records_state(
            fs.preprocessing_retained_records_file(), kept_records)
        print(
            f"Retained records saved to {fs.preprocessing_retained_records_file()}"
        )
    print("Serialization of records complete.")
