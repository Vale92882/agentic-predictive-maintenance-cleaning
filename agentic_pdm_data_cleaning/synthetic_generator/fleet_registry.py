import copy
import random

from faker import Faker

from agentic_pdm_data_cleaning.synthetic_generator.base_generator import \
    BaseDataClassGenerator
from agentic_pdm_data_cleaning.synthetic_generator.seeds import SeedBundle
from agentic_pdm_data_cleaning.synthetic_generator.table_generator import \
    TableGenerator
from agentic_pdm_data_cleaning.utils import (constants, generate_device_id,
                                             generate_license_plate)


# === Abstract Base Generator ===
class FleetRegistryRecordGenerator():
    def __init__(self, global_params: dict = {}):
        self.count = global_params.get("count", 20)
        self.country = global_params.get("country", "luxembourg")
        self.noise_label = global_params.get(
            "noise_label", constants.LABEL_DEVICE_TAB_NONE)
        # Scoped RNGs
        self.seed = global_params.get("seed", 42)
        self.bundle = SeedBundle(self.seed, "fleet_registry", "record")
        self.rb = self.bundle.py
        self.fake = self.bundle.faker

    def _generate_entry(self, global_params: dict):
        device_id = generate_device_id()
        license_plate = generate_license_plate(
            global_params.get("country", self.country))
        entry = {
            "device_id": device_id,
            "name": f"({device_id})",
            "license_plate": license_plate,
            "VIN": self.fake.unique.vin(),
            "active_from": global_params.get("monitoring_start_date", "01-01-2017 00:00"),
            "active_to": global_params.get("monitoring_end_date", "01-01-2050 00:00"),
            "corrupted_record": 0,
            "noise_label": "none",
            "label": "accept",
        }
        return entry


class CleanFleetRegistryEntryGenerator(BaseDataClassGenerator):

    def __init__(self, global_params: dict = {}):

        self.count = global_params.get("count", 20)
        self.noise_label = global_params.get(
            "noise_label", constants.LABEL_DEVICE_TAB_NONE)
        self.country = global_params.get("country", 'luxembourg')
        self.distribution = global_params.get("distribution", 'uniform')
        self.clean_maintenance_records = []
        self.noisy_maintenance_records = []

    def generate_clean_entries(self, table_records_generator, global_params: dict = {}) -> dict:
        for _ in range(self.count):
            entry = table_records_generator._generate_entry(global_params)
            self.clean_maintenance_records.append(entry)
        return self.clean_maintenance_records

    def generate_noisy_entries(self, table_records_generator, global_params: dict = {}) -> dict:
        self.noisy_maintenance_records = copy.deepcopy(
            self.clean_maintenance_records)
        return self.noisy_maintenance_records


class CorruptedVINDeviceEntryGenerator(CleanFleetRegistryEntryGenerator):
    def __init__(self, global_params: dict = {}):
        super().__init__(global_params=global_params)
        self.noise_label = "corrupted_vin_device_entry"
        self.label = "update"

    def generate_noisy_entries(self, record_generator, global_params={}):
        for clean_record in self.clean_maintenance_records:
            # Truncate VIN to 10 characters
            record = copy.deepcopy(clean_record)
            record['VIN'] = record.get('VIN', '')[0:10]
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["label"] = self.label
            self.noisy_maintenance_records.append(record)
        return self.noisy_maintenance_records


class NoisyNameDeviceEntryGenerator(CleanFleetRegistryEntryGenerator):

    def __init__(self, global_params: dict = {}):
        super().__init__(global_params=global_params)
        self.faker = SeedBundle(global_params.get(
            "seed"), "fleet_registry", "noisy_name").faker
        self.noise_label = "noisy_name_device_entry"
        self.label = "update"

    def generate_noisy_entries(self, record_generator, global_params={}):
        for clean_record in self.clean_maintenance_records:
            # Truncate VIN to 10 characters
            record = copy.deepcopy(clean_record)
            plate = record["license_plate"]
            noise_variants = [
                f"{plate} {self.faker.word().capitalize()}",
                f"{plate} Line {random.randint(1, 99)}",
                f"{self.faker.random_element(['TRUCK', 'BUS', 'VAN'])}-{plate}",
                f"({plate}) vehicle_type: {self.faker.random_element(['bus', 'truck', 'van'])}",
                f"{plate}",
                f"{plate} (depot: {self.faker.random_uppercase_letter()}{self.faker.random.randint(1, 9)})",
                f"{plate} {self.faker.random_element(['truck', 'bus', 'van'])}",
                f"{plate} # route {self.faker.random.randint(10, 999)}",
                f"{self.faker.first_name()} {plate}",
                f"{plate} {self.faker.random_element(['decommissioned', 'active', 'maintenance'])}",
            ]
            record["name"] = SeedBundle(global_params.get(
                "seed"), "fleet_registry", "name_choice", record["license_plate"]).py.choice(noise_variants)
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["label"] = self.label
            self.noisy_maintenance_records.append(record)
        return self.noisy_maintenance_records


class UnstandardizedNameDeviceEntryGenerator(CleanFleetRegistryEntryGenerator):

    def __init__(self, global_params: dict = {}):
        super().__init__(global_params=global_params)
        self.noise_label = "unstandardized_name"

    def generate_noisy_entries(self, record_generator, global_params={}):
        for clean_record in self.clean_maintenance_records:
            # Truncate VIN to 10 characters
            record = copy.deepcopy(clean_record)
            record["name"] = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2)) + str(random.randint(10, 99))}"
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["label"] = self.label
            self.noisy_maintenance_records.append(record)
        return self.noisy_maintenance_records


DEVICE_TABLE_GENERATORS = {
    "CleanFleetRegistryEntryGenerator": CleanFleetRegistryEntryGenerator,
    "CorruptedVINDeviceEntryGenerator": CorruptedVINDeviceEntryGenerator,
    "NoisyNameDeviceEntryGenerator": NoisyNameDeviceEntryGenerator,
}


def instantiate_generators(generator_configs, generator_map=DEVICE_TABLE_GENERATORS):
    instances = []
    for generator in generator_configs:
        gen_class = generator_map[generator["type"]]
        instances.append(gen_class(generator["params"]))
    return instances


def generate_fleet_registry(benchmark_id, fleet_config):

    print("Starting Fleet Registry generation...")
    device_table_config = fleet_config["fleet_registry"]
    global_params = fleet_config.get("global_params", {})
    global_params['benchmark_id'] = benchmark_id

    fleet_registry_generator = TableGenerator(
        table_name="fleet_registry",
        table_records_generator=FleetRegistryRecordGenerator(
            global_params=global_params),
        class_records_generator=instantiate_generators(
            device_table_config["generators"], generator_map=DEVICE_TABLE_GENERATORS),
        global_params=global_params
    )

    fleet_registry_generator.generate_dataset()
    print("Fleet Registry dataset generation completed.")


if __name__ == "__main__":
    print("Starting dataset generation...")
    country = "luxembourg"
    # Generate Dataset
    device_table_generator = TableGenerator(
        table_name="fleet_registry",
        table_records_generator=FleetRegistryRecordGenerator(
            global_params={"country": country}),
        class_records_generator=[
            CleanFleetRegistryEntryGenerator(country=country, count=20),
            CorruptedVINDeviceEntryGenerator(count=10),
            NoisyNameDeviceEntryGenerator(count=10)],
        global_params={"country": country,
                       "monitoring_start_date": "01-01-2017 00:00",
                       "monitoring_end_date": "01-01-2050 00:00"
                       }
    )
    cleaned_df, noisy_df = device_table_generator.generate_dataset()
    print("Dataset generation completed.")
    cleaned_df.to_csv("cleaned_device_table.csv", index=False)
    noisy_df.to_csv("noisy_device_table.csv", index=False)
