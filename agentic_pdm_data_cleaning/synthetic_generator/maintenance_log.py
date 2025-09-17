import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentic_pdm_data_cleaning.synthetic_generator.base_generator import \
    BaseDataClassGenerator
from agentic_pdm_data_cleaning.synthetic_generator.table_generator import \
    TableGenerator
from agentic_pdm_data_cleaning.utils import constants, llm
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem
from agentic_pdm_data_cleaning.utils.utility import generate_license_plate

from .seeds import SeedBundle, derive_seed


class MaintenanceTableRecordGenerator:
    counter = 0  # Class variable shared across all instances

    def __init__(self, context: dict):
        self.seed = context.get("seed", 42)
        self.bundle = SeedBundle(self.seed, "maintenance_log")
        self.rb = self.bundle.py
        self.rng = self.bundle.np
        self.fake = self.bundle.faker

        self.all_vehicles = set(context.get("license_plates", []))
        # Use a deterministically shuffled list instead of a set (set.pop is order-dependent)
        self.vehicles_without_maitenance_activity = sorted(
            context.get("license_plates", []))
        self.rb.shuffle(self.vehicles_without_maitenance_activity)
        self.license_plates = context.get("license_plates", [])
        self.activity_tuples = context.get("activity_tuples", [])
        self.model = context.get("model", "llama3.1:latest")
        self.wear_state = context.get("wear_state", {})
        self.country = context.get("country", "luxembourg")
        self.devices = context.get("devices", [])
        self.work_order_numbers = set()

    def _generate_entry(self, license_plate, global_params: dict, corrupted_record=0, noise_label=constants.LABEL_MAIN_RECORD_CLEAN, noise_type=constants.LABEL_NOISE_TYPE_NONE, label=constants.LABEL_CLASS_ACCEPT):
        start_extract_date = global_params.get(
            'start_extract_date', '01-01-2021')
        end_extract_date = global_params.get('end_extract_date', '01-01-2023')

        system, subsystem, component, activity, strategy = self.rb.choice(
            self.activity_tuples)
        vehicle_state = self.wear_state.get(license_plate, {
            "odometer_km": self.rb.randint(0, 300000),
            "vehicle_age_months": self.rb.randint(1, 120)
        })
        odometer_km = vehicle_state["odometer_km"]
        vehicle_age = vehicle_state["vehicle_age_months"]

        # freq_info = self.frequency_data.get(component, {})
        # failure_rate = freq_info.get("failure_rate_per_100k_km")
        # pm_interval_km = freq_info.get("pm_interval_km")
        # pm_interval_time = freq_info.get("pm_interval_time_months")
        # pm_mandatory = freq_info.get("pm_mandatory")
#
        # is_pm_due = False
        # if pm_interval_km and odometer_km >= pm_interval_km:
        #    is_pm_due = True
        # if pm_interval_time and vehicle_age >= pm_interval_time:
        #    is_pm_due = True
#
        # is_failure = False
        # if failure_rate:
        #    lambda_per_km = failure_rate / 100000
        #    if lambda_per_km > 0:
        #        sampled_km = np.random.exponential(1 / lambda_per_km)
        #        if sampled_km <= odometer_km:
        #            is_failure = True

        work_order_type = self.rb.choice(["corrective"])

        start_date = self.fake.date_between(
            start_date=datetime.strptime(start_extract_date, "%d-%m-%Y"),
            end_date=datetime.strptime(end_extract_date, "%d-%m-%Y")
        )
        end_date = start_date + timedelta(days=self.rb.randint(4, 7))
        while True:
            work_order_number = f"WO{self.rb.randint(100000, 999999)}"
            if work_order_number not in self.work_order_numbers:
                self.work_order_numbers.add(work_order_number)
                break
        workshop_location = self.rb.choice(
            ["Workshop A", "Workshop B", "Workshop C"])
        work_description = f"{activity} performed on {component.lower()} ({work_order_type})."
        if global_params.get("use_llm_generation", False):
            try:
                model_name = "llama3.1:latest"
                model = llm.get_model(model_name)
                prompt = (f"Write ONE concise, technician-style maintenance note (1 sentences, 10–15 words). "
                          "Use the exact terms:\n"
                          f"- activity: {activity}\n"
                          f"- component: {component}\n"
                          f"Keep in consideration that the component belongs to the hierarchy {system}-{subsystem} and it's about a {work_order_type} maintenance session."
                          "Do not mention other activities, no extra words, no costs, no guarantees like 'safe to drive', no brand names, no personal data."
                          "The sentence must end into a maintenance log and describe the work perfomed by the technician.")
                model_settings = {'temperature': 0.0}
                agent = Agent(
                    model=model,
                    name="MaintenanceNotesGenerator",
                    output_retries=30,
                    model_settings=model_settings,
                )
                work_description = agent.run_sync(
                    user_prompt=prompt,
                ).output.strip()
                if (not work_description) \
                        or (activity.lower() not in work_description.lower()) \
                        or (component.lower() not in work_description.lower()):
                    work_description = f"{activity} {self.fake.word()} {component.lower()}"
            except Exception as e:
                print(f"Error: {e}")
                work_description = f"{activity} performed on {component.lower()} ({work_order_type})."
        work_description = work_description.replace('"', '')
        record = {
            "id": str(self.__class__.counter),
            "work_order_number": work_order_number,
            "start_date": start_date,
            "end_date": end_date,
            "license_plate": license_plate,
            "system": system,
            "subsystem": subsystem,
            "component": component,
            "activity": activity,
            "work_description": work_description,
            "work_order_type": work_order_type,
            # "workshop_location": workshop_location,
            "corrupted_record": corrupted_record,
            "noise_label": noise_label,
            "noise_type": noise_type,
            "label": label
        }
        self.__class__.counter += 1
        return record

    def generate_single_entry_per_vehicle(self, count, global_params: dict):
        results = []
        for _ in range(count):
            license_plate = self.vehicles_without_maitenance_activity.pop(0)
            record = self._generate_entry(license_plate, global_params)
            results.append(record)
        return results

    def generate_entry_with_vehicle_not_in_fleet(self, count, global_params: dict):
        results = []
        for _ in range(count):
            # Generate a license plate not in the fleet
            random_license_plate = generate_license_plate(country=self.country)
            while random_license_plate in self.all_vehicles:
                random_license_plate = generate_license_plate(
                    country=self.country)
            record = self._generate_entry(
                self.rb.choice(self.license_plates), global_params)
            record["license_plate"] = random_license_plate
            results.append(record)
        return results


class CleanMainTabEntryGenerator(BaseDataClassGenerator):
    def __init__(self, global_params: dict = {}):
        super().__init__(global_params)
        self.max_days_shift = global_params.get('max_days_shift', 10)
        self.noise_label = global_params.get(
            'noise_label', constants.LABEL_MAIN_RECORD_CLEAN)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_NONE)
        self.clean_records = []
        self.noisy_records = []
        self.seed = global_params.get('seed', 42)
        self.bundle = SeedBundle(self.seed, self.__class__.__name__)
        self.rb = self.bundle.py
        self.rng = self.bundle.np

    def generate_clean_entries(self, table_records_generator, global_params: dict):
        self.clean_records.extend(table_records_generator.generate_single_entry_per_vehicle(
            self.count, global_params))
        return self.clean_records

    def generate_noisy_entries(self, table_records_generator, global_params: dict):
        self.noisy_records = copy.deepcopy(self.clean_records)
        return self.noisy_records


class WrongEndDateGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params: dict = {}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            'noise_label', constants.LABEL_MAIN_RECORD_WRONG_END_DATE)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)
        self.max_days_shift = global_params.get('max_days_shift', 10)

    def generate_noisy_entries(self, table_records_generator, global_params: dict):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            shift_days = self.rb.randint(1, self.max_days_shift)
            record["end_date"] = record["end_date"] + \
                timedelta(days=shift_days)
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records

# Add/replace your previous generator with this version


class DigitalSystemTestGenerator(CleanMainTabEntryGenerator):

    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label",
            constants.LABEL_MAIN_RECORD_TEST
        )
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_GENERATIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_REJECT)
        # Activities that look like tests (English)
        self.activity_aliases = global_params.get(
            "activity_aliases",
            [
                "digital system test",
                "system calibration",
                "platform onboarding",
                "telematics setup",
            ],
        )

    def generate_clean_entries(self, table_records_generator, global_params):
        # Generative: no clean entries
        return []

    def generate_noisy_entries(self, table_records_generator, global_params: dict):
        vehicle_not_in_fleet_records = table_records_generator.generate_entry_with_vehicle_not_in_fleet(
            count=self.count,
            global_params=global_params
        )
        for record in vehicle_not_in_fleet_records:
            random_number = self.rb.randint(0, 2)
            if random_number == 0:
                record["license_plate"] = self.rb.choice(
                    ["", "-", "test", "Test"])
                prefix = self.rb.choice(
                    ["Digital", "Calibration", "Telematics", "Test"])
                record["system"] = prefix + "System"
                record["subsystem"] = prefix + "TestSubsystem"
                record["component"] = prefix + "TestComponent"
            elif random_number == 1:
                record["license_plate"] = self.rb.choice(["test", "Test"])
                common_pattern = self.rb.choice(["", "-"])
            else:
                record["system"] = "TelematicsSystem"
                record["subsystem"] = "TelematicsSubsystem"
                record["component"] = "TelematicsComponent"
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
        self.noisy_records.extend(vehicle_not_in_fleet_records)
        return self.noisy_records


class OutOfFleetVehicleGenerator(CleanMainTabEntryGenerator):
    """
    Generates records with vehicles not present in the fleet.
    """

    def __init__(self, global_params: dict = {}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            'noise_label', constants.LABEL_MAIN_RECORD_VEHICLE_NOT_IN_FLEET)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_GENERATIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_REJECT)
        self.distribution = global_params.get('distribution', "uniform")

    def generate_clean_entries(self, table_records_generator, global_params: dict):
        return []

    def generate_noisy_entries(self, table_records_generator, global_params: dict):
        vehicle_not_in_fleet_records = table_records_generator.generate_entry_with_vehicle_not_in_fleet(
            count=self.count, global_params=global_params)
        for record in vehicle_not_in_fleet_records:
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
        self.noisy_records.extend(vehicle_not_in_fleet_records)
        return self.noisy_records


def generate_maintenance_log(benchmark_id, fleet_config):
    print("Starting maintenance log generation...")
    global_params = fleet_config.get("global_params", {})
    country = global_params.get('country', 'luxembourg')
    fleet_id = global_params.get('fleet_id', 'fleet_n')
    maintenance_log_config = fleet_config["maintenance_log"]
    extract_start_date = maintenance_log_config.get(
        'start_extract_date', '01-01-2021')
    extract_end_date = maintenance_log_config.get(
        'end_extract_date', '01-01-2028')
    seed = global_params.get('seed', 42)

    model = global_params.get('model', 'llama3.1:latest')

    filesystem = FileSystem(context=fleet_config)

    # Load noisy fleet data
    df_fleet_registry = pd.read_csv(
        filesystem.noisy_table_file("fleet_registry"))
    df_service_catalog = pd.read_csv(maintenance_log_config.get(
        "service_catalog_path", "resources/service_catalog.csv"))

    license_plates = df_fleet_registry['license_plate'].dropna(
    ).unique().tolist()
    activity_tuples = df_service_catalog[[
        "system", "subsystem", "component", "activity", "strategy_type"]].dropna().values.tolist()

    freq_df = df_service_catalog.drop_duplicates(
        "component").set_index("component")
    # frequency_data = freq_df[["failure_rate_per_100k_km", "pm_interval_km",
    #                           "pm_interval_time_months", "pm_mandatory"]].to_dict(orient="index")

    bundle = SeedBundle(seed, "fleet_registry", "record")

    wear_state = {
        plate: {
            "odometer_km": bundle.py.randint(0, 300000),
            "vehicle_age_months": bundle.py.randint(1, 120)
        } for plate in license_plates
    }

    context = {
        "devices": df_fleet_registry,
        "ontology": df_service_catalog,
        "license_plates": license_plates,
        "activity_tuples": activity_tuples,
        "model": model,
        "wear_state": wear_state,
        "country": country,
        "fleet_id": fleet_id,
        'start_extract_date': extract_start_date,
        'end_extract_date': extract_end_date,
        'seed': seed,
    }

    # Instantiate clean generator
    table_record_generator = MaintenanceTableRecordGenerator(
        context
    )

    # Instantiate noise generators
    generators = instantiate_generators(
        generator_configs=maintenance_log_config["generators"],
        extra_kwargs={"seed": seed}
    )

    # Create and run generator
    generator = TableGenerator(
        table_name="maintenance_log",
        table_records_generator=table_record_generator,
        class_records_generator=generators,
        global_params=global_params,
    )

    generator.generate_dataset()
    print("Maintenance log generation completed.")


def instantiate_generators(generator_configs, extra_kwargs=None):
    instances = []
    extra_kwargs = extra_kwargs or {}
    for gen in generator_configs:
        gen_class = GENERATOR_CLASSES[gen["type"]]
        params = {**gen["params"], **extra_kwargs}
        instances.append(gen_class(params))
    return instances


class VehicleIdMisalignmentGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", constants.LABEL_MAIN_RECORD_WRONG_VEHICLE_ID)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.context = global_params.get("context", {})

    def generate_noisy_entries(self, table_records_generator, global_params):
        df_devices = table_records_generator.devices
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            license_plate = record["license_plate"]
            column_name = self.rb.choice(["device_id", "name", "VIN"])
            record["license_plate"] = df_devices.loc[
                df_devices["license_plate"] == license_plate, column_name
            ].iloc[0]
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class WrongStartDateGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get("noise_label", "wrong_start_date")
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            day_gap = (record["end_date"] - record["start_date"]).days - 1
            shift_days = self.rb.randint(1, min(self.max_days_shift, day_gap))
            record["start_date"] = record["start_date"] + \
                timedelta(days=shift_days)
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class WrongDatesGenerator(CleanMainTabEntryGenerator):
    def __init__(self, maintenance_activities_generator, global_params={}):
        super().__init__(maintenance_activities_generator, global_params)
        self.noise_label = global_params.get(
            "noise_label", constants.LABEL_MAIN_RECORD_INCORRECT_DATES)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            day_gap = (record["end_date"] - record["start_date"]).days - 1
            shift_start_days = self.rb.randint(
                1, min(self.max_days_shift, day_gap))
            shift_end_days = self.rb.randint(1, self.max_days_shift)
            record["start_date"] += timedelta(days=shift_start_days)
            record["end_date"] += timedelta(days=shift_end_days)
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class InvalidComponentGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", "invalid_component")
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            record["component"] = "InvalidComponentX"
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class MismatchedHierarchyGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", "mismatched_hierarchy")
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.context = global_params.get("context", {})
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        df_ontology = self.context.get("ontology")
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            wrong_row = df_ontology.sample(1, random_state=derive_seed(
                self.seed, "ontology_sample", record.get("id", 0))).iloc[0]
            record["system"] = wrong_row["system"]
            record["subsystem"] = wrong_row["subsystem"]
            record["component"] = wrong_row["component"]
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class OntologyFieldSwapGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", "ontology_field_swap")
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            record["system"], record["component"] = record["component"], record["system"]
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class BlankOntologyFieldGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", "blank_ontology_field")
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            field = self.rb.choice(
                ["system", "subsystem", "component", "activity"])
            record[field] = ""
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class MissingActivityGenerator(CleanMainTabEntryGenerator):
    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get("noise_label", "missing_activity")
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            record["activity"] = ""
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class InvalidValueGenerator(CleanMainTabEntryGenerator):
    """
    Introduces typos into categorical ontology fields (system, subsystem, component, activity).
    Only applies character-level edits (swap, delete, insert) to push them outside the expected domain.

    Config example:
      - type: CategoricalTypoGenerator
        params:
          count: 200
          fields: ["system", "subsystem", "component", "activity"]
    """

    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", constants.LABEL_FIELD_INTEGRITY_CATEGORICAL_TYPO)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)
        self.context = global_params.get("context", {})
        self.fields = global_params.get(
            "fields", ["license_plate", "system",
                       "subsystem", "component", "activity"]
        )

    # --- typo helpers ---
    def _swap_adjacent(self, s: str) -> str:
        if len(s) < 2:
            return s + self.rb.choice("abcdefghijklmnopqrstuvwxyz")
        i = self.rb.randrange(len(s)-1)
        return s[:i] + s[i+1] + s[i] + s[i+2:]

    def _delete_char(self, s: str) -> str:
        if not s:
            return s
        i = self.rb.randrange(len(s))
        return s[:i] + s[i+1:]

    def _insert_char(self, s: str) -> str:
        i = self.rb.randrange(len(s)+1)
        ch = self.rb.choice("?!+@%abcdefghijklmnopqrstuvwxyz")
        return s[:i] + ch + s[i:]

    def _typo(self, s: str) -> str:
        if self.rb.random() < 0.33:
            return self._swap_adjacent(s)
        if self.rb.random() < 0.66:
            return self._delete_char(s)
        return self._insert_char(s)

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            # ensure at least one field is selected
            field_to_corrupt = self.rb.choice(self.fields)
            value_field = record[field_to_corrupt]
            corrupted_value = self._typo(value_field)
            if field_to_corrupt == "license_plate":
                while corrupted_value in table_records_generator.devices["license_plate"].tolist():
                    corrupted_value = self._typo(value_field)
            record[field_to_corrupt] = corrupted_value
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


class MissingValueGenerator(CleanMainTabEntryGenerator):
    """
    Introduces typos into categorical ontology fields (system, subsystem, component, activity).
    Only applies character-level edits (swap, delete, insert) to push them outside the expected domain.

    Config example:
      - type: MissingValueGenerator
        params:
          count: 200
          fields: ["system", "subsystem", "component", "activity"]
    """

    def __init__(self, global_params={}):
        super().__init__(global_params)
        self.noise_label = global_params.get(
            "noise_label", constants.LABEL_FIELD_INTEGRITY_MISSING_VALUE)
        self.noise_type = global_params.get(
            'noise_type', constants.LABEL_NOISE_TYPE_CORRUPTIVE)
        self.label = global_params.get('label', constants.LABEL_CLASS_UPDATE)
        self.context = global_params.get("context", {})
        self.fields = global_params.get(
            "fields", ["system", "subsystem", "component", "activity"]
        )

    def generate_noisy_entries(self, table_records_generator, global_params):
        for clean in self.clean_records:
            record = copy.deepcopy(clean)
            # ensure at least one field is selected
            field_to_corrupt = self.rb.choice(self.fields)
            record[field_to_corrupt] = ""
            record["corrupted_record"] = 1
            record["noise_label"] = self.noise_label
            record["noise_type"] = self.noise_type
            record["label"] = self.label
            self.noisy_records.append(record)
        return self.noisy_records


GENERATOR_CLASSES = {
    "OutOfFleetVehicleGenerator": OutOfFleetVehicleGenerator,
    "VehicleIdMisalignmentGenerator": VehicleIdMisalignmentGenerator,
    "WrongEndDateGenerator": WrongEndDateGenerator,
    "MissingActivityGenerator": MissingActivityGenerator,
    "CleanMainTabEntryGenerator": CleanMainTabEntryGenerator,
    "InvalidComponentGenerator": InvalidComponentGenerator,
    "MismatchedHierarchyGenerator": MismatchedHierarchyGenerator,
    "OntologyFieldSwapGenerator": OntologyFieldSwapGenerator,
    "BlankOntologyFieldGenerator": BlankOntologyFieldGenerator,
    "WrongStartDateGenerator": WrongStartDateGenerator,
    "WrongDatesGenerator": WrongDatesGenerator,
    "DigitalSystemTestGenerator": DigitalSystemTestGenerator,
    "InvalidValueGenerator": InvalidValueGenerator,
    "MissingValueGenerator": MissingValueGenerator
}
