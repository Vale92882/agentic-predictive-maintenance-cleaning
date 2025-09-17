import random
import json
from datetime import timedelta
from faker import Faker
import shutil
import os
import pandas as pd
import yaml

fake = Faker()

# === License Plate ===


def generate_license_plate(country: str) -> str:
    country = country.lower()
    if country == 'usa':
        return f"{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}{random.randint(100, 999)}"
    elif country == 'germany':
        city_code = random.choice(['B', 'M', 'F', 'HH', 'S', 'K'])
        letters = ''.join(random.choices(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(1, 2)))
        numbers = random.randint(1, 9999)
        return f"{city_code}-{letters}{numbers}"
    elif country == 'italy':
        return f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{random.randint(100, 999)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}"
    elif country == 'colombia':
        return f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{random.randint(100, 999)}"
    elif country == 'luxembourg':
        return f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{random.randint(1000, 9999)}"
    elif country == 'romania':
        county_codes = ['B', 'CJ', 'TM', 'IS', 'BR', 'AG', 'AR', 'BV', 'CL', 'CT', 'DJ', 'GL', 'GR',
                        'HD', 'HR', 'IF', 'IL', 'MH', 'MS', 'NT', 'OT', 'PH', 'SB', 'SM', 'SV', 'TL', 'TR', 'VL', 'VN']
        county = random.choice(county_codes)
        digits = random.randint(10, 99)
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        return f"{county}{digits}{letters}"
    else:
        raise ValueError(f"Unsupported country: {country}")


def generate_device_id() -> str:
    return f"b{random.randint(100, 999)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}"


def generate_active_period() -> tuple[str, str]:
    start = fake.date_time_between(start_date='-10y', end_date='-1y')
    if random.random() < 0.7:
        end = "2050-01-01 00:00"
    else:
        end_dt = start + timedelta(days=random.randint(300, 1500))
        end = end_dt.strftime('%Y-%m-%d %H:%M')
    return start.strftime('%Y-%m-%d %H:%M'), end


def load_json_records(file_json_records):
    with open(file_json_records, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    # Convert each JSON string into a dictionary
    parsed_data = [json.loads(item) for item in raw_data]
    return parsed_data


def load_config(config_path):
    """
    Loads a YAML configuration file from the specified path.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def scratch_cleaned_log(noisy_maintenance_log, cleaned_maintenannce_log):
    if not os.path.exists(cleaned_maintenannce_log):
        shutil.copy(noisy_maintenance_log, cleaned_maintenannce_log)
        df = pd.read_csv(cleaned_maintenannce_log)
        df.loc[:, 'label'] = 'failed'
        df.to_csv(cleaned_maintenannce_log, index=False)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace(":", "_").replace(".", "_").replace("-", "_")
