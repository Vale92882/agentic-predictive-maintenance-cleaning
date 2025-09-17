import os
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from dateutil import parser

from agentic_pdm_data_cleaning.utils import FileSystem

from .seeds import SeedBundle, derive_seed


class BaseSignalGenerator(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs

    def set_context(self, context):
        self.context = context
        self.fleet_id = context.get('fleet_id', 'default_fleet')
        self.seed = context.get('seed', 42)
        self.fs = FileSystem(context=context)
        self.start_monitoring_date = context.get(
            'start_monitoring_date', '01-01-2020')
        self.end_monitoring_date = context.get(
            'end_monitoring_date', '01-01-2025')

    @abstractmethod
    def generate(self, global_params: dict = {}) -> pd.DataFrame:
        pass


class CleanOdometerGenerator(BaseSignalGenerator):
    def generate(self, global_params: dict = {}) -> pd.DataFrame:

        # Load devices and maintenance log
        fleet_registry_file = self.fs.clean_table_file("fleet_registry")
        maintenance_file = self.fs.clean_table_file("maintenance_log")

        df_registry_file = pd.read_csv(fleet_registry_file).sort_values(
            ["license_plate", "device_id"])
        df_maintenance = pd.read_csv(maintenance_file).sort_values(
            ["license_plate", "start_date", "end_date"])

        df_maintenance['start_date'] = pd.to_datetime(
            df_maintenance['start_date'])
        df_maintenance['end_date'] = pd.to_datetime(df_maintenance['end_date'])

        simulation_start = parser.parse(global_params.get(
            'start_monitoring_date', '01-01-2020')).strftime("%Y-%m-%d")
        simulation_end = parser.parse(global_params.get(
            'end_monitoring_date', '01-01-2025')).strftime("%Y-%m-%d")

        signal_id = self.context.get('signal_id', 'default_signal_if')
        # Prepare output directory
        signal_dir = self.fs.get_signal_dir(signal_id)
        all_signals = []
        os.makedirs(signal_dir, exist_ok=True)

        for _, row in df_registry_file.iterrows():
            license_plate = row['license_plate']
            device_id = row['device_id']

            maint = df_maintenance[df_maintenance['license_plate']
                                   == license_plate]

            maintenance_periods = list(
                zip(maint['start_date'], maint['end_date']))

            initial_odometer = self.context['wear_state'].get(
                license_plate, {}).get('odometer_km')
            if initial_odometer is None:
                initial_odometer = SeedBundle(
                    self.seed, "signals", "init_odo", device_id).np.integers(0, 300000)

            df_signal = simulate_odometer(
                device_id=device_id,
                start_date=simulation_start,
                end_date=simulation_end,
                initial_odometer=initial_odometer,
                mean_daily_km=self.context['mean_daily_km'],
                std_daily_km=self.context['std_daily_km'],
                maintenance_periods=maintenance_periods,
                no_drive_prob=self.context['no_drive_prob'],
                seed=self.seed
            )

            # Save per vehicle
            df_signal.to_csv(
                f"{signal_dir}/{device_id}_odometer_signal.csv", index=False)
            all_signals.append(df_signal)

        print("✅ Odometer signals generated for entire fleet.")
        return pd.concat(all_signals, ignore_index=True)


SIGNAL_GENERATORS = {
    "CleanOdometerGenerator": CleanOdometerGenerator,
}


def instantiate_signal_generators(generator_configs, generator_map=SIGNAL_GENERATORS):
    instances = []
    for generator in generator_configs:
        gen_class = generator_map[generator["type"]]
        params = generator.get("params", {})
        if len(params) == 0:
            instances.append(gen_class())
            continue
        instances.append(gen_class(**generator["params"]))
    return instances


def simulate_odometer(
    device_id: str,
    start_date: str,
    end_date: str,
    initial_odometer: float,
    mean_daily_km: float,
    std_daily_km: float,
    maintenance_periods: list = None,
    no_drive_prob: float = 0.1,
    seed: int = 42
):
    rng = np.random.default_rng(derive_seed(
        seed, "signals", "odometer", device_id))

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_distances = rng.normal(
        mean_daily_km, std_daily_km, size=len(dates))
    no_drive_mask = rng.random(len(dates)) < no_drive_prob
    daily_distances[no_drive_mask] = 0
    daily_distances = np.maximum(daily_distances, 0)

    odometer_readings = []
    reading_dates = []
    current_odometer = int(initial_odometer)
    distance_travelled_per_day = []
    for i, date in enumerate(dates):
        # Check maintenance periods
        in_maintenance_middle = False
        is_maintenance_start_or_end = False

        if maintenance_periods:
            for period in maintenance_periods:
                start, end = pd.to_datetime(
                    period[0]), pd.to_datetime(period[1])
                if start < date < end:
                    in_maintenance_middle = True
                    break
                elif date == start or date == end:
                    is_maintenance_start_or_end = True
                    break

        if in_maintenance_middle:
            # No reading for maintenance middle days
            distance_travelled = 0
        elif is_maintenance_start_or_end:
            # Half distance for start or end maintenance days
            distance_travelled = daily_distances[i] / 2
        else:
            # Normal day
            distance_travelled = daily_distances[i]

        current_odometer += distance_travelled
        reading_dates.append(date)
        odometer_readings.append(int(current_odometer))
        distance_travelled_per_day.append(int(distance_travelled))

    df = pd.DataFrame({
        'device_id': device_id,
        'date': reading_dates,
        'odometer_value': odometer_readings,
        'km_travelled': distance_travelled_per_day
    })

    return df


def generate_signals(benchmark_id, fleet_config):

    global_params = fleet_config.get("global_params", {})
    context = {
        'fleet_id': global_params.get('fleet_id', 'default_fleet'),
        'mean_daily_km': 200,
        'std_daily_km': 20,
        'no_drive_prob': 0.1,
        'start_monitoring_date': fleet_config.get('monitoring_start_date', '01-01-2020'),
        'end_monitoring_date': fleet_config.get('monitoring_end_date', '01-01-2025'),
        'run_id': global_params.get('run_id', 0),
        'seed': global_params.get('seed', 42),
        'benchmark_id': benchmark_id
    }

    filesystem = FileSystem(context=context)

    # Load noisy fleet data
    df_fleet_registry = pd.read_csv(
        filesystem.noisy_table_file("fleet_registry"))
    license_plates = df_fleet_registry['license_plate'].dropna(
    ).unique().tolist()
    wear_state = {
        plate: {
            "odometer_km": random.randint(0, 300000),
            "vehicle_age_months": random.randint(1, 120)
        } for plate in license_plates
    }

    context['wear_state'] = wear_state

    print("Starting signal generation...")

    for signal_name, signal_conf in fleet_config.get('signals', {}).items():
        if signal_name in ["seed"]:
            continue
        print(f"Processing signal type: {signal_name}")

        context['signal_id'] = signal_name

        clean_generators = instantiate_signal_generators(
            signal_conf.get("generators", SIGNAL_GENERATORS))

        for gen in clean_generators:
            gen.set_context(context)

        clean_dfs = [gen.generate() for gen in clean_generators]
        if len(clean_dfs) == 0:
            print("No  signal generators provided, returning empty dataset.")
            return pd.DataFrame()
    print("Signal generation complete.")
    return pd.DataFrame()


# Example usage
if __name__ == "__main__":

    context = {
        'fleet_id': 'fleet_1',
        'mean_daily_km': 200,
        'std_daily_km': 20,
        'no_drive_prob': 0.1,
    }

    filesystem = FileSystem()

    # Load noisy fleet data
    df_devices = pd.read_csv(filesystem.noisy_devices_file(fleet_id='fleet_1'))
    license_plates = df_devices['license_plate'].dropna().unique().tolist()
    wear_state = {
        plate: {
            "odometer_km": random.randint(0, 300000),
            "vehicle_age_months": random.randint(1, 120)
        } for plate in license_plates
    }

    context['wear_state'] = wear_state

    df_odometer = simulate_odometer(
        device_id="V123",
        start_date="2025-01-01",
        end_date="2025-01-10",
        initial_odometer=50000,
        mean_daily_km=200,
        std_daily_km=20,
        maintenance_periods=[],
        no_drive_prob=0.15,
        seed=42
    )

    df_odometer.to_csv("simulated_odometer.csv", index=False)
