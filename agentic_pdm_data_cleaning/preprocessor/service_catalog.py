from pathlib import Path

import pandas as pd


class ServiceCatalog:
    def __init__(self, path: str = Path('/home/vdimidov/VSCodeProjects/agentic_pdm_data_cleaning/data/synthetic') / 'workshop_service_catalog.csv'):
        """
        Initialize the WorkshopCatalog with workshop data.

        :param workshop_data: DataFrame containing workshop data.
        """
        self.workshop_data = pd.DataFrame()
        self.workshop_data = pd.read_csv(path)

    def get_systems(self):
        """
        Get the systems from the workshop data.

        :return: DataFrame containing systems.
        """
        return self.workshop_data['system'].unique().tolist()

    def get_subsystems(self):
        """
        Get the systems from the workshop data.

        :return: list containing subsystems.
        """
        return self.workshop_data['subsystem'].unique().tolist()

    def get_components(self):
        """
        Get the systems from the workshop data.

        :return: list containing components.
        """
        return self.workshop_data['component'].unique().tolist()

    def check_record_consistency(self, record):
        """
        Check whether a new record respects service_catalog constraints before inserting:
        - component must belong to one consistent subsystem
        - subsystem must belong to one consistent system
        - code must be associated only with components within a single subsystem

        :param record: dict with keys: system, subsystem, component, code
        :return: (True, None) if valid, (False, reason) if invalid
        """
        df = self.workshop_data
        sys = record['system']
        subsys = record['subsystem']
        comp = record['component']

        # 1. Check component → subsystem consistency
        existing_subsystems = df[df['component'] == comp]['subsystem'].unique()
        if len(existing_subsystems) > 0 and any(existing_subsystems != subsys):
            return False, f"Inconsistent subsystem for component '{comp}': existing = {existing_subsystems}, new = {subsys}"

        # 2. Check subsystem → system consistency
        existing_systems = df[df['subsystem'] == subsys]['system'].unique()
        if len(existing_systems) > 0 and any(existing_systems != sys):
            return False, f"Inconsistent system for subsystem '{subsys}': existing = {existing_systems}, new = {sys}"

        # 3. Check Activity

        return True, None


if __name__ == "__main__":
    workshop_service_catalog = ServiceCatalog()
    print(workshop_service_catalog.get_systems())
    print(workshop_service_catalog.get_subsystems())
    print(workshop_service_catalog.get_components())
