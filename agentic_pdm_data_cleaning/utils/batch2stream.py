import json

import pandas as pd

from agentic_pdm_data_cleaning.utils.config_parser import ConfigSingleton
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem


class BatchToStream:
    """
    BatchToStream is an iterable class that converts a batch of maintenance log records from a CSV file into a stream of JSON-formatted records.

    Args:
        fleet_config (dict): Configuration dictionary containing fleet-specific parameters, such as 'fleet_id'.
        max_iterations (int, optional): Maximum number of records to stream. If set to -1 or not provided, streams all records. Defaults to -1.

    Attributes:
        config: Singleton configuration instance.
        path (str): Path to the noisy maintenance log CSV file.
        projection (list): List of columns to include in the output records.
        df (pd.DataFrame): DataFrame containing the loaded CSV data.
        max_iterations (int): Number of records to stream.
        records (list): List of projected records as dictionaries.
        index (int): Current position in the records list.

    Methods:
        __iter__(): Returns the iterator object itself.
        has_next(): Returns True if there are more records to stream, False otherwise.
        __next__(): Returns the next record as a pretty-printed JSON string. Raises StopIteration when all records have been streamed.
    """

    def __init__(self, fleet_config, max_iterations: int = -1):
        """
        Initialize the BatchToStream class.
        """
        self.config = ConfigSingleton.get_instance()
        fs = FileSystem(
            context=fleet_config
        )
        self.path = fs.noisy_table_file('maintenance_log')

        self.projection = [
            # "id",
            "work_order_number",
            "start_date",
            "end_date",
            "license_plate",
            "system",
            "subsystem",
            "component",
            "activity",
            "work_description",
            # "work_order_type",
            # "workshop_location",
        ]

        self.df = pd.read_csv(self.path)
        self.max_iterations = max_iterations if max_iterations > 0 else len(
            self.df)
        self.records = json.loads(
            self.df[self.projection].to_json(orient='records'))
        self.index = 0

    def __iter__(self):
        return self

    def has_next(self):
        return self.index < self.max_iterations

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        record = self.records[self.index]
        self.index += 1
        return json.dumps(record, indent=2)


if __name__ == "__main__":
    # Example usage

    fs = FileSystem(context={'fleet_id': 'fleet_1'})
    path = fs.noisy_table_file('maintenance_log')
    print(f"Reading data from: {path}")
    print('columns: ', pd.read_csv(path).columns.tolist())
    fleet_config = {'fleet_id': 'fleet_1'}

    streaming_adapter = BatchToStream(
        fleet_config=fleet_config,
        max_iterations=10
    )

    for json_record in streaming_adapter:
        print(json_record)
        break
