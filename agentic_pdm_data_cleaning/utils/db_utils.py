import json
import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine

from agentic_pdm_data_cleaning.utils.filesystem import FileSystem

try:
    from sqlalchemy import Engine
    from sqlalchemy.inspection import inspect
    from sqlalchemy.orm import Session
    from sqlalchemy.sql.expression import text
except ImportError:
    raise ImportError(
        "SQLAlchemy is required for this module. Please install it using 'pip install sqlalchemy'.")


def generate_db(benchmark_id, fleet_config={}) -> None:
    fs_context = fleet_config.get("global_params", {})
    fs = FileSystem(context=fs_context)

    # Delete the database file if it exists
    db_file = fs.get_fleet_db_file()
    if os.path.exists(db_file):
        os.remove(db_file)

    url = f'duckdb:///{db_file}'
    # Create SQLAlchemy engine
    engine = create_engine(url)

    # Load and insert fleet table
    fleet = pd.read_csv(fs.noisy_table_file("fleet_registry"))
    fleet[['device_id', 'name', 'license_plate', 'VIN', 'active_from', 'active_to']]\
        .to_sql('fleet_registry', con=engine, if_exists='replace', index=False)

    # load service_catalog file
    service_catalog_file = fleet_config.get("maintenance_log", {}).get("service_catalog_path", "resources/service_catalog.csv")
    service_catalog_df = pd.read_csv(service_catalog_file)
    service_catalog_df.to_sql('service_catalog', con=engine,
                              if_exists='replace', index=False)

    # Directory containing odometer signal CSVs
    for signal_type in os.listdir(fs.signals_dir):

        signal_dir = fs.get_signal_dir(signal_type)
        # Process each odometer signal CSV
        for filename in os.listdir(signal_dir):
            if filename.endswith('.csv'):
                filepath = signal_dir / filename

                # Load CSV into DataFrame
                df = pd.read_csv(filepath)

                # Convert date column to datetime.date
                df['date'] = pd.to_datetime(df['date']).dt.date

                # Insert into odometer_signals table
                df.to_sql(f"signal_{signal_type}", con=engine,
                          if_exists='append', index=False)

        print(f"✅ {signal_type.title()} data loaded successfully.")

    engine.dispose()


def list_tables(db_engine: Engine) -> list[str]:
    """
    Use this function to get a list of table names in the database.

    :param db_engine: SQLAlchemy Engine object.
    :return: List of table names.
    """
    inspector = inspect(db_engine)
    return json.dumps(inspector.get_table_names())


def describe_table(db_engine: Engine, table_name: str) -> str:
    try:
        inspector = inspect(db_engine)
        table_schema = inspector.get_columns(table_name)
        return json.dumps([str(column) for column in table_schema])
    except Exception as e:
        return f'Error getting table schema for table "{table_name}": {e}'


def run_sql_query(db_engine: Engine, query: str, limit: Optional[int] = 10) -> str:
    """
    Use this function to run a SQL query on the database.

    Args:
        db_engine (Engine): The SQLAlchemy engine to use.
        query (str): The SQL query to run.
        limit (Optional[int]): The maximum number of rows to return.

    Returns:
        str: The result of the query.
    """
    with Session(db_engine) as session, session.begin():

        try:
            result = session.execute(text(query))
            if limit:
                rows = result.fetchmany(limit)
            else:
                rows = result.fetchall()

            recordset = [row._asdict() for row in rows]
            return json.dumps(recordset, default=str)
        except Exception as e:
            return str(e)


if __name__ == "__main__":
    context = {'global_params': {'fleet_id': 'fleet_mixt'}}
    generate_db(context)
    fs = FileSystem(context=context)
    url: str = f'sqlite:///{fs.get_fleet_db_file()}'
    db_engine = create_engine(url)
    print("Tables:", list_tables(db_engine))

    query = 'SELECT * FROM fleet_registry'
    print("Run SQL query 'SELECT * FROM fleet_registry LIMIT 5':",
          run_sql_query(db_engine, query, limit=5))
    query = 'SELECT * FROM odometer'
    print("Run SQL query 'SELECT * FROM odometer LIMIT 5':",
          run_sql_query(db_engine, query, limit=5))
