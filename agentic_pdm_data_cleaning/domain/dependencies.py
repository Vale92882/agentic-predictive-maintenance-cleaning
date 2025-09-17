from dataclasses import dataclass

from sqlalchemy import Engine


@dataclass
class Dependencies:
    db_engine_ro: Engine
    # query_engine: BaseQueryEngine  # Replace with actual type if known
