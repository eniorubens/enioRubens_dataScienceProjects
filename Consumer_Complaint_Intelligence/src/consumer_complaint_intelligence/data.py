"""Bounded Parquet and DuckDB access for the project."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import polars as pl


def read_parquet_sample(
    path: str | Path,
    limit: int,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read at most ``limit`` rows using a lazy Polars scan.

    Args:
        path: Parquet file to scan.
        limit: Positive maximum number of rows to collect.
        columns: Optional projection of columns to read.

    Returns:
        A bounded Polars DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``limit`` is not positive.
    """

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    if limit <= 0:
        raise ValueError("limit must be positive")

    frame = pl.scan_parquet(parquet_path)
    if columns is not None:
        frame = frame.select(list(columns))
    return frame.head(limit).collect()


def query_parquet(
    path: str | Path,
    query: str,
    temp_directory: str | Path | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> list[dict[str, Any]]:
    """Execute a caller-owned DuckDB query against one Parquet file.

    This helper is intentionally small: SQL remains at the audit or notebook
    boundary, while connection limits are centralized and explicit.

    Args:
        path: Parquet file referenced by the query.
        query: SQL statement containing one ``?`` placeholder for the Parquet
            path, for example ``read_parquet(?)``.
        temp_directory: Optional spill directory for DuckDB.
        memory_limit: DuckDB memory limit for this connection.
        threads: Number of DuckDB worker threads.

    Returns:
        Query rows as JSON-friendly dictionaries.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``threads`` is not positive.
        duckdb.Error: If DuckDB rejects the SQL statement.
    """

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    if threads <= 0:
        raise ValueError("threads must be positive")

    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET preserve_insertion_order = false")
        if temp_directory is not None:
            spill_path = Path(temp_directory)
            spill_path.mkdir(parents=True, exist_ok=True)
            connection.execute(
                f"SET temp_directory = '{spill_path.as_posix()}'"
            )
        result = connection.execute(query, [str(parquet_path)]).fetchall()
        names = [item[0] for item in connection.description]
    return [dict(zip(names, row)) for row in result]


def query_parquet_batch(
    path: str | Path,
    setup_query: str,
    queries: Mapping[str, str],
    temp_directory: str | Path | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Run setup and aggregate queries in one bounded DuckDB session.

    The setup query receives one bound Parquet path and may materialize a
    narrow temporary table. Subsequent queries run against that table without
    rereading the source file.

    Args:
        path: Parquet file referenced by the setup query.
        setup_query: Statement containing one ``?`` path placeholder.
        queries: Named aggregate statements executed after setup.
        temp_directory: Optional spill directory for DuckDB.
        memory_limit: DuckDB memory limit for this connection.
        threads: Number of DuckDB worker threads.

    Returns:
        Mapping of query names to JSON-friendly result rows.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``threads`` is not positive.
        duckdb.Error: If setup or an aggregate query fails.
    """

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    if threads <= 0:
        raise ValueError("threads must be positive")

    results: dict[str, list[dict[str, Any]]] = {}
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET preserve_insertion_order = false")
        if temp_directory is not None:
            spill_path = Path(temp_directory)
            spill_path.mkdir(parents=True, exist_ok=True)
            connection.execute(
                f"SET temp_directory = '{spill_path.as_posix()}'"
            )
        connection.execute(setup_query, [str(parquet_path)])
        for name, query in queries.items():
            result = connection.execute(query)
            names = [item[0] for item in connection.description]
            results[name] = [dict(zip(names, row)) for row in result.fetchall()]
    return results
