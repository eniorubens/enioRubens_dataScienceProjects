"""Project paths and bounded-audit configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve project-relative paths without coupling code to a working directory.

    Args:
        root: Absolute or relative path to the project root.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        """Create a path registry from a project root.

        Args:
            root: Path to the project directory.

        Returns:
            A normalized ``ProjectPaths`` instance.

        Raises:
            FileNotFoundError: If ``root`` does not exist.
            NotADirectoryError: If ``root`` is not a directory.
        """

        path = Path(root).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Project root does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {path}")
        return cls(root=path)

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory."""

        return self.root / "dataset"

    @property
    def parquet_path(self) -> Path:
        """Return the canonical processed Parquet path."""

        return self.dataset_dir / "processed" / "complaints.parquet"

    @property
    def temp_dir(self) -> Path:
        """Return the project temporary directory."""

        return self.root / "temp"

    @property
    def src_dir(self) -> Path:
        """Return the source-package directory."""

        return self.root / "src"

    def require_parquet(self) -> Path:
        """Validate and return the canonical Parquet path.

        Returns:
            Existing Parquet dataset path.

        Raises:
            FileNotFoundError: If the Parquet dataset is absent.
            IsADirectoryError: If the path points to a directory.
        """

        path = self.parquet_path
        if not path.exists():
            raise FileNotFoundError(f"Parquet dataset does not exist: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Parquet path is a directory: {path}")
        return path


@dataclass(frozen=True)
class S0AuditConfig:
    """Set limits and columns for sample and full-corpus S0 audits.

    Args:
        sample_rows: Maximum number of Parquet rows read by the sample audit.
        top_k: Number of labels or duplicate groups retained in sample output.
        text_column: Narrative column name.
        date_column: Complaint date column name.
        product_column: Product label column name.
        issue_column: Issue label column name.
        duckdb_memory_limit: Per-query DuckDB memory limit.
        duckdb_threads: Number of DuckDB worker threads per query.
    """

    sample_rows: int = 100_000
    top_k: int = 20
    text_column: str = "Consumer complaint narrative"
    date_column: str = "Date received"
    product_column: str = "Product"
    issue_column: str = "Issue"
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 2

    def __post_init__(self) -> None:
        """Reject unsafe or unusable audit limits."""

        if self.sample_rows <= 0:
            raise ValueError("sample_rows must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.duckdb_threads <= 0:
            raise ValueError("duckdb_threads must be positive")
