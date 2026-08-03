from pathlib import Path

from src.utils import PROJECT_ROOT, public_path


def test_public_path_renders_project_files_relatively():
    path = PROJECT_ROOT / "dataset" / "example.csv"
    assert public_path(path) == "dataset/example.csv"


def test_public_path_hides_external_directory_layout():
    path = Path("C:/external/private/artifact.csv")
    assert public_path(path) == "artifact.csv"


def test_public_path_preserves_model_uris_and_missing_values():
    assert public_path("runs:/abc/model") == "runs:/abc/model"
    assert public_path(None) is None


def test_public_path_normalizes_relative_separators():
    assert public_path(Path("dataset") / "artifact.json") == "dataset/artifact.json"
