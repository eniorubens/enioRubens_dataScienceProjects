"""Funções de carregamento e diagnóstico exploratório do NASA C-MAPSS.

As funções não treinam modelos. Elas mantêm a leitura e os diagnósticos fora
dos notebooks para que as edições PT-BR e EN-US compartilhem a mesma lógica.
"""

from pathlib import Path
import pandas as pd

BASE_COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS


def load_split(data_dir: str | Path, subset: str, fd: str) -> pd.DataFrame:
    """Carrega um arquivo train/test FD001--FD004 sem descartar colunas."""
    path = Path(data_dir) / f"{subset}_{fd}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")


def structural_summary(data_dir: str | Path) -> pd.DataFrame:
    """Resume linhas, motores, ciclos, duplicatas e ausências por arquivo."""
    rows = []
    for subset in ("train", "test"):
        for i in range(1, 5):
            fd = f"FD{i:03d}"
            df = load_split(data_dir, subset, fd)
            per_unit = df.groupby("unit")["cycle"].agg(["min", "max", "count"])
            rows.append({
                "subset": subset,
                "dataset": fd,
                "rows": len(df),
                "columns": df.shape[1],
                "units": df["unit"].nunique(),
                "cycle_min": int(df["cycle"].min()),
                "cycle_max": int(df["cycle"].max()),
                "missing_values": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "cycles_monotonic_by_unit": all(
                    group["cycle"].is_monotonic_increasing
                    for _, group in df.groupby("unit")
                ),
                "terminal_cycle_median": float(per_unit["count"].median()),
            })
    return pd.DataFrame(rows)


def sensor_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula escala, cardinalidade e ausência para cada sensor/configuração."""
    features = ["setting1", "setting2", "setting3", *SENSOR_COLUMNS]
    result = pd.DataFrame({
        "feature": features,
        "missing": [int(df[c].isna().sum()) for c in features],
        "unique_values": [int(df[c].nunique(dropna=False)) for c in features],
        "minimum": [float(df[c].min()) for c in features],
        "maximum": [float(df[c].max()) for c in features],
        "mean": [float(df[c].mean()) for c in features],
        "std": [float(df[c].std()) for c in features],
    })
    return result.sort_values(["unique_values", "feature"]).reset_index(drop=True)


def feature_columns(include_settings: bool = True) -> list[str]:
    """Retorna as colunas numéricas usadas nos diagnósticos dos sinais."""
    return (["setting1", "setting2", "setting3"] if include_settings else []) + SENSOR_COLUMNS


def correlation_matrix(df: pd.DataFrame, include_settings: bool = True) -> pd.DataFrame:
    """Calcula a correlação de Pearson entre configurações e sensores."""
    return df[feature_columns(include_settings)].corr(method="pearson")


def sensor_variability_by_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Resume a variabilidade de cada sensor entre motores."""
    rows = []
    for feature in SENSOR_COLUMNS:
        per_unit_std = df.groupby("unit")[feature].std()
        rows.append({
            "feature": feature,
            "median_within_unit_std": float(per_unit_std.median()),
            "mean_within_unit_std": float(per_unit_std.mean()),
            "between_unit_std": float(df.groupby("unit")[feature].mean().std()),
        })
    return pd.DataFrame(rows).sort_values("between_unit_std", ascending=False).reset_index(drop=True)


def operating_settings_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Resume a distribuição das três condições operacionais."""
    settings = ["setting1", "setting2", "setting3"]
    return pd.DataFrame({
        "feature": settings,
        "unique_values": [int(df[c].nunique()) for c in settings],
        "minimum": [float(df[c].min()) for c in settings],
        "maximum": [float(df[c].max()) for c in settings],
        "mean": [float(df[c].mean()) for c in settings],
        "std": [float(df[c].std()) for c in settings],
    })
