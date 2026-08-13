"""Constantes de negócio, seed, paths e nomes de coluna do projeto Uplift Hillstrom."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
DATA_PATH = DATA_DIR / "Kevin_Hillstrom_MineThatData.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
META_LEARNERS_PATH = ARTIFACTS_DIR / "meta_learners.joblib"

# Tratamento observado (3 braços) e tratamento pooled usado em S3-S6
TREATMENT_COL = "segment"
POOLED_TREATMENT_COL = "treatment"

ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
TREATED_ARMS = ["Mens E-Mail", "Womens E-Mail"]
CONTROL_ARM = "No E-Mail"

OUTCOMES = ["visit", "conversion", "spend"]
PRIMARY_OUTCOME = "visit"
SECONDARY_OUTCOMES = ["conversion", "spend"]

CONT_VARS = ["recency", "history"]
BIN_VARS = ["mens", "womens", "newbie"]
CAT_VARS = ["zip_code", "channel", "history_segment"]
FEATURE_COLS = CONT_VARS + BIN_VARS + CAT_VARS

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

# Padrão herdado do Customer-Churn-Prediction-v3 / Bike-Sharing-Demand_v4:
# quando False, funções de src/ carregam artefato serializado em vez de retreinar.
RETRAIN = False
