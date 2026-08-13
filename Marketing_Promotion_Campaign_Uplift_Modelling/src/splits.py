"""Split estratificado (braço x visit) em treino / validação / teste selado.

O teste selado é a peça de disciplina metodológica central do projeto: ele só
pode ser aberto uma vez, em S6, para avaliação **confirmatória** — depois que
a configuração final já tiver sido selecionada e congelada ao término de
S4+S5, usando exclusivamente os dados de desenvolvimento (`train_df`/
`val_df`). O teste selado não participa da escolha do modelo. Por isso
`make_splits` não retorna os índices de teste diretamente — eles são gravados
em disco, e só `load_sealed_test` os expõe, e somente sob `unlock=True`
explícito.

`persist_test=True` (default de `make_splits`) grava os três manifests de
índice (treino/validação/teste) mais um fingerprint SHA-256 do CSV fonte, em
`dataset_manifest.json`. Isso existe para fechar uma fragilidade: antes,
treino/validação eram recalculados via `train_test_split` a cada execução de
notebook (determinístico só enquanto `SEED`, a lógica de estratificação e o
próprio CSV nunca mudarem — nada garantia isso). Agora `get_train_val` e
`load_sealed_test` carregam os manifests persistidos quando existem, e falham
alto (`ValueError`) se o fingerprint/`n_rows` do dataset atual não baterem com
o gravado, ou se o teste selado já existir mas os manifests de treino/
validação estiverem ausentes ou inconsistentes — em vez de silenciosamente
recalcular ou aplicar partições que podem não corresponder ao mesmo dataset.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, SEED, SPLITS_DIR, TEST_FRAC, TRAIN_FRAC, TREATMENT_COL, VAL_FRAC

TEST_INDEX_PATH = SPLITS_DIR / "sealed_test_index.parquet"
TRAIN_INDEX_PATH = SPLITS_DIR / "train_index.parquet"
VAL_INDEX_PATH = SPLITS_DIR / "validation_index.parquet"
MANIFEST_PATH = SPLITS_DIR / "dataset_manifest.json"


def _strata_key(df):
    """Chave de estratificação conjunta braço x visit."""
    return df[TREATMENT_COL].astype(str) + "__visit_" + df["visit"].astype(str)


def dataset_fingerprint(path=None):
    """SHA-256 dos bytes do CSV fonte (`DATA_PATH` por padrão).

    Usado para detectar se os manifests de índice persistidos ainda
    correspondem ao dataset em disco — se o CSV mudar (nova versão, linhas
    adicionadas/removidas), os `row_index` salvos podem passar a apontar para
    linhas diferentes das originais, silenciosamente. `get_train_val` compara
    este fingerprint contra o gravado em `dataset_manifest.json` e falha alto
    em caso de divergência, em vez de carregar uma partição inconsistente.
    """
    target = path if path is not None else DATA_PATH
    sha256 = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _write_index(idx, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": idx}).to_parquet(path, index=False)


def _read_index(path):
    return pd.read_parquet(path)["row_index"].values


def _validate_split_artifacts(splits_dir, df, idx=None):
    """Valida que os artefatos de split persistidos em `splits_dir` ainda
    correspondem a `df`: fingerprint SHA-256 do CSV fonte, `n_rows`, e (se
    `idx` for fornecido) que os índices persistidos são um subconjunto do
    índice de `df`. Levanta `ValueError` claro em qualquer divergência.
    Usada tanto por `get_train_val` quanto por `load_sealed_test`, para que a
    mesma checagem proteja os dois pontos de entrada e não fique duplicada.

    Retorna o manifest carregado (dict).
    """
    manifest_path = splits_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    current_fp = dataset_fingerprint()
    if manifest["dataset_sha256"] != current_fp:
        raise ValueError(
            f"Fingerprint do dataset não bate com {manifest_path} "
            f"(manifest={manifest['dataset_sha256']}, atual={current_fp}) — "
            "o CSV fonte mudou desde que os splits foram persistidos. Rode "
            "make_splits(df_pooled, persist_test=True) novamente em "
            "02_Baseline_Propensity_PT para regerar os manifests."
        )
    if manifest["n_rows"] != len(df):
        raise ValueError(
            f"n_rows do dataset não bate com {manifest_path} "
            f"(manifest={manifest['n_rows']}, atual={len(df)}) — o DataFrame "
            "recebido não corresponde ao dataset que originou os splits "
            "persistidos."
        )
    if idx is not None and not pd.Index(idx).isin(df.index).all():
        raise ValueError(
            f"Índices persistidos em {splits_dir} não são um subconjunto do "
            "índice do DataFrame recebido — os artefatos de split não "
            "correspondem a este dataset, mesmo com fingerprint/n_rows "
            "batendo (ex.: índice do DataFrame foi resetado ou reordenado)."
        )
    return manifest


def make_splits(df, seed=SEED, persist_test=True, splits_dir=None):
    """Particiona `df` em treino (60%), validação (20%) e teste selado (20%).

    Estratifica por (braço x visit) conjuntamente, preservando a proporção de
    cada braço de tratamento e a prevalência de `visit` em cada partição.

    Quando `persist_test=True` (default), grava os três manifests de índice —
    `train_index.parquet`, `validation_index.parquet`,
    `sealed_test_index.parquet` — mais `dataset_manifest.json` (fingerprint
    SHA-256 do CSV fonte + seed + tamanhos), em `splits_dir` (ou `SPLITS_DIR`
    por padrão). O teste selado nunca é retornado por esta função — use
    `load_sealed_test(df, unlock=True)` para acessá-lo, e apenas em S6.

    Retorna
    -------
    dict com chaves 'train_idx' e 'val_idx' (arrays de índices de `df`).
    """
    strata = _strata_key(df)

    train_idx, holdout_idx = train_test_split(
        df.index.values,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=strata,
        random_state=seed,
    )

    holdout_strata = strata.loc[holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=holdout_strata,
        random_state=seed,
    )

    if persist_test:
        target_dir = splits_dir if splits_dir is not None else SPLITS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        _write_index(train_idx, target_dir / "train_index.parquet")
        _write_index(val_idx, target_dir / "validation_index.parquet")
        _write_index(test_idx, target_dir / "sealed_test_index.parquet")
        manifest = {
            "dataset_sha256": dataset_fingerprint(),
            "seed": seed,
            "n_rows": len(df),
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        }
        (target_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {"train_idx": train_idx, "val_idx": val_idx}


def get_train_val(df_pooled, seed=SEED, persist_test=False, splits_dir=None):
    """Treino/validação a partir de `df_pooled`, usado pelos notebooks 03+.

    Se `train_index.parquet`/`validation_index.parquet`/`dataset_manifest.json`
    já existirem em `splits_dir` (ou `SPLITS_DIR` por padrão — gravados por
    `make_splits(..., persist_test=True)`, chamado em
    `02_Baseline_Propensity_PT`), **carrega** os índices persistidos em vez de
    recalcular via `train_test_split`, e levanta `ValueError` se o
    fingerprint/`n_rows` do dataset atual não baterem com o manifest (dataset
    mudou desde que os splits foram selados).

    Se nenhum artefato existir ainda (bootstrap antes de rodar o notebook 02,
    ou `splits_dir` isolado de teste), recalcula via `make_splits` como antes
    — `persist_test=False` por padrão significa que este recálculo não grava
    nada; só `02_Baseline_Propensity_PT` deve persistir.

    Mas se `sealed_test_index.parquet` já existir e os manifests de treino/
    validação estiverem ausentes ou incompletos, **não cai no fallback de
    recálculo**: levanta `ValueError` exigindo intervenção explícita. Uma vez
    que um teste selado existe, nunca geramos automaticamente um novo treino/
    validação que poderia não corresponder a esse mesmo teste.
    """
    target_dir = splits_dir if splits_dir is not None else SPLITS_DIR
    train_path = target_dir / "train_index.parquet"
    val_path = target_dir / "validation_index.parquet"
    test_path = target_dir / "sealed_test_index.parquet"
    manifest_path = target_dir / "dataset_manifest.json"

    manifest_complete = train_path.exists() and val_path.exists() and manifest_path.exists()

    if manifest_complete:
        train_idx = _read_index(train_path)
        val_idx = _read_index(val_path)
        _validate_split_artifacts(target_dir, df_pooled, idx=np.concatenate([train_idx, val_idx]))
    elif test_path.exists():
        raise ValueError(
            f"Estado incompleto dos artefatos de split em {target_dir}: "
            f"{test_path.name} já existe, mas train_index.parquet, "
            "validation_index.parquet e/ou dataset_manifest.json estão "
            "ausentes ou inconsistentes. Uma vez que um teste selado existe, "
            "get_train_val nunca recalcula silenciosamente um novo treino/"
            "validação que poderia não corresponder a esse teste — corrija "
            "ou regenere os artefatos manualmente antes de continuar."
        )
    else:
        split_idx = make_splits(df_pooled, seed=seed, persist_test=persist_test, splits_dir=splits_dir)
        train_idx, val_idx = split_idx["train_idx"], split_idx["val_idx"]

    train_df = df_pooled.loc[train_idx].copy()
    val_df = df_pooled.loc[val_idx].copy()
    return train_df, val_df


def load_sealed_test(df, unlock=False, splits_dir=None):
    """Carrega o teste selado gravado por `make_splits`.

    Levanta `PermissionError` se `unlock` não for `True` — barreira deliberada
    contra abertura acidental do teste antes de S6. Depois de destravado,
    valida os artefatos persistidos contra `df` (mesma checagem usada por
    `get_train_val` — fingerprint SHA-256, `n_rows`, e que os índices do
    teste são um subconjunto do índice de `df`) antes de aplicar
    `sealed_test_index.parquet` — protege contra aplicar silenciosamente
    índices antigos sobre um dataset diferente do que os originou.
    """
    if not unlock:
        raise PermissionError(
            "O teste selado só pode ser aberto com unlock=True, e isso só deve "
            "acontecer em S6, uma única vez, para avaliação confirmatória — "
            "depois que a configuração final já tiver sido selecionada e "
            "congelada ao término de S4+S5, usando exclusivamente os dados de "
            "desenvolvimento. O teste selado não participa da escolha do "
            "modelo."
        )
    target_dir = splits_dir if splits_dir is not None else SPLITS_DIR
    test_path = target_dir / "sealed_test_index.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} não existe — rode make_splits(df) primeiro.")
    test_idx = _read_index(test_path)
    _validate_split_artifacts(target_dir, df, idx=test_idx)
    return df.loc[test_idx]
