"""Features derivadas para a iteração de S4.5 — extensão opt-in, isolada.

Nunca toca `FEATURE_COLS`/`CAT_VARS`/`CONT_VARS`/`BIN_VARS` globais de
`config.py`: S3 e os quatro meta-learners originais de S4.1-4.4 continuam
produzindo exatamente os mesmos números se re-executados. Quem quiser as
features derivadas passa `EXTENDED_*_VARS` explicitamente para
`build_meta_learner_encoder`/`encode_meta_learner_features`.
"""
from .config import BIN_VARS, CAT_VARS, CONT_VARS


def add_engineered_features(df):
    """Retorna cópia de `df` com as 4 features derivadas de S4.5 adicionadas.

    1. `history_per_recency` — razão genuína (não reconstruível por splits
       nas duas variáveis isoladas): distingue cliente de alto valor
       "adormecido" (history alto, recency alto) de um ativo (history alto,
       recency baixo).
    2. `newbie_x_channel` — canal de aquisição pode proxy engajamento/confiança
       digital de forma diferente para clientes novos vs. estabelecidos.
    3. `mens_and_womens` — clientes que compraram nas duas categorias podem
       ser um perfil distinto (compradores para terceiros/famílias).
    4. `zip_code_x_channel` — combinação pode proxy um segmento de engajamento
       digital que nenhuma das duas variáveis captura sozinha.
    """
    out = df.copy()
    out["history_per_recency"] = out["history"] / (out["recency"] + 1)
    out["newbie_x_channel"] = (
        out["newbie"].astype(str) + "_" + out["channel"].astype(str)
    ).astype("category")
    out["mens_and_womens"] = (out["mens"].astype(bool) & out["womens"].astype(bool)).astype(int)
    out["zip_code_x_channel"] = (
        out["zip_code"].astype(str) + "_" + out["channel"].astype(str)
    ).astype("category")
    return out


EXTENDED_CONT_VARS = CONT_VARS + ["history_per_recency"]
EXTENDED_CAT_VARS = CAT_VARS + ["newbie_x_channel", "zip_code_x_channel"]
EXTENDED_BIN_VARS = BIN_VARS + ["mens_and_womens"]
