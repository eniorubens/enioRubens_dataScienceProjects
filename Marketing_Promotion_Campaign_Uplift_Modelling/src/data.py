"""Carregamento do dataset Hillstrom, dicionário de variáveis e tratamento pooled."""
import pandas as pd

from .config import (
    ARMS,
    CONTROL_ARM,
    DATA_PATH,
    POOLED_TREATMENT_COL,
    TREATMENT_COL,
)


def load_hillstrom(path=DATA_PATH):
    """Carrega o CSV Hillstrom e tipa a coluna de tratamento como categórica.

    A ordem das categorias segue ARMS, para que groupby/plots mantenham
    sempre a mesma ordem (No E-Mail, Mens E-Mail, Womens E-Mail).
    """
    df = pd.read_csv(path)
    df[TREATMENT_COL] = pd.Categorical(df[TREATMENT_COL], categories=ARMS, ordered=False)
    return df


def add_pooled_treatment(df):
    """Adiciona a coluna de tratamento pooled (1 = qualquer e-mail, 0 = No E-Mail).

    Não modifica o dataframe recebido; retorna uma cópia.
    """
    out = df.copy()
    out[POOLED_TREATMENT_COL] = (out[TREATMENT_COL] != CONTROL_ARM).astype(int)
    return out


def variable_dictionary():
    """Retorna o dicionário de variáveis do dataset como DataFrame.

    Espelha a tabela em markdown de S1.5 do notebook, para uso programático
    (ex.: montagem automática de tabelas em reports.py).
    """
    rows = [
        ("recency", "covariável", "int", "Meses desde a última compra (1-12)"),
        ("history", "covariável", "float", "Valor gasto (USD) nos últimos 12 meses"),
        ("history_segment", "covariável", "categórica", "Faixa de gasto histórico (7 níveis)"),
        ("mens", "covariável", "binária", "Comprou produto masculino no último ano"),
        ("womens", "covariável", "binária", "Comprou produto feminino no último ano"),
        ("zip_code", "covariável", "categórica", "Rural / Suburban / Urban"),
        ("newbie", "covariável", "binária", "Cliente novo (<=12 meses)"),
        ("channel", "covariável", "categórica", "Phone / Web / Multichannel"),
        ("segment", "tratamento", "categórica", "No E-Mail / Mens E-Mail / Womens E-Mail"),
        ("visit", "outcome", "binária", "Visitou o site (outcome primário)"),
        ("conversion", "outcome", "binária", "Comprou (outcome secundário, raro)"),
        ("spend", "outcome", "contínua", "Valor gasto (outcome secundário, massa em zero)"),
    ]
    return pd.DataFrame(rows, columns=["variavel", "papel", "tipo", "descricao"])
