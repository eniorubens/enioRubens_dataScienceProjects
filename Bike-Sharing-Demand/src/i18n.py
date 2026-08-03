"""Fábrica de internacionalização do projeto Bike Sharing Demand v4.

Português é o idioma **canônico** (fonte) do projeto: todos os textos
visíveis enviados ao ``LangMap`` são escritos em PT-BR. Os notebooks PT
usam ``make_lang("pt")``, que opera em modo *passthrough* — retorna os
textos originais sem nenhuma chamada de rede. No futuro, ``make_lang("en")``
traduzirá esses mesmos textos canônicos para inglês.

Esta camada NÃO reimplementa tradução: apenas embrulha o ``multilang.LangMap``
compartilhado com os defaults do projeto (idioma-base PT e cache versionado).

Instalar o pacote multilang (editável local):
    pip install deep-translator
    pip install -e ".../ds_toolkit/multilang"   # caminho do pacote compartilhado
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from multilang import LangMap  # type: ignore[import]

# Português é o idioma-fonte canônico. make_lang("pt") -> passthrough (sem rede).
BASE_LANG: str = "pt"
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / ".multilang_cache"


def make_lang(target: str) -> LangMap:
    """Retorna um ``LangMap`` configurado para este projeto.

    Parameters
    ----------
    target:
        Código ISO do idioma de saída (por exemplo, ``"pt"`` ou ``"en"``).
        Quando ``target == BASE_LANG`` (``"pt"``), o ``LangMap`` opera em
        modo *passthrough*: zero chamadas de API, retornando os textos
        canônicos em português sem modificação.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return LangMap(source=BASE_LANG, target=target, cache_dir=str(CACHE_DIR))


def resolve_lang(lang: LangMap | None) -> LangMap:
    """Resolve o ``lang`` recebido por uma função de gráfico ou relatório.

    Ponto único de resolução de idioma do projeto: se ``lang`` já for um
    ``LangMap``, ele é devolvido inalterado; caso contrário, o idioma-base
    (Português, *passthrough*, sem rede) é retornado. Substitui as antigas
    implementações duplicadas de ``_resolve_lang`` espalhadas pelos módulos,
    cujo fallback era inglês.
    """
    if lang is None:
        return make_lang(BASE_LANG)
    return lang


def localize_table(
    dataframe: pd.DataFrame,
    lang: LangMap,
    columns: Mapping[str, str],
    value_columns: Sequence[str] = (),
    value_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Retorna uma cópia do relatório com cabeçalhos e valores localizados.

    O schema interno de ``dataframe`` (os nomes de coluna reais devolvidos
    pelas funções de cálculo) nunca é alterado — apenas a cópia retornada
    por esta função recebe os rótulos canônicos em PT via ``lang``. Isso
    permite que testes e código consumidor continuem referenciando o schema
    estável enquanto a camada de apresentação exibe texto localizado.

    Parameters
    ----------
    dataframe:
        Relatório interno (schema estável) a ser localizado para exibição.
    lang:
        ``LangMap`` já resolvido (ver ``resolve_lang``).
    columns:
        Mapa nome-de-coluna-interno -> texto canônico em PT. Só as colunas
        presentes em ``dataframe`` são renomeadas; chaves extras são ignoradas.
    value_columns:
        Nomes de colunas (internos, antes da renomeação) cujos *valores*
        também devem ser localizados via ``value_labels``.
    value_labels:
        Mapa valor-interno -> texto canônico em PT, aplicado a cada coluna
        listada em ``value_columns``. Valores sem entrada no mapa são
        mantidos como estão.
    """
    display_df = dataframe.copy()
    if value_columns and value_labels:
        mapped_labels = lang(dict(value_labels))
        for column in value_columns:
            if column in display_df.columns:
                # Cast to object first: a categorical column's own .fillna() rejects
                # the mapped result once its categories differ from the original
                # (e.g. "Winter" vs "Inverno"), even though both are plain labels here.
                original = display_df[column].astype("object")
                display_df[column] = original.map(mapped_labels).fillna(original)
    return display_df.rename(columns=lang(dict(columns)))
