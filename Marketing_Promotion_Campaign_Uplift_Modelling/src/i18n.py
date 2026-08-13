"""Fábrica de internacionalização do projeto Uplift Modeling (Hillstrom).

Português é o idioma **canônico** (fonte) do projeto: todo texto visível
enviado ao ``LangMap`` é escrito em PT-BR. O notebook PT usa
``make_lang("pt")``, que opera em modo *passthrough* — retorna os textos
originais sem nenhuma chamada de rede. ``make_lang("en")`` (quando a edição
EN for de fato criada) consultará o catálogo PT->EN revisado e versionado em
``src/i18n_catalogs``, para que a edição publicada também execute sem
depender de serviços externos.

Esta camada NÃO reimplementa tradução: apenas embrulha o ``multilang.LangMap``
compartilhado do portfólio com os defaults deste projeto (idioma-base PT,
catálogo versionado quando existir, cache de desenvolvimento caso contrário).

Regra de uso: só texto de UI (título, label, prefixo de print) passa por
``lang(...)``. Valores de dado (``segment``, ``zip_code``, ``channel`` — já
em inglês no dataset original) nunca devem ser traduzidos.
"""
from __future__ import annotations

from pathlib import Path

from multilang import LangMap  # type: ignore[import]

from .config import PROJECT_ROOT

# Português é o idioma-fonte canônico. make_lang("pt") -> passthrough (sem rede).
BASE_LANG: str = "pt"
CACHE_DIR: Path = PROJECT_ROOT / ".multilang_cache"
CATALOG_DIR: Path = PROJECT_ROOT / "src" / "i18n_catalogs"


def make_lang(target: str) -> LangMap:
    """Retorna um ``LangMap`` configurado para este projeto.

    Quando ``target == BASE_LANG`` (``"pt"``), o ``LangMap`` opera em modo
    *passthrough*: zero chamadas de API, retornando os textos canônicos em
    português sem modificação. Quando existir um catálogo revisado em
    ``CATALOG_DIR`` para o par (``BASE_LANG``, ``target``), ele é usado em
    modo ``offline=True`` — a edição publicada não depende de rede.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    catalog_path = CATALOG_DIR / f"{BASE_LANG}_{target}.json"
    if catalog_path.exists():
        return LangMap(
            source=BASE_LANG,
            target=target,
            cache_dir=str(CATALOG_DIR),
            offline=True,
        )
    return LangMap(source=BASE_LANG, target=target, cache_dir=str(CACHE_DIR))


def resolve_lang(lang: LangMap | None) -> LangMap:
    """Resolve o ``lang`` recebido por uma função de gráfico ou relatório.

    Ponto único de resolução de idioma do projeto: se ``lang`` já for um
    ``LangMap``, é devolvido inalterado; caso contrário, o idioma-base
    (português, passthrough, sem rede) é retornado como fallback.
    """
    if lang is None:
        return make_lang(BASE_LANG)
    return lang
