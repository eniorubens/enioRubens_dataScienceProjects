"""Dicionario de colunas e rotulos bilingues (PT-BR e EN-US).

Fonte do dicionario: referencia CASdatasets (``freMTPL``); ver o relatorio
da etapa 1. A coluna ``availability`` e a avaliacao desta auditoria sobre a
disponibilidade da variavel na data em que a taxa seria estimada.
"""

from __future__ import annotations

LANGUAGES = ("pt_br", "en_us")

HYPOTHESIS_PT = (
    "hipotese: disponivel no cenario; a fonte descreve o atributo e "
    "omite a data de referencia"
)
HYPOTHESIS_EN = (
    "hypothesis: available in the scenario; the source describes the "
    "attribute and omits the reference date"
)

# Para cada coluna: papel, unidade/categoria e disponibilidade.
COLUMN_DICTIONARY: dict[str, dict[str, dict[str, str]]] = {
    "IDpol": {
        "pt_br": {
            "role": "identificador do registro de apolice",
            "unit": "inteiro (rastreabilidade)",
            "availability": "disponivel; nao e preditor",
        },
        "en_us": {
            "role": "policy record identifier",
            "unit": "integer (traceability)",
            "availability": "available; not a predictor",
        },
    },
    "ClaimNb": {
        "pt_br": {
            "role": "alvo: numero de sinistros no periodo de exposicao",
            "unit": "contagem inteira >= 0",
            "availability": "desfecho; indisponivel na estimacao da taxa",
        },
        "en_us": {
            "role": "target: number of claims during exposure",
            "unit": "integer count >= 0",
            "availability": "outcome; unavailable at rate estimation",
        },
    },
    "Exposure": {
        "pt_br": {
            "role": "exposicao (offset log) e dado do cenario de previsao",
            "unit": "anos de cobertura",
            "availability": "disponivel como duracao do cenario",
        },
        "en_us": {
            "role": "exposure (log offset); scenario input",
            "unit": "years of coverage",
            "availability": "available as scenario duration",
        },
    },
    "Area": {
        "pt_br": {
            "role": "covariavel territorial (densidade ordinal)",
            "unit": "categoria A (rural) a F (centro urbano)",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "territorial covariate (ordinal density)",
            "unit": "category A (rural) to F (urban centre)",
            "availability": HYPOTHESIS_EN,
        },
    },
    "VehPower": {
        "pt_br": {
            "role": "covariavel do veiculo",
            "unit": "valores ordenados (sem unidade publicada)",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "vehicle covariate",
            "unit": "ordered values (no published unit)",
            "availability": HYPOTHESIS_EN,
        },
    },
    "VehAge": {
        "pt_br": {
            "role": "covariavel do veiculo",
            "unit": "anos",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "vehicle covariate",
            "unit": "years",
            "availability": HYPOTHESIS_EN,
        },
    },
    "DrivAge": {
        "pt_br": {
            "role": "covariavel do condutor",
            "unit": "anos",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "driver covariate",
            "unit": "years",
            "availability": HYPOTHESIS_EN,
        },
    },
    "BonusMalus": {
        "pt_br": {
            "role": "covariavel do condutor (historico de sinistros)",
            "unit": "indice 50-350; <100 bonus, >100 malus",
            "availability": "INDETERMINADA: instante de medicao nao "
                            "documentado",
        },
        "en_us": {
            "role": "driver covariate (claims history)",
            "unit": "index 50-350; <100 bonus, >100 malus",
            "availability": "INDETERMINATE: measurement time not "
                            "documented",
        },
    },
    "VehBrand": {
        "pt_br": {
            "role": "covariavel do veiculo",
            "unit": "11 categorias B1-B6, B10-B14 (significado "
                    "nao publicado)",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "vehicle covariate",
            "unit": "11 categories B1-B6, B10-B14 (meaning not "
                    "published)",
            "availability": HYPOTHESIS_EN,
        },
    },
    "VehGas": {
        "pt_br": {
            "role": "covariavel do veiculo",
            "unit": "categoria Diesel ou Regular",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "vehicle covariate",
            "unit": "category Diesel or Regular",
            "availability": HYPOTHESIS_EN,
        },
    },
    "Density": {
        "pt_br": {
            "role": "covariavel territorial",
            "unit": "habitantes por km2 da cidade do condutor",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "territorial covariate",
            "unit": "inhabitants per km2 of the driver's city",
            "availability": HYPOTHESIS_EN,
        },
    },
    "Region": {
        "pt_br": {
            "role": "covariavel territorial",
            "unit": "regiao francesa (classificacao 1970-2015)",
            "availability": HYPOTHESIS_PT,
        },
        "en_us": {
            "role": "territorial covariate",
            "unit": "French region (1970-2015 classification)",
            "availability": HYPOTHESIS_EN,
        },
    },
}

# Explicacoes de nomes abreviados para exibicao introdutoria. Mantidas fora de
# COLUMN_DICTIONARY para preservar o contrato de saida de describe_column().
COLUMN_MEANINGS: dict[str, dict[str, str]] = {
    "VehPower": {
        "pt_br": "potência do veículo",
        "en_us": "vehicle power",
    },
    "VehAge": {
        "pt_br": "idade do veículo",
        "en_us": "vehicle age",
    },
    "DrivAge": {
        "pt_br": "idade do condutor",
        "en_us": "driver age",
    },
    "VehBrand": {
        "pt_br": "grupo anonimizado da marca do veículo",
        "en_us": "anonymized vehicle-brand group",
    },
    "VehGas": {
        "pt_br": "tipo de combustível do veículo",
        "en_us": "vehicle fuel type",
    },
}


def describe_column_meaning(name: str, language: str = "pt_br") -> str:
    """Devolva o significado introdutorio ou, na ausencia, o papel.

    Args:
        name: nome da coluna do ``freMTPL2freq``.
        language: ``pt_br`` ou ``en_us``.

    Returns:
        Significado explicito para nomes abreviados; para as demais colunas,
        o papel definido no dicionario.

    Raises:
        ValueError: se o idioma nao for suportado.
    """
    description = describe_column(name, language)
    return COLUMN_MEANINGS.get(name, {}).get(language, description["role"])


def describe_column(name: str, language: str = "pt_br") -> dict[str, str]:
    """Devolve papel, unidade e disponibilidade de uma coluna.

    Args:
        name: nome da coluna do ``freMTPL2freq``.
        language: ``pt_br`` ou ``en_us``.

    Returns:
        Dicionario com ``role``, ``unit`` e ``availability``; valores
        ``n/d`` se a coluna nao constar do dicionario.

    Raises:
        ValueError: se o idioma nao for suportado.
    """
    if language not in LANGUAGES:
        raise ValueError(f"idioma nao suportado: {language}")
    entry = COLUMN_DICTIONARY.get(name)
    if entry is None:
        return {"role": "n/d", "unit": "n/d", "availability": "n/d"}
    return dict(entry[language])


TEXT: dict[str, dict[str, str]] = {
    "pt_br": {
        "rows": "registros", "exposure_years": "exposição (anos)",
        "claims": "sinistros", "rate100": "taxa por 100 apólices-ano",
        "claimnb_title": "Distribuição de ClaimNb (treino, escala log)",
        "claimnb_x": "ClaimNb", "exposure_title":
            "Distribuição de Exposure (treino)",
        "exposure_x": "Exposure (anos)", "exactly_one": "= 1 ano",
        "low_support": "baixo suporte", "ok_support": "suporte suficiente",
        "short_title": "Observado / esperado pela taxa do perfil",
        "short_y": "razão observado/esperado (escala log)",
        "band": "faixa de exposição", "pred_minus_obs":
            "prevista − observada (por 100 apólices-ano)",
        "cal_title": "Calibração por faixa de exposição (validação)",
        "forest_title": "Diferenças pareadas e IC 95% (validação)",
        "cost_title": "Tempo de parede por ajuste", "seconds": "segundos",
        "degree": "degrau de ajuste", "candidate": "candidato",
        "zero_title": "Zeros previstos − observados (validação)",
        "age_veh": "idade do veículo", "age_drv": "idade do condutor",
        "region": "região", "area": "Area",
        "validation": "validação", "test": "teste",
        "panel_title": "Painel de calibração: prevista − observada",
        "paired_title": "Desvio médio: Poisson − taxa global (teste)",
        "val_vs_test_title": "Faixas de exposição: validação e teste",
        "bootstrap_main": "bootstrap por IDpol",
        "bootstrap_profile": "bootstrap por perfil",
        "observed": "observada", "poisson_offset": "Poisson (offset fixo)",
        "poisson_free": "Poisson (log-exposição livre, diagnóstico)",
        "rates_title":
            "Taxas por faixa de exposição (validação)",
        "stability_title": "Desvio: Poisson − taxa global por degrau",
        "dev_diff": "diferença de desvio médio (IC 95%)",
    },
    "en_us": {
        "rows": "records", "exposure_years": "exposure (years)",
        "claims": "claims", "rate100": "rate per 100 policy-years",
        "claimnb_title": "Distribution of ClaimNb (training, log scale)",
        "claimnb_x": "ClaimNb", "exposure_title":
            "Distribution of Exposure (training)",
        "exposure_x": "Exposure (years)", "exactly_one": "= 1 year",
        "low_support": "low support", "ok_support": "adequate support",
        "short_title": "Observed / expected from the profile rate",
        "short_y": "observed/expected ratio (log scale)",
        "band": "exposure band", "pred_minus_obs":
            "predicted − observed (per 100 policy-years)",
        "cal_title": "Calibration by exposure band (validation)",
        "forest_title": "Paired differences and 95% CI (validation)",
        "cost_title": "Wall time per fit", "seconds": "seconds",
        "degree": "fit step", "candidate": "candidate",
        "zero_title": "Predicted − observed zeros (validation)",
        "age_veh": "vehicle age", "age_drv": "driver age",
        "region": "region", "area": "Area",
        "validation": "validation", "test": "test",
        "panel_title": "Calibration panel: predicted − observed",
        "paired_title": "Mean deviance: Poisson − global rate (test)",
        "val_vs_test_title": "Exposure bands: validation and test",
        "bootstrap_main": "bootstrap by IDpol",
        "bootstrap_profile": "profile bootstrap",
        "observed": "observed", "poisson_offset": "Poisson (fixed offset)",
        "poisson_free": "Poisson (free log-exposure, diagnostic)",
        "rates_title":
            "Rates by exposure band (validation)",
        "stability_title": "Deviance: Poisson − global rate by step",
        "dev_diff": "mean deviance difference (95% CI)",
    },
}


def text(key: str, language: str = "pt_br") -> str:
    """Devolva o rotulo ``key`` no idioma pedido.

    Args:
        key: chave do mapa ``TEXT``.
        language: ``pt_br`` ou ``en_us``.

    Raises:
        ValueError: se o idioma nao for suportado.
        KeyError: se a chave nao existir.
    """
    if language not in LANGUAGES:
        raise ValueError(f"idioma nao suportado: {language}")
    return TEXT[language][key]
