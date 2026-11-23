"""Parametros congelados do protocolo de avaliacao (etapa 2).

Esta e a fonte unica dos parametros: o relatorio e o arquivo JSON legivel
por maquina sao gerados daqui. Nenhum valor depende do desfecho de
validacao ou de teste.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from freq_sinistros import metrics, partition, profiles, segments

SAMPLE_PREFIX = "freMTPL2freq-v1-etapa2-amostra|"
SAMPLE_LADDER = [100_000, 200_000, None]  # None = treino inteiro
FIT_LIMITS = {"max_fit_wall_s": 600, "max_peak_ram_gib": 8,
              "convergence_required": True}


def nested_train_sample(ids: pd.Series, n: int | None) -> pd.Series:
    """Amostra reprodutivel e aninhada do treino, por hash do ``IDpol``.

    Os ``n`` menores valores de ``SHA-256(prefixo + IDpol)`` formam a
    amostra; a amostra de ``n`` e prefixo da de ``m > n``. Independe da
    ordem das linhas.

    Args:
        ids: ``IDpol`` das linhas de treino.
        n: tamanho da amostra; ``None`` devolve todo o treino.

    Returns:
        ``IDpol`` selecionados, em ordem crescente.
    """
    ids = ids.astype("int64")
    if n is None or n >= len(ids):
        return ids.sort_values(kind="stable").reset_index(drop=True)
    keyed = sorted(ids.tolist(), key=lambda i: hashlib.sha256(
        f"{SAMPLE_PREFIX}{i}".encode("utf-8")).digest())
    return pd.Series(sorted(keyed[:n]), dtype="int64")


def protocol_config() -> dict:
    """Devolva todos os parametros do protocolo como dicionario."""
    return {
        "production_eligible": False,
        "population": "678.013 registros do ARFF OpenML 41214 v1; "
                      "caudas validas preservadas",
        "partition": {
            "unit": "IDpol canonico", "rule": partition.RULE_TEXT,
            "modulus": partition.MODULUS,
            "bounds": {"train": [0, partition.TRAIN_END],
                       "validation": [partition.TRAIN_END,
                                      partition.VALIDATION_END],
                       "test": [partition.VALIDATION_END,
                                partition.MODULUS]},
            "test_policy": "so integridade e tamanho nesta etapa; "
                           "avaliado uma unica vez apos congelar modelo e "
                           "roteiro (etapa 4)",
        },
        "profile_key": {
            "columns": profiles.PROFILE_COLUMNS,
            "separator": profiles.SEPARATOR,
            "missing_token": profiles.MISSING_TOKEN,
            "grouped_partition": {
                "scope": "treino + validacao apenas",
                "rule": "bucket(sha256(prefixo+chave)[:8]) % 10000 < "
                        f"{profiles.GROUP_TRAIN_END} -> group_train",
                "prefix": profiles.GROUP_PREFIX,
            },
        },
        "model_main": {
            "covariates": profiles.PROFILE_COLUMNS,
            "sensitivity_extra_covariate": "BonusMalus",
            "target": "ClaimNb (escala de contagem)",
            "offset": "log(Exposure), coeficiente fixo 1 (Poisson, NB2)",
            "baseline": "taxa global do treino = sum(ClaimNb)/sum(Exposure); "
                        "mu_i = taxa * Exposure_i",
            "nb2_variance": "mu + alpha*mu^2",
        },
        "training_sample": {
            "method": "menores SHA-256(prefixo+IDpol) do treino; amostras "
                      "aninhadas",
            "prefix": SAMPLE_PREFIX,
            "ladder": SAMPLE_LADDER,
            "advance_rule": "ampliar ao proximo degrau se o ajuste do "
                            "degrau atual convergir, levar <= "
                            f"{FIT_LIMITS['max_fit_wall_s']} s e usar "
                            f"<= {FIT_LIMITS['max_peak_ram_gib']} GiB de "
                            "pico; senao fixar o degrau atual",
            "limits": FIT_LIMITS,
        },
        "segments": {
            "exposure_bands": segments.BAND_LABELS,
            "exposure_invalid": segments.INVALID_BAND,
            "vehicle_age": {"labels": segments.VEHAGE_LABELS,
                            "lower_inclusive": segments.VEHAGE_LOWER,
                            "invalid": segments.VEHAGE_INVALID},
            "region": "codigo cadastral; ausente/desconhecido = "
                      + segments.REGION_MISSING,
            "low_support": {"rows_lt": segments.LOW_SUPPORT_ROWS,
                            "exposure_years_lt":
                                segments.LOW_SUPPORT_EXPOSURE},
        },
        "metrics": {
            "primary": "desvio medio de Poisson por registro: (2/n) "
                       "sum[y log(y/mu) - (y-mu)]; termo y log(y/mu)=0 "
                       "se y=0; mu finita e > 0 exigida",
            "invalid_policy": "ClaimNb, Exposure ou mu invalidos "
                              "interrompem a avaliacao; nenhuma linha e "
                              "descartada em silencio",
            "population": "mesmas linhas de validacao para todos os "
                          "candidatos, com contagens e exposicoes "
                          "originais",
            "calibration": "totais previstos e observados; diferenca de "
                           "taxas prevista menos observada, por 100 "
                           "apolices-ano, com a exposicao total do grupo",
            "distribution": "log-verossimilhanca preditiva media por "
                            "registro; zeros previstos = sum P(Y_i=0)",
        },
        "uncertainty": {
            "method": "bootstrap percentil, modelos ajustados fixos, "
                      "reamostragem de registros (IDpol) com reposicao",
            "replicas": metrics.BOOTSTRAP_REPLICAS,
            "seed": metrics.BOOTSTRAP_SEED,
            "level": metrics.CI_LEVEL,
            "paired": "mesmos indices para todos os modelos; diferencas "
                      "pareadas",
            "undefined_policy": "replicas com denominador zero ficam fora "
                                "do indicador e sao contadas; intervalo "
                                "so e informado se <= 1% das replicas",
            "sensitivity": "bootstrap por chave de perfil (conglomerados)",
        },
    }


def amendment_stage3(eda_summary: dict | None = None) -> dict:
    """Emenda da etapa 3 ao protocolo (gravada antes do primeiro ajuste).

    Incorpora as decisoes do parecer da etapa 2, corrige o escopo das
    reconciliacoes da etapa 2 e registra as verificacoes derivadas da EDA.

    Args:
        eda_summary: resumo da EDA do treino (opcional); seus numeros
            entram na secao ``eda_derived_checks``.

    Returns:
        Dicionario serializavel.
    """
    short = (eda_summary or {}).get("short_exposure_check", {})
    offset = (eda_summary or {}).get("offset_check", {})
    return {
        "production_eligible": False,
        "decisions_from_stage2_opinion": {
            "sample_ladder": [100_000, 200_000, "train_full"],
            "fit_limits": FIT_LIMITS,
            "ladder_rule": "ampliar se os tres candidatos convergirem e "
                           "couberem nos limites; se um degrau falhar, "
                           "usar o ultimo degrau comum elegivel; se o "
                           "primeiro falhar, devolver diagnostico",
            "low_support": {"rows_lt": segments.LOW_SUPPORT_ROWS,
                            "exposure_years_lt":
                                segments.LOW_SUPPORT_EXPOSURE},
            "bootstrap": "1000 replicas pareadas por IDpol (principal); "
                         "por chave de perfil (sensibilidade); intervalos "
                         "condicionados aos modelos ajustados",
            "choice_rules": {
                "poisson_over_global": "IC95% pareado da melhora no desvio "
                                       "medio exclui zero E |diferenca de "
                                       "taxa global| do Poisson <= a da "
                                       "taxa global (estimativas pontuais)",
                "nb2_over_poisson": "IC95% pareado da melhora na "
                                    "log-verossimilhanca media exclui zero "
                                    "E desvio medio do NB2 <= Poisson E "
                                    "|diferenca de taxa global| do NB2 <= "
                                    "Poisson (estimativas pontuais)",
                "nb2_vs_global_if_global_still_selected": "NB2 precisa "
                    "ganho pareado no desvio medio contra a taxa global",
                "stability": "ganho com IC excluindo zero no maior degrau "
                             "viavel e estimativa pontual com o mesmo "
                             "sinal em todos os degraus viaveis; com um "
                             "unico degrau, estabilidade por tamanho "
                             "'nao verificada' e comparacao provisoria",
                "convergence": "exigida em todos os candidatos",
            },
            "zero_inflation_gate": "ZIP/ZINB na etapa 4 somente se o modelo "
                "de contagem de referencia SUBESTIMAR zeros: IC95% de "
                "(zeros previstos - observados) inteiramente abaixo de zero "
                "no total e em >= 2 faixas de exposicao com suporte, e "
                "hipotese plausivel documentada",
        },
        "reconciliation_scope_correction": (
            "Na etapa 2 foram reconciliados registros e exposicao em "
            "treino e validacao e sinistros apenas no treino; nao houve "
            "reconciliacao de sinistros na validacao nem qualquer uso do "
            "teste alem da contagem de IDs. A reconciliacao de sinistros "
            "da validacao ocorre pela primeira vez na etapa 3."),
        "eda_derived_checks": {
            "source": "outputs/etapa_03/eda_summary.json (somente treino)",
            "short_exposure_vs_profile": {
                "method": "estratos faixa de idade do condutor x faixa de "
                          "idade do veiculo; taxa de referencia com "
                          "Exposure >= 0,75 por estrato com suporte; razao "
                          "observado/esperado por faixa de exposicao",
                "result": short,
            },
            "offset_assumption": {
                "assumed": "E[ClaimNb] proporcional a Exposure "
                           "(coeficiente 1 em log(Exposure))",
                "train_diagnostic_free_coefficient": offset,
                "implication": "se o coeficiente livre for bem menor que "
                               "1, a taxa por ano e maior nas coberturas "
                               "curtas e o offset subestima as faixas "
                               "curtas e superestima as longas",
                "validation_check": "calibracao por faixa de exposicao "
                                    "(prevista - observada, por 100 "
                                    "apolices-ano, com IC) para os tres "
                                    "candidatos; sem recalibracao pos-hoc "
                                    "nem fatores por segmento",
                "exploratory_diagnostic": "Poisson com log(Exposure) como "
                    "covariavel livre, avaliado em validacao e rotulado "
                    "diagnostico: nao e candidato; seu uso exigiria nova "
                    "decisao do Codex na etapa 4",
            },
            "extremes": "caudas validas preservadas; casos com ClaimNb > 4 "
                        "concentram-se em poucos perfis; bootstrap por "
                        "perfil e a sensibilidade correspondente",
            "category_coverage": "niveis com < 500 linhas de ajuste vao "
                                 "para OTHER; niveis desconhecidos vao "
                                 "para OTHER (ou referencia se nao houver "
                                 "OTHER); verificado em linhas de treino "
                                 "fora da amostra de ajuste",
        },
        "operational_definitions": {
            "evaluation_population": "todas as 101.573 linhas de validacao "
                                     "com contagens e exposicoes originais",
            "rate_difference": "100 * (soma(mu) - soma(ClaimNb)) / "
                               "soma(Exposure)",
            "segment_intervals": "apenas segmentos sem baixo suporte na "
                                 "validacao",
            "sensitivities": ["BonusMalus", "limites didaticos so no "
                              "ajuste", "bootstrap por perfil",
                              "particao agrupada do desenvolvimento",
                              "faixa (0; 0,02] excluida (descritiva)",
                              "log(Exposure) livre (diagnostico)"],
            "sensitivity_degree": "maior degrau viavel",
            "grouped_partition": "pre-processamento, ajustes e avaliacao "
                                 "refeitos em group_train/group_validation, "
                                 "mesmo tamanho nominal de ajuste, resultados "
                                 "reportados a parte",
        },
        "disclosure": "Antes desta emenda foi executado um teste de fumaca "
                      "do estimador em 100 mil linhas de treino (tempo e "
                      "equivalencia com o statsmodels), sem uso de validacao.",
    }
