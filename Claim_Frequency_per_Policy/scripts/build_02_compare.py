"""Construa 02_comparacao_modelos.ipynb (PT-BR e EN-US).

Os numeros da narrativa sao lidos de ``outputs/etapa_03`` na construcao.
O notebook, ao ser executado, apenas carrega e exibe esses artefatos apos
verificar a origem (hashes do ARFF, das atribuicoes e da emenda).
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from notebook_common import EN, PT, Builder, num

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "etapa_03"
FULL = "train_full"


def values(lang: str) -> dict:
    """Numeros citados na narrativa, formatados para o idioma."""
    f = lambda x, d=2: num(x, d, lang)  # noqa: E731
    cfg = json.loads((OUT / "comparison_config.json").read_text("utf-8"))
    fits = pd.read_csv(OUT / "fits.csv")
    met = pd.read_csv(OUT / "validation_metrics.csv")
    iv = pd.read_csv(OUT / "intervals.csv")
    cal = pd.read_csv(OUT / "calibration_by_segment.csv")
    zer = pd.read_csv(OUT / "zeros_preliminary.csv")
    disp = pd.read_csv(OUT / "dispersion_diagnostics.csv").set_index(
        "degree")
    cost = json.loads((OUT / "cost.json").read_text("utf-8"))
    full_preprocessing = json.loads(
        (OUT / f"preprocessing_n{FULL}.json").read_text("utf-8"))
    main = met[(met["population"] == "validation")
               & (met["subset"] == "all_rows")]
    full = main[main["degree"].astype(str) == FULL].set_index("candidate")
    mi = iv[(iv["degree"].astype(str) == FULL)
            & (iv["population"] == "validation")].set_index("indicator")
    ci = iv[iv["population"] == "validation_cluster_bootstrap"].set_index(
        "indicator")

    def est(frame, key, d=5):
        r = frame.loc[key]
        return (f(r["estimate"], d), f(r["ci_low"], d), f(r["ci_high"], d))

    def sens(pop, key, d=5):
        r = iv[(iv["population"] == pop)
               & (iv["indicator"] == key)].iloc[0]
        return (f(r["estimate"], d), f(r["ci_low"], d), f(r["ci_high"], d))

    band = cal[(cal["population"] == "validation")
               & (cal["degree"].astype(str) == FULL)
               & (cal["family"] == "exposure_band")
               & (cal["candidate"] == "poisson")].set_index("segment")
    zp = zer[(zer["population"] == "validation")
             & (zer["degree"].astype(str) == FULL)
             & (zer["candidate"] == "poisson")].set_index("band")
    zn = zer[(zer["population"] == "validation")
             & (zer["degree"].astype(str) == FULL)
             & (zer["candidate"] == "nb2")].set_index("band")
    main_fits = fits[fits["context"] == "main"]
    nb_full = main_fits[(main_fits["degree"].astype(str) == FULL)
                        & (main_fits["candidate"] == "nb2")].iloc[0]
    po_full = main_fits[(main_fits["degree"].astype(str) == FULL)
                        & (main_fits["candidate"] == "poisson")].iloc[0]
    d_pg, d_np = est(mi, "d_dev|poisson-global"), est(mi, "d_ll|nb2-poisson")
    d_dnp = est(mi, "d_dev|nb2-poisson")
    d_png = est(ci, "d_dev|poisson-global")
    bonus = sens("validation_bonus_malus", "d_dev|poisson_bonus-poisson")
    clip = sens("validation_clipped_fit", "d_dev|poisson_clipfit-poisson")
    free = sens("validation_free_exposure_diagnostic",
                "d_dev|poisson_free_exposure-poisson")
    g_pg = sens("grouped_validation", "d_dev|poisson-global")
    g_nl = sens("grouped_validation", "d_ll|nb2-poisson")
    g_nd = sens("grouped_validation", "d_dev|nb2-poisson")
    exc = main[(main["subset"] == "all_rows")]
    short_desc = met[(met["subset"] == "exposure_gt_0.02_descriptive")
                     ].set_index("candidate")
    free_coef = met[met["population"]
                    == "validation_free_exposure_diagnostic"]
    free_coef = float(free_coef["free_log_exposure_coef"].dropna().iloc[0])
    grp = met[met["population"] == "grouped_validation"].set_index(
        "candidate")
    gp = met[(met["population"] == "grouped_validation")
             & (met["candidate"] == "poisson")].iloc[0]
    sd = short_desc.loc["poisson"]
    fm = met[met["candidate"] == "poisson_free_exposure"].iloc[0]
    fb = cal[(cal["population"] == "validation_free_exposure_diagnostic")
             & (cal["candidate"] == "poisson_free_exposure")
             & (cal["segment"] == "(0, 0.02]")].iloc[0]
    zf = zer[(zer["population"] == "validation_free_exposure_diagnostic")
             & (zer["candidate"] == "poisson_free_exposure")
             ].set_index("band")
    nbm = main_fits[(main_fits["degree"].astype(str) == FULL)
                    & (main_fits["candidate"] == "nb2")].iloc[0]
    raises = (short_desc.loc["poisson", "mean_poisson_deviance"]
              > full.loc["poisson", "mean_poisson_deviance"])
    return {
        "n_grp": f(gp["rows"], 0), "claims_grp": f(gp["observed_claims"], 0),
        "n_sub": f(sd["rows"], 0), "claims_sub": f(sd["observed_claims"], 0),
        "short_dir_pt": "aumenta" if raises else "diminui",
        "short_dir_en": "increases" if raises else "decreases",
        "dev_p_all": f(full.loc["poisson", "mean_poisson_deviance"], 5),
        "dev_g_all": f(full.loc["global", "mean_poisson_deviance"], 5),
        "free_pred_b1": f(fb["predicted_rate_per_100"], 2),
        "free_obs_b1": f(fb["observed_rate_per_100"], 2),
        "free_abs": f(fm["abs_rate_error"], 3),
        "pois_abs": f(full.loc["poisson", "abs_rate_error"], 3),
        "zf_all": f(zf.loc["overall", "diff_pred_minus_obs"], 1),
        "zf_all_lo": f(zf.loc["overall", "ci_low"], 1),
        "zf_all_hi": f(zf.loc["overall", "ci_high"], 1),
        "zf_b2": f(zf.loc["(0.02, 0.08]", "diff_pred_minus_obs"], 1),
        "zf_b2_lo": f(zf.loc["(0.02, 0.08]", "ci_low"], 1),
        "zf_b2_hi": f(zf.loc["(0.02, 0.08]", "ci_high"], 1),
        "zf_b4": f(zf.loc["(0.25, 0.50]", "diff_pred_minus_obs"], 1),
        "zf_b4_lo": f(zf.loc["(0.25, 0.50]", "ci_low"], 1),
        "zf_b4_hi": f(zf.loc["(0.25, 0.50]", "ci_high"], 1),
        "nb_ram": f(nbm["peak_rss_mib"], 0),
        "n_val": f(cfg["validation"]["rows"], 0),
        "exp_val": f(cfg["validation"]["exposure"]),
        "claims_val": f(cfg["validation"]["observed_claims"], 0),
        "zeros_val": f(cfg["validation"]["observed_zeros"], 0),
        "n_features_full": f(full_preprocessing["n_columns"], 0),
        "dev_g": f(full.loc["global", "mean_poisson_deviance"], 5),
        "dev_p": f(full.loc["poisson", "mean_poisson_deviance"], 5),
        "dev_n": f(full.loc["nb2", "mean_poisson_deviance"], 5),
        "ll_p": f(full.loc["poisson", "mean_loglik"], 5),
        "ll_n": f(full.loc["nb2", "mean_loglik"], 5),
        "rd_g": f(full.loc["global", "rate_diff_pred_minus_obs"], 3),
        "rd_p": f(full.loc["poisson", "rate_diff_pred_minus_obs"], 3),
        "rd_n": f(full.loc["nb2", "rate_diff_pred_minus_obs"], 3),
        "obs_rate": f(full.loc["poisson", "observed_rate_per_100"], 2),
        "d_pg": d_pg[0], "d_pg_lo": d_pg[1], "d_pg_hi": d_pg[2],
        "d_np": d_np[0], "d_np_lo": d_np[1], "d_np_hi": d_np[2],
        "d_dnp": d_dnp[0], "d_dnp_lo": d_dnp[1], "d_dnp_hi": d_dnp[2],
        "d_png": d_png[0], "d_png_lo": d_png[1], "d_png_hi": d_png[2],
        "alpha": f(nb_full["alpha"], 3),
        "phi": f(disp.loc[FULL, "pearson_dispersion_phi"], 2),
        "ct_alpha": f(disp.loc[FULL, "cameron_trivedi_alpha"], 3),
        "ct_t": f(disp.loc[FULL, "cameron_trivedi_t"], 1),
        "df_full": f(disp.loc[FULL, "degrees_of_freedom"], 0),
        "n_full": f(disp.loc[FULL, "population_rows"], 0),
        "t_nb": f(nb_full["wall_s"], 1), "t_po": f(po_full["wall_s"], 1),
        "ram_max": f(cost["peak_rss_mib"], 0),
        "total_s": f(cost["total_wall_s"], 0),
        "obs_b1": f(band.loc["(0, 0.02]", "observed_rate_per_100"], 1),
        "pred_b1": f(band.loc["(0, 0.02]", "predicted_rate_per_100"], 1),
        "obs_one": f(band.loc["= 1", "observed_rate_per_100"], 2),
        "pred_one": f(band.loc["= 1", "predicted_rate_per_100"], 2),
        "diff_b1": f(band.loc["(0, 0.02]",
                              "rate_difference_pred_minus_obs"], 1),
        "zp_all": f(zp.loc["overall", "diff_pred_minus_obs"], 1),
        "zp_all_lo": f(zp.loc["overall", "ci_low"], 1),
        "zp_all_hi": f(zp.loc["overall", "ci_high"], 1),
        "zn_all": f(zn.loc["overall", "diff_pred_minus_obs"], 1),
        "zn_all_lo": f(zn.loc["overall", "ci_low"], 1),
        "zn_all_hi": f(zn.loc["overall", "ci_high"], 1),
        "zp_short": f(zp.loc["(0.02, 0.08]", "diff_pred_minus_obs"], 1),
        "zp_one": f(zp.loc["= 1", "diff_pred_minus_obs"], 1),
        "bonus": bonus[0], "bonus_lo": bonus[1], "bonus_hi": bonus[2],
        "clip": clip[0], "clip_lo": clip[1], "clip_hi": clip[2],
        "free": free[0], "free_lo": free[1], "free_hi": free[2],
        "free_coef": f(free_coef, 2),
        "g_pg": g_pg[0], "g_pg_lo": g_pg[1], "g_pg_hi": g_pg[2],
        "g_nl": g_nl[0], "g_nl_lo": g_nl[1], "g_nl_hi": g_nl[2],
        "g_nd": g_nd[0], "g_nd_lo": g_nd[1], "g_nd_hi": g_nd[2],
        "g_rd_p": f(grp.loc["poisson", "rate_diff_pred_minus_obs"], 3),
        "g_rd_n": f(grp.loc["nb2", "rate_diff_pred_minus_obs"], 3),
        "short_dev_p": f(short_desc.loc["poisson", "mean_poisson_deviance"],
                         5),
        "short_dev_g": f(short_desc.loc["global", "mean_poisson_deviance"],
                         5),
        "short_rows": f(short_desc.loc["poisson", "rows"], 0),
        "rd_p100": f(main[(main["degree"].astype(str) == "100000")
                          & (main["candidate"] == "poisson")][
            "rate_diff_pred_minus_obs"].iloc[0], 3),
        "rd_g100": f(main[(main["degree"].astype(str) == "100000")
                          & (main["candidate"] == "global")][
            "rate_diff_pred_minus_obs"].iloc[0], 3),
        "cl_pg_lo": d_png[1], "cl_pg_hi": d_png[2],
        "n_exc": f(len(exc), 0),
    }


def build() -> Builder:
    """Defina as celulas do notebook de comparacao."""
    b = Builder()
    b.md("""# Comparação em validação: taxa global, Poisson e NB2

Na partição de **validação**, é comparada a capacidade de três candidatos de prever o número de sinistros por apólice. A taxa global do treino serve de referência para medir o ganho das regressões de Poisson e binomial negativa NB2.

Os resultados foram gerados pelo comando `python -m freq_sinistros compare`. Ao executar o notebook, suas origens são verificadas antes da exibição das tabelas e dos gráficos. A decisão final do modelo é registrada na etapa 4, e a partição de teste é reservada para a etapa 5.""",
         """# Validation comparison: global rate, Poisson and NB2

The ability of three candidates to predict the number of claims per policy is compared on the **validation** partition. The training global rate serves as the baseline for measuring gains from Poisson and NB2 negative binomial regression.

The results were generated with `python -m freq_sinistros compare`. When the notebook runs, their origins are verified before the tables and figures are displayed. The final model decision is recorded in stage 4, and the test partition is reserved for stage 5.""")
    b.code('''%matplotlib inline
import pandas as pd
from IPython.display import display

from freq_sinistros import artifacts, plots

LANG = "__LANG__"
ROOT = artifacts.project_root()
pd.options.display.float_format = "{:,.5f}".format
pd.options.display.max_columns = 30
FULL = "train_full"''')
    b.md("""## 1. Origem dos resultados

**Pergunta.** Os resultados exibidos derivam do arquivo, das partições e do protocolo aprovados?

**Onde os modelos foram ajustados.** Os ajustes e a avaliação em validação foram executados no terminal, a partir da raiz do projeto, com `python -I -m freq_sinistros --root . compare`. As métricas, os registros de ajuste e os parâmetros do pré-processamento foram gravados em `outputs/etapa_03/`. Ao executar este notebook, a origem dos resultados é conferida e as saídas já gravadas são usadas para exibir tabelas e gráficos. Nenhum modelo é reajustado nesta leitura.

**Como a tabela analítica foi montada.** Para o Poisson e o NB2, foi construída em memória uma matriz numérica a partir da própria base, com uma linha por apólice. No treino completo, a matriz teve **{n_full} linhas e {n_features_full} colunas**, incluindo o intercepto. Foram aplicadas três transformações às oito covariáveis do cenário principal.

- Área, potência, marca, combustível e região foram codificados em colunas indicadoras de 0 e 1, com agrupamento de categorias raras.
- Idade do veículo e idade do condutor foram convertidas em faixas por quantis.
- A densidade populacional foi transformada em log e padronizada.

As categorias, os cortes e os parâmetros de padronização foram aprendidos no treino de cada ajuste e reaplicados na validação. `ClaimNb` foi usado como alvo, `Exposure` como offset e `IDpol` como identificador de rastreabilidade. A taxa global foi calculada diretamente dividindo o total de sinistros pelo total de exposição do treino. [Parâmetros do pré-processamento no treino completo](../../outputs/etapa_03/preprocessing_ntrain_full.json).""",
         """## 1. Origin of the results

**Question.** Do the displayed results derive from the approved file, partitions and protocol?

**Where the models were fitted.** Model fitting and validation were run in the terminal from the project root using `python -I -m freq_sinistros --root . compare`. Metrics, fit records, and preprocessing parameters were saved in `outputs/etapa_03/`. When this notebook is run, the origin of the results is checked and the saved outputs are used to display tables and figures. No model is refitted during this process.

**How the analytical table was built.** For Poisson and NB2, a numeric matrix with one row per policy was built in memory from the same dataset. With the full training set, the matrix contained **{n_full} rows and {n_features_full} columns**, including the intercept. Three transformations were applied to the eight covariates in the main specification.

- Area, vehicle power, brand, fuel, and region were encoded as 0/1 indicator columns, with rare categories pooled.
- Vehicle age and driver age were converted into quantile-based bands.
- Population density was log-transformed and standardized.

Categories, band boundaries, and standardization parameters were learned from the training rows for each fit and applied to validation. `ClaimNb` was used as the target, `Exposure` as the offset, and `IDpol` as the identifier for traceability. The global rate was calculated directly by dividing the total number of training claims by the total training exposure. [Full-training preprocessing parameters](../../outputs/etapa_03/preprocessing_ntrain_full.json).""")
    b.code('''origin = artifacts.verify_origin(ROOT)
display(pd.Series(origin["checks"], name="ok"))
out = artifacts.load_outputs(ROOT)
display(pd.Series({"arff_sha256": origin["arff_sha256"],
                   "assignment_sha256": origin["assignment_sha256"],
                   "all_ok": origin["all_ok"]}))''')
    b.md("""### Insights

Todas as verificações de origem foram satisfeitas: o SHA-256 do ARFF confere com o manifesto da aquisição, as atribuições coincidem com a configuração da etapa 2, a emenda ao protocolo é a mesma registrada antes do primeiro ajuste em validação, e nenhuma linha de teste é devolvida pelo carregador nem usada. O ARFF integral é interpretado em memória, mas só as partições de treino e validação são mantidas. A população de avaliação tem {n_val} registros de validação, {exp_val} apólices-ano, {claims_val} sinistros observados e {zeros_val} zeros.""",
         """### Insights

All origin checks were satisfied: the ARFF SHA-256 matches the acquisition manifest, the assignments coincide with the stage 2 configuration, the protocol amendment is the one recorded before the first validation fit, and no test row is returned by the loader or used. The whole ARFF is parsed in memory, but only the training and validation partitions are kept. The evaluation population has {n_val} validation records, {exp_val} policy-years, {claims_val} observed claims and {zeros_val} zeros.""")
    b.md("""## 2. Método

**Pergunta.** O que distingue os três candidatos e como foi feita a comparação?

Em todos os candidatos foi prevista a contagem esperada de sinistros (`ClaimNb`) para o período coberto por cada apólice (`Exposure`, em anos). Na tabela, essa contagem esperada é representada por `μ`.

| Candidato | Como a previsão é calculada | O que é representado |
| --- | --- | --- |
| `global` — taxa global | A taxa é calculada dividindo-se o total de sinistros pela soma das exposições no treino. Em seguida, é multiplicada pela exposição de cada apólice. | Uma única taxa anual é aplicada a toda a carteira, sem covariáveis. É a referência da comparação. |
| `poisson` — regressão de Poisson | É estimada uma taxa anual `exp(x′β)` a partir das covariáveis da apólice. A média é `μ = Exposure · exp(x′β)`. | A taxa varia por perfil e a variância da contagem, para um dado perfil e exposição, é igual a `μ`. |
| `nb2` — regressão binomial negativa NB2 | É usada a mesma forma para a média do Poisson, com coeficientes próprios. É estimado também o parâmetro de dispersão `α`. | A variância é `μ + αμ²`. Com `α > 0`, é acomodada uma variabilidade maior que a média mesmo após o ajuste pelas covariáveis. |

Para a média do Poisson e do NB2, foi usada a estrutura dos **modelos lineares generalizados (GLM) com ligação log**. Na fórmula, `x` reúne as covariáveis da apólice e `β` representa os coeficientes estimados. Na comparação principal, `log(Exposure)` foi incluído como **offset**, um termo com coeficiente fixado em 1. Na prática, para um mesmo perfil, dobrar o tempo de cobertura dobra a contagem esperada. A taxa global aplica essa mesma proporcionalidade ao multiplicar a taxa única pela exposição (Cameron & Trivedi, 2013).

**Implementação em Python.** Os três estimadores foram implementados em `freq_sinistros.models`. No Poisson, os coeficientes foram estimados por mínimos quadrados ponderados iterativos (IRLS), uma forma de obter o ajuste de máxima verossimilhança. No NB2, o ajuste de `β` por IRLS foi alternado com a otimização de `α` por máxima verossimilhança, com convergência conjunta verificada pelo escore (Cameron & Trivedi, 2013). Foi usado NumPy para a álgebra linear e SciPy para a otimização de `α`. O ajuste de Poisson também foi conferido contra `statsmodels.GLM`.

**Protocolo de comparação.** Em cada degrau, as mesmas linhas de treino foram usadas para os três candidatos, e o pré-processamento foi ajustado apenas nessas linhas. Foram executados os degraus aninhados de 100 mil, 200 mil e do treino inteiro, com limites de 600 s e 8 GiB por ajuste. Na validação, a métrica primária foi o desvio médio de Poisson por registro, complementado pela log-verossimilhança preditiva média e pela calibração por segmento. Foram calculados intervalos percentílicos de 95% com 1.000 réplicas bootstrap pareadas por `IDpol`, mantendo os modelos fixos.""",
         """## 2. Method

**Question.** What distinguishes the three candidates, and how were they compared?

For each candidate, the expected claim count (`ClaimNb`) was predicted for the period covered by each policy (`Exposure`, in years). This expected count is denoted by `μ` in the table.

| Candidate | How the prediction is calculated | What is represented |
| --- | --- | --- |
| `global` — global rate | The total number of claims is divided by the total exposure in training. This rate is multiplied by the exposure of each policy. | A single annual rate is applied across the portfolio, with no covariates. It is the comparison baseline. |
| `poisson` — Poisson regression | An annual rate `exp(x′β)` is estimated from the policy covariates. The mean is `μ = Exposure · exp(x′β)`. | The rate varies by profile, and the count variance for a given profile and exposure equals `μ`. |
| `nb2` — NB2 negative binomial regression | The same form for the mean as in Poisson regression is used, with separately fitted coefficients. The dispersion parameter `α` is also estimated. | The variance is `μ + αμ²`. When `α > 0`, variability beyond the mean is allowed even after adjusting for covariates. |

The Poisson and NB2 means were specified using a **generalized linear model (GLM) with a log link**. In the formula, `x` contains the policy covariates and `β` represents the estimated coefficients. In the main comparison, `log(Exposure)` was included as an **offset**, a term with its coefficient fixed at 1. For a given profile, doubling the coverage period doubles the expected claim count. The global rate applies the same proportionality by multiplying the single rate by exposure (Cameron & Trivedi, 2013).

**Python implementation.** The three estimators were implemented in `freq_sinistros.models`. For Poisson, the coefficients were estimated by iteratively reweighted least squares (IRLS), which obtains the maximum-likelihood fit. For NB2, IRLS updates for `β` were alternated with maximum-likelihood optimization of `α`, with joint convergence checked using the score (Cameron & Trivedi, 2013). NumPy was used for linear algebra and SciPy to optimize `α`. The Poisson fit was also checked against `statsmodels.GLM`.

**Comparison protocol.** Within each training step, the same rows were used for all three candidates, and preprocessing was fitted on those rows only. The nested steps of 100,000, 200,000 and the full training set were run, with limits of 600 s and 8 GiB per fit. On validation, the primary metric was mean Poisson deviance per record, complemented by mean predictive log-likelihood and calibration by segment. Percentile 95% intervals were calculated from 1,000 bootstrap replicates paired by `IDpol`, with the fitted models kept fixed.""")
    b.md("""## 3. Convergência, custo e dispersão

**Pergunta.** Os ajustes convergiram dentro dos limites, e há dispersão condicional após ajustar por perfil e exposição?

Na tabela de ajustes, **`context` identifica o cenário em que o ajuste foi executado**. O estimador é indicado por `candidato` (`candidate` no CSV), o tamanho planejado pelo degrau (`degree`) e o número de registros efetivamente usados por `n_fit`. Uma linha com degrau `100000`, `context = main` e candidato `poisson`, por exemplo, corresponde ao Poisson ajustado com 100 mil registros no cenário principal.

| Valor de `context` | Cenário de ajuste |
| --- | --- |
| `main` | Foi usada a partição principal por `IDpol`, com as contagens e exposições originais. O Poisson e o NB2 foram ajustados sem `BonusMalus` e com offset fixo. |
| `sens_bonus_malus` | Foi acrescentada a variável `BonusMalus` às regressões para medir a sensibilidade dos resultados à sua inclusão. |
| `sens_clipped_fit` | Foram aplicados os limites de 4 sinistros e 1 ano de exposição somente no ajuste. Na validação, os valores originais foram mantidos. |
| `diagnostic_free_exposure` | Foi ajustado um Poisson com coeficiente livre para `log(Exposure)` para examinar a hipótese de proporcionalidade. Ele aparece como `poisson_free_exposure` na coluna `candidato`. |
| `sens_grouped_partition` | Treino e validação foram reparticionados por perfis de covariáveis, de modo que cada perfil fosse mantido em um único lado. Os ajustes e a avaliação foram feitos nessa divisão alternativa. |""",
         """## 3. Convergence, cost and dispersion

**Question.** Did the fits converge within the limits, and is there conditional dispersion after adjusting for profile and exposure?

In the fit table, **`context` identifies the scenario in which the fit was run**. The estimator is shown under `candidate`, the planned training step under `degree`, and the number of records actually used under `n_fit`. A row with training step `100000`, `context = main` and candidate `poisson`, for example, represents the Poisson fit on 100,000 records in the main scenario.

| `context` value | Fit scenario |
| --- | --- |
| `main` | The main split by `IDpol` was used with the original counts and exposures. Poisson and NB2 were fitted without `BonusMalus` and with a fixed offset. |
| `sens_bonus_malus` | `BonusMalus` was added to the regressions to measure how sensitive the results were to its inclusion. |
| `sens_clipped_fit` | Claim counts were capped at 4 and exposure at 1 year during fitting only. The original values were kept for validation. |
| `diagnostic_free_exposure` | A Poisson model with a freely estimated coefficient for `log(Exposure)` was fitted to examine the proportionality assumption. It is listed as `poisson_free_exposure` under `candidate`. |
| `sens_grouped_partition` | Training and validation were split again by covariate profile so that each profile was kept on one side. Fitting and evaluation were carried out on this alternative split. |""")
    b.code('''fits = out["fits"]
cols = ["degree", "context", "candidate", "n_fit", "converged",
        "iterations", "wall_s", "peak_rss_mib", "alpha",
        "alpha_at_poisson_limit", "within_limits"]
display(artifacts.localize(fits[cols], LANG))
plots.show(plots.plot_cost(fits, LANG))
display(artifacts.localize(out["dispersion_diagnostics"], LANG))''')
    b.md("""### Insights

Todos os ajustes principais e de sensibilidade convergiram e couberam nos limites; o maior degrau viável é o treino inteiro, e os três degraus foram verificados. No treino inteiro, o Poisson levou {t_po} s e o NB2, {t_nb} s; o pico de memória do ajuste NB2 foi de {nb_ram} MiB, e o do processo completo de `compare`, de {ram_max} MiB (ajuste de diagnóstico com `log(Exposure)` livre). Tempos e picos são os de `fits.csv` e `cost.json`, da mesma execução que gerou as demais tabelas. A dispersão estimada do NB2 foi `α = {alpha}`, longe do limite de Poisson (`α → 0`), e o escore no ótimo foi desprezível (abaixo de 1e-6 por linha). Para o diagnóstico condicional, a população é o Poisson ajustado nas {n_full} linhas do degrau, com {df_full} graus de liberdade. A estatística de Pearson dividida pelos graus de liberdade foi {phi}. O estimador de Cameron e Trivedi (1990) para `α` foi {ct_alpha} (estatística t robusta {ct_t}), em linha com a dispersão acima de 1. A estatística de Pearson pode ser sensível a linhas com média prevista muito pequena, de modo que o valor deve ser lido junto ao estimador de `α`.""",
         """### Insights

All main and sensitivity fits converged and fit within the limits; the largest viable step is the full training set, and all three steps were verified. On the full training set, Poisson took {t_po} s and NB2 {t_nb} s; the memory peak of the NB2 fit was {nb_ram} MiB, and that of the whole `compare` process was {ram_max} MiB (free `log(Exposure)` diagnostic fit). Times and peaks are those of `fits.csv` and `cost.json`, from the same run that produced the other tables. The estimated NB2 dispersion was `α = {alpha}`, far from the Poisson limit (`α → 0`), and the score at the optimum was negligible (below 1e-6 per row). For the conditional diagnostic, the population is the Poisson fitted on the {n_full} rows of the step, with {df_full} degrees of freedom. The Pearson statistic divided by the degrees of freedom was {phi}. The Cameron and Trivedi (1990) estimator of `α` was {ct_alpha} (robust t statistic {ct_t}), in line with dispersion above 1. The Pearson statistic can be sensitive to rows with a very small predicted mean, so the value should be read together with the estimator of `α`.""")
    b.md("""## 4. Métricas em validação

**Pergunta.** Qual candidato prediz melhor as contagens em validação, e as diferenças são distinguíveis da variabilidade da amostra?""",
         """## 4. Validation metrics

**Question.** Which candidate predicts counts best in validation, and are the differences distinguishable from sampling variability?""")
    b.code('''display(artifacts.localize(artifacts.main_metrics(out), LANG))
keys = artifacts.KEY_INDICATORS
sub = artifacts.select_intervals(out["intervals"], FULL, "validation", keys)
display(artifacts.localize(sub[["indicator", "estimate", "ci_low",
                                "ci_high"]], LANG))
plots.show(plots.plot_forest(sub, LANG, keys))''')
    b.md("""### Insights

No treino inteiro, o desvio médio de Poisson foi {dev_g} (taxa global), {dev_p} (Poisson) e {dev_n} (NB2). A diferença de desvio Poisson − taxa global é {d_pg} (IC 95% {d_pg_lo} a {d_pg_hi}; valores negativos favorecem o Poisson); o intervalo exclui zero, e o resultado se mantém no bootstrap por perfil ({cl_pg_lo} a {cl_pg_hi}). O NB2 melhora a log-verossimilhança média frente ao Poisson em {d_np} (IC {d_np_lo} a {d_np_hi}), mas seu desvio médio é maior em {d_dnp} (IC {d_dnp_lo} a {d_dnp_hi}), isto é, suas médias previstas são ligeiramente piores. A diferença de taxa global (prevista − observada, por 100 apólices-ano, com taxa observada de {obs_rate}) foi {rd_g} (global), {rd_p} (Poisson) e {rd_n} (NB2). Esses são os valores publicados, sem recalibração.""",
         """### Insights

On the full training set, the mean Poisson deviance was {dev_g} (global rate), {dev_p} (Poisson) and {dev_n} (NB2). The Poisson − global rate deviance difference is {d_pg} (95% CI {d_pg_lo} to {d_pg_hi}; negative values favor Poisson); the interval excludes zero, and the result holds under the profile bootstrap ({cl_pg_lo} to {cl_pg_hi}). NB2 improves the mean log-likelihood over Poisson by {d_np} (CI {d_np_lo} to {d_np_hi}), but its mean deviance is higher by {d_dnp} (CI {d_dnp_lo} to {d_dnp_hi}), that is, its predicted means are slightly worse. The overall rate difference (predicted − observed, per 100 policy-years, with an observed rate of {obs_rate}) was {rd_g} (global), {rd_p} (Poisson) and {rd_n} (NB2). These are the published values, without recalibration.""")
    b.md("""## 5. Regras de escolha (provisórias)

**Pergunta.** O que as regras do parecer indicam em cada degrau e na partição agrupada? A decisão de modelo pertence à etapa 4.""",
         """## 5. Selection rules (provisional)

**Question.** What do the opinion's rules indicate at each step and on the grouped partition? The model decision belongs to stage 4.""")
    b.code('''display(artifacts.rules_table(out))''')
    b.md("""### Insights

No treino inteiro, o Poisson satisfaz as duas condições frente à taxa global (ganho de desvio com intervalo excluindo zero e erro absoluto de taxa global menor), enquanto o NB2 não satisfaz as condições frente ao Poisson, pois a melhora na log-verossimilhança vem acompanhada de desvio e de erro de taxa global maiores. A condição de calibração global do Poisson frente à taxa global não se sustenta nos degraus menores (diferenças de {rd_p100} contra {rd_g100} no degrau de 100 mil), e o ganho no desvio mantém o mesmo sinal nos três degraus. Assim, a preferência pelo Poisson é estável quanto ao desvio e instável quanto ao erro de taxa global entre degraus. Na partição agrupada, as condições do NB2 são satisfeitas, ao contrário da partição principal; o resultado depende da população de avaliação.""",
         """### Insights

On the full training set, Poisson satisfies both conditions against the global rate (deviance gain with an interval excluding zero and a smaller absolute overall-rate error), whereas NB2 does not satisfy the conditions against Poisson, because the log-likelihood improvement comes with a larger deviance and overall-rate error. The overall-calibration condition of Poisson against the global rate does not hold at the smaller steps (differences of {rd_p100} versus {rd_g100} at the 100,000 step), and the deviance gain keeps the same sign across the three steps. Thus, the preference for Poisson is stable in deviance and unstable in overall-rate error across steps. On the grouped partition, the NB2 conditions are satisfied, unlike the main partition; the result depends on the evaluation population.""")
    b.md("""## 6. Calibração por segmento

**Pergunta.** Onde as taxas previstas se afastam das observadas? As diferenças são prevista − observada, por 100 apólices-ano, calculadas com a exposição total do grupo. Segmentos de baixo suporte são mostrados sem intervalo.""",
         """## 6. Calibration by segment

**Question.** Where do predicted rates depart from observed rates? Differences are predicted − observed, per 100 policy-years, computed with the total exposure of the group. Low-support segments are shown without an interval.""")
    b.code('''cal = out["calibration_by_segment"]
plots.show(plots.plot_calibration_bands(
    cal[(cal["population"] == "validation")
        & (cal["degree"].astype(str) == FULL)], LANG))
for family in ("exposure_band", "vehicle_age", "region"):
    display(artifacts.localize(
        artifacts.calibration_pivot(out, family, FULL), LANG))''')
    b.md("""### Insights

O offset com coeficiente 1 não reproduz as faixas de exposição. Em `(0; 0,02]`, a taxa observada é {obs_b1} e a prevista pelo Poisson, {pred_b1} (diferença de {diff_b1}); em exatamente um ano, a observada é {obs_one} e a prevista, {pred_one}. Os três candidatos subestimam as faixas curtas e superestimam as longas, como antecipado pelo diagnóstico do treino, e, para o Poisson, os intervalos excluem zero em todas as faixas com suporte. Por idade do veículo e por região, os desvios são pequenos e, em geral, os intervalos incluem zero; as regiões e a faixa `> 1` de baixo suporte são mostradas sem intervalo. Nenhuma recalibração pós-hoc foi aplicada.""",
         """### Insights

The offset with coefficient 1 does not reproduce the exposure bands. In `(0; 0.02]`, the observed rate is {obs_b1} and the Poisson-predicted rate is {pred_b1} (difference of {diff_b1}); at exactly one year, the observed rate is {obs_one} and the predicted rate is {pred_one}. All three candidates underestimate short bands and overestimate long bands, as anticipated by the training diagnostic, and, for Poisson, the intervals exclude zero in all bands with support. By vehicle age and by region, deviations are small and, in general, the intervals include zero; the low-support regions and the `> 1` band are shown without an interval. No post hoc recalibration was applied.""")
    b.md("""## 7. Zeros previstos (resumo preliminar)

**Pergunta.** Quantos zeros são previstos, `Σ P(Y_i = 0)`, em comparação com os observados? O diagnóstico formal pertence à etapa 4.""",
         """## 7. Predicted zeros (preliminary summary)

**Question.** How many zeros are predicted, `Σ P(Y_i = 0)`, compared with those observed? The formal diagnostic belongs to stage 4.""")
    b.code('''zeros = artifacts.zero_table(out, FULL)
display(artifacts.localize(zeros, LANG))
plots.show(plots.plot_zero_diff(zeros, LANG))
diagnostic = artifacts.zero_table(
    out, FULL, "validation_free_exposure_diagnostic",
    ("poisson_free_exposure",))
display(artifacts.localize(diagnostic, LANG))''')
    b.md("""### Insights

No total, a diferença entre zeros previstos e observados é {zp_all} no Poisson (IC {zp_all_lo} a {zp_all_hi}) e {zn_all} no NB2 (IC {zn_all_lo} a {zn_all_hi}); ambos os intervalos incluem zero. Por faixa, o padrão tem sinais opostos: zeros são superestimados nas faixas curtas (por exemplo, {zp_short} em `(0,02; 0,08]` no Poisson) e subestimados nas faixas longas (por exemplo, {zp_one} em exatamente um ano). Isso reflete a subestimação das contagens curtas, não necessariamente um segundo processo gerador de zeros. Essa leitura vale para a taxa global, o Poisson e o NB2 com exposição proporcional.

O diagnóstico exploratório com `log(Exposure)` livre não é candidato. Ele prevê menos zeros do que os observados: {zf_all} no total (IC {zf_all_lo} a {zf_all_hi}), com déficits de {zf_b2} em `(0,02; 0,08]` (IC {zf_b2_lo} a {zf_b2_hi}) e de {zf_b4} em `(0,25; 0,50]` (IC {zf_b4_lo} a {zf_b4_hi}), ambas faixas com suporte. Esse resultado é apresentado à parte e não satisfaz, por si, o requisito de uma hipótese plausível para um segundo processo. A avaliação da inflação de zeros cabe à etapa 4.""",
         """### Insights

Overall, the difference between predicted and observed zeros is {zp_all} for Poisson (CI {zp_all_lo} to {zp_all_hi}) and {zn_all} for NB2 (CI {zn_all_lo} to {zn_all_hi}); both intervals include zero. By band, the pattern has opposite signs: zeros are overestimated in short bands (for example, {zp_short} in `(0.02; 0.08]` for Poisson) and underestimated in long bands (for example, {zp_one} at exactly one year). This reflects the underestimation of short-exposure counts, not necessarily a second zero-generating process. That reading holds for the global rate, Poisson and NB2 with proportional exposure.

The exploratory diagnostic with free `log(Exposure)` is not a candidate. It predicts fewer zeros than observed: {zf_all} overall (CI {zf_all_lo} to {zf_all_hi}), with deficits of {zf_b2} in `(0.02; 0.08]` (CI {zf_b2_lo} to {zf_b2_hi}) and {zf_b4} in `(0.25; 0.50]` (CI {zf_b4_lo} to {zf_b4_hi}), both bands with support. This result is presented separately and does not by itself meet the requirement of a plausible hypothesis for a second process. The assessment of zero inflation belongs to stage 4.""")
    b.md("""## 8. Reconciliação

**Pergunta.** Registros, exposição, sinistros observados e previstos de cada segmentação reconciliam com os totais?""",
         """## 8. Reconciliation

**Question.** Do records, exposure, observed and predicted claims of each segmentation reconcile with the totals?""")
    b.code('''display(artifacts.population_totals(out))
display(artifacts.reconciliation_summary(out))''')
    b.md("""### Insights

Cada população tem seus próprios denominadores. A validação principal tem {n_val} registros, {exp_val} apólices-ano e {claims_val} sinistros; a validação agrupada, {n_grp} registros e {claims_grp} sinistros. As sensibilidades de `BonusMalus`, de limites aplicados ao ajuste e de `log(Exposure)` livre reusam a validação principal. Em todas essas populações, a soma dos segmentos de cada segmentação coincide com os totais de registros, exposição, sinistros observados e previstos de cada candidato (uma verificação por candidato, segmentação e indicador). O recorte descritivo sem a faixa `(0; 0,02]`, com {n_sub} registros e {claims_sub} sinistros, entra apenas no resumo da métrica geral e não foi reconciliado por segmento. Os totais de registros e exposição da validação coincidem com os da etapa 2; os sinistros de validação são reconciliados aqui pela primeira vez.""",
         """### Insights

Each population has its own denominators. The main validation has {n_val} records, {exp_val} policy-years and {claims_val} claims; the grouped validation has {n_grp} records and {claims_grp} claims. The `BonusMalus`, fit-limit and free `log(Exposure)` sensitivities reuse the main validation. In all these populations, the sum of the segments of each segmentation coincides with the totals of records, exposure, observed claims and predicted claims of each candidate (one check per candidate, segmentation and indicator). The descriptive cut without the `(0; 0.02]` band, with {n_sub} records and {claims_sub} claims, enters only the overall-metric summary and was not reconciled by segment. The validation totals of records and exposure coincide with those of stage 2; validation claims are reconciled here for the first time.""")
    b.md("""## 9. Sensibilidades

**Pergunta.** Os resultados mudam com `BonusMalus`, com limites aplicados só ao ajuste, com bootstrap por perfil, com a partição agrupada ou sem a faixa muito curta? O Poisson com `log(Exposure)` livre é apenas diagnóstico.""",
         """## 9. Sensitivities

**Question.** Do the results change with `BonusMalus`, with limits applied only to the fit, with a profile bootstrap, with the grouped partition or without the very short band? The Poisson with free `log(Exposure)` is a diagnostic only.""")
    b.code('''display(artifacts.localize(artifacts.sensitivity_table(out), LANG))''')
    b.md("""### Insights

(i) Com `BonusMalus`, a diferença de desvio do Poisson é {bonus} (IC {bonus_lo} a {bonus_hi}; valores negativos favorecem a inclusão); como a disponibilidade da variável na data de estimação é indeterminada, o resultado serve de sensibilidade e não altera a especificação principal. (ii) Aplicar os limites didáticos só no ajuste dá uma diferença de desvio do Poisson de {clip} (IC {clip_lo} a {clip_hi}), desprezível. (iii) O bootstrap por perfil mantém as conclusões de sinal do bootstrap principal, com intervalos um pouco mais largos. (iv) Na partição agrupada do desenvolvimento, a diferença de desvio Poisson − taxa global é {g_pg} (IC {g_pg_lo} a {g_pg_hi}); a diferença de log-verossimilhança NB2 − Poisson é {g_nl} (IC {g_nl_lo} a {g_nl_hi}), com diferença de desvio de {g_nd} (IC {g_nd_lo} a {g_nd_hi}) e erros de taxa global de {g_rd_p} (Poisson) e {g_rd_n} (NB2). (v) Sem a faixa `(0; 0,02]` (descritivo, {short_rows} registros), o desvio médio {short_dir_pt} de {dev_p_all} para {short_dev_p} no Poisson e de {dev_g_all} para {short_dev_g} na taxa global; a métrica primária segue sobre todas as linhas. (vi) O Poisson com `log(Exposure)` livre (coeficiente {free_coef} no ajuste) tem diferença de desvio de {free} (IC {free_lo} a {free_hi}) frente ao Poisson com offset, isto é, uma redução maior do que o ganho obtido pelas covariáveis. Na faixa `(0; 0,02]`, prevê {free_pred_b1} por 100 apólices-ano, contra {free_obs_b1} observados (o Poisson com offset previu {pred_b1}): a melhora é relevante e ainda deixa um desvio expressivo. O erro absoluto da diferença de taxa global, porém, aumenta de {pois_abs} para {free_abs}; a melhora por faixa e a variação do erro global devem ser lidas em conjunto. É um diagnóstico exploratório, não um candidato, e seu uso exigiria nova decisão na etapa 4.""",
         """### Insights

(i) With `BonusMalus`, the Poisson deviance difference is {bonus} (CI {bonus_lo} to {bonus_hi}; negative values favor including it); because the availability of the variable at estimation time is indeterminate, the result serves as a sensitivity and does not change the main specification. (ii) Applying the didactic limits only to the fit gives a Poisson deviance difference of {clip} (CI {clip_lo} to {clip_hi}), negligible. (iii) The profile bootstrap keeps the sign conclusions of the main bootstrap, with slightly wider intervals. (iv) On the grouped development partition, the Poisson − global rate deviance difference is {g_pg} (CI {g_pg_lo} to {g_pg_hi}); the NB2 − Poisson log-likelihood difference is {g_nl} (CI {g_nl_lo} to {g_nl_hi}), with a deviance difference of {g_nd} (CI {g_nd_lo} to {g_nd_hi}) and overall-rate errors of {g_rd_p} (Poisson) and {g_rd_n} (NB2). (v) Without the `(0; 0.02]` band (descriptive, {short_rows} records), the mean deviance {short_dir_en} from {dev_p_all} to {short_dev_p} for Poisson and from {dev_g_all} to {short_dev_g} for the global rate; the primary metric stays on all rows. (vi) The Poisson with free `log(Exposure)` (coefficient {free_coef} in the fit) has a deviance difference of {free} (CI {free_lo} to {free_hi}) relative to the offset Poisson, that is, a reduction larger than the gain obtained from the covariates. In the `(0; 0.02]` band, it predicts {free_pred_b1} per 100 policy-years against {free_obs_b1} observed (the offset Poisson predicted {pred_b1}): the improvement is relevant and still leaves a substantial gap. The absolute overall-rate error, however, increases from {pois_abs} to {free_abs}; the per-band improvement and the change in overall error must be read together. It is an exploratory diagnostic, not a candidate, and its use would require a new decision in stage 4.""")
    b.md("""## 10. Limites e decisões que dependem da etapa 4

**Pergunta.** O que esta comparação não estabelece?""",
         """## 10. Limits and decisions that depend on stage 4

**Question.** What does this comparison not establish?""")
    b.code('''display(pd.Series({
    "viable_degrees": out["config"]["viable_degrees"],
    "largest_viable_degree": out["config"]["largest_viable_degree"],
    "test_used": False, "production_eligible": False}))''')
    b.md("""### Insights

A comparação é interna ao benchmark e condicionada aos modelos ajustados; os intervalos não incluem a variabilidade do ajuste. A calibração por faixa de exposição mostra que o offset com coeficiente 1 não descreve os dados, o que deve ser tratado na etapa 4 antes de qualquer decisão sobre inflação de zeros. A escolha entre Poisson e NB2 é sensível à população de avaliação (principal ou agrupada) e ao critério de calibração global. A decisão de modelo, a avaliação formal dos zeros e o teste final permanecem pendentes. A leitura segue a sequência de diagnóstico para dados de contagem (Fávero & Belfiore, 2017; Cameron & Trivedi, 2013).""",
         """### Insights

The comparison is internal to the benchmark and conditional on the fitted models; the intervals do not include the variability of the fit. The calibration by exposure band shows that the offset with coefficient 1 does not describe the data, which should be addressed in stage 4 before any decision on zero inflation. The choice between Poisson and NB2 is sensitive to the evaluation population (main or grouped) and to the overall-calibration criterion. The model decision, the formal zero assessment and the final test remain pending. The reading follows the diagnostic sequence for count data (Fávero & Belfiore, 2017; Cameron & Trivedi, 2013).""")
    return b


REFS = ["cameron1990", "cameron2013", "favero2017"]


class _Safe(dict):
    """Preserve chaves desconhecidas entre chaves."""

    def __missing__(self, key: str) -> str:
        """Devolva a propria chave entre chaves."""
        return "{" + key + "}"


def main() -> None:
    """Grave os dois notebooks."""
    builder = build()
    for lang in (PT, EN):
        nb = builder.build(lang, REFS)
        v = values(lang)
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.format_map(_Safe(v))
        path = ROOT / "notebooks" / lang / "02_comparacao_modelos.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, path)
        print("escrito", path)


if __name__ == "__main__":
    main()
