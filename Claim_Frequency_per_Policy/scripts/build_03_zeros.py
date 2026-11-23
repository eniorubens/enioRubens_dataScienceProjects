"""Construa 03_diagnostico_zeros.ipynb (PT-BR e EN-US).

Os numeros da narrativa sao lidos de ``outputs/etapa_03`` e ``etapa_04`` na
construcao. Ao ser executado, o notebook apenas carrega e exibe os
artefatos apos verificar a origem e o congelamento.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from notebook_common import EN, PT, Builder, num

ROOT = Path(__file__).resolve().parents[1]
O3, O4 = ROOT / "outputs" / "etapa_03", ROOT / "outputs" / "etapa_04"
FULL = "train_full"


def values(lang: str) -> dict:
    """Numeros citados na narrativa, formatados para o idioma."""
    f = lambda x, d=2: num(x, d, lang)  # noqa: E731
    dec = pd.read_csv(O4 / "model_decision.csv")
    dec["degree"] = dec["degree"].astype(str)
    bands = pd.read_csv(O4 / "exposure_band_rates.csv").set_index("band")
    zer = pd.read_csv(O4 / "zero_diagnostics.csv")
    mz = pd.read_csv(O4 / "mean_vs_zero_by_band.csv")
    gate = json.loads((O4 / "zero_gate.json").read_text("utf-8"))[
        "by_candidate"]
    frozen = json.loads((O4 / "frozen_model.json").read_text("utf-8"))
    ver = json.loads((O4 / "verify_freeze.json").read_text("utf-8"))
    main = dec[(dec["degree"] == FULL) & (dec["population"] == "validation")
               ].iloc[0]
    d1 = dec[(dec["degree"] == "100000")].iloc[0]
    d2 = dec[(dec["degree"] == "200000")].iloc[0]
    grp = dec[dec["population"] == "grouped_validation"].iloc[0]

    def z(cand, band, col):
        return zer[(zer["candidate"] == cand) & (zer["band"] == band)
                   ].iloc[0][col]

    def ztxt(cand, band):
        return (f(z(cand, band, "diff_pred_minus_obs"), 1),
                f(z(cand, band, "ci_low"), 1), f(z(cand, band, "ci_high"), 1))

    zp, zn, zg, zf = (ztxt(c, "overall") for c in (
        "poisson", "nb2", "global", "poisson_free_exposure"))
    zf2 = ztxt("poisson_free_exposure", "(0.02, 0.08]")
    zf4 = ztxt("poisson_free_exposure", "(0.25, 0.50]")
    tot = mz[mz["band"] == "total"].set_index("candidate")
    return {
        "dev_g": f(main["dev_global"], 5), "dev_p": f(main["dev_poisson"], 5),
        "dev_n": f(main["dev_nb2"], 5),
        "d_pg": f(main["d_dev_poisson_minus_global"], 5),
        "d_pg_lo": f(main["d_dev_poisson_minus_global_lo"], 5),
        "d_pg_hi": f(main["d_dev_poisson_minus_global_hi"], 5),
        "e_g": f(main["abs_rate_err_global"], 3),
        "e_p": f(main["abs_rate_err_poisson"], 3),
        "e_n": f(main["abs_rate_err_nb2"], 3),
        "e_p1": f(d1["abs_rate_err_poisson"], 3),
        "e_g1": f(d1["abs_rate_err_global"], 3),
        "e_p2": f(d2["abs_rate_err_poisson"], 3),
        "e_g2": f(d2["abs_rate_err_global"], 3),
        "d_pg1": f(d1["d_dev_poisson_minus_global"], 5),
        "d_pg2": f(d2["d_dev_poisson_minus_global"], 5),
        "ll": f(main["d_ll_nb2_minus_poisson"], 5),
        "ll_lo": f(main["d_ll_nb2_minus_poisson_lo"], 5),
        "ll_hi": f(main["d_ll_nb2_minus_poisson_hi"], 5),
        "dn": f(main["d_dev_nb2_minus_poisson"], 5),
        "dn_lo": f(main["d_dev_nb2_minus_poisson_lo"], 5),
        "dn_hi": f(main["d_dev_nb2_minus_poisson_hi"], 5),
        "g_ll": f(grp["d_ll_nb2_minus_poisson"], 5),
        "g_dn": f(grp["d_dev_nb2_minus_poisson"], 5),
        "g_dn_lo": f(grp["d_dev_nb2_minus_poisson_lo"], 5),
        "g_dn_hi": f(grp["d_dev_nb2_minus_poisson_hi"], 5),
        "g_en": f(grp["abs_rate_err_nb2"], 3),
        "g_ep": f(grp["abs_rate_err_poisson"], 3),
        "obs1": f(bands.loc["(0, 0.02]", "observed_rate_per_100"], 2),
        "pp1": f(bands.loc["(0, 0.02]", "predicted_rate_poisson"], 2),
        "pf1": f(bands.loc["(0, 0.02]",
                           "predicted_rate_poisson_free_exposure"], 2),
        "obs_one": f(bands.loc["= 1", "observed_rate_per_100"], 2),
        "pp_one": f(bands.loc["= 1", "predicted_rate_poisson"], 2),
        "pf_one": f(bands.loc["= 1", "predicted_rate_poisson_free_exposure"],
                    2),
        "free_abs": f(abs(bands.loc["total", "diff_poisson_free_exposure"]),
                      3),
        "pois_abs": f(abs(bands.loc["total", "diff_poisson"]), 3),
        "zp": zp[0], "zp_lo": zp[1], "zp_hi": zp[2],
        "zn": zn[0], "zn_lo": zn[1], "zn_hi": zn[2],
        "zg": zg[0], "zg_lo": zg[1], "zg_hi": zg[2],
        "zf": zf[0], "zf_lo": zf[1], "zf_hi": zf[2],
        "zf2": zf2[0], "zf2_lo": zf2[1], "zf2_hi": zf2[2],
        "zf4": zf4[0], "zf4_lo": zf4[1], "zf4_hi": zf4[2],
        "p_bands": ", ".join(f"`{b}`" for b in gate["poisson"][
            "bands_with_support_ci_below_zero"]),
        "resc_p": f(tot.loc["poisson", "zeros_diff_if_band_rescaled"], 1),
        "resc_f": f(tot.loc["poisson_free_exposure",
                            "zeros_diff_if_band_rescaled"], 1),
        "n_cols": f(len(frozen["estimator"]["columns"]), 0),
        "n_fit": f(frozen["estimator"]["n_fit"], 0),
        "rate100": f(100 * frozen["global_rate"]["rate"], 3),
        "maxdiff": f(ver["frozen_reference"]["max_abs_diff"], 12),
        "n_cmp": f(ver["frozen_reference"]["n_compared"], 0),
    }


def build() -> Builder:
    """Defina as celulas do notebook de diagnostico de zeros."""
    b = Builder()
    b.md("""# Diagnóstico de zeros, decisão de modelo e congelamento

Este notebook formaliza, na partição de **validação**, a decisão entre taxa global, Poisson e NB2, preserva a interpretação da exposição, diagnostica os zeros previstos e documenta o congelamento do Poisson com offset fixo e da taxa global de referência. Cada seção segue a sequência pergunta → execução → insights. Os resultados são carregados dos artefatos das etapas 3 e 4 após a verificação de origem. O teste permanece reservado para a etapa 5 e nenhuma de suas linhas é usada aqui. Nenhum ZIP ou ZINB foi ajustado.""",
         """# Zero diagnostics, model decision and freezing

This notebook formalizes, on the **validation** partition, the decision among the global rate, Poisson and NB2, preserves the interpretation of exposure, diagnoses predicted zeros and documents the freezing of the fixed-offset Poisson and of the global reference rate. Each section follows the sequence question → execution → insights. Results are loaded from the stage 3 and stage 4 artifacts after the origin check. The test partition remains reserved for stage 5 and none of its rows is used here. No ZIP or ZINB was fitted.""")
    b.code('''%matplotlib inline
import pandas as pd
from IPython.display import display

from freq_sinistros import artifacts, plots

LANG = "__LANG__"
ROOT = artifacts.project_root()
pd.options.display.float_format = "{:,.4f}".format
pd.options.display.max_columns = 40
s3 = artifacts.load_outputs(ROOT)
s4 = artifacts.load_stage4(ROOT)''')
    b.md("""## 1. Origem dos resultados

**Pergunta.** Os resultados derivam do arquivo, das partições, do protocolo e do artefato congelado aprovados?""",
         """## 1. Origin of the results

**Question.** Do the results derive from the approved file, partitions, protocol and frozen artifact?""")
    b.code('''display(pd.Series(artifacts.verify_origin(ROOT)["checks"], name="ok"))
display(pd.Series(artifacts.verify_stage4(ROOT)["checks"], name="ok"))''')
    b.md("""### Insights

Todas as verificações foram satisfeitas: o ARFF, as atribuições, o protocolo emendado e os arquivos de resultados da etapa 3 conferem com os hashes registrados no congelamento, o reajuste determinístico reproduziu a etapa 3 e a verificação do congelamento terminou sem reajuste. A partição de teste não foi usada.""",
         """### Insights

All checks were satisfied: the ARFF, the assignments, the amended protocol and the stage 3 result files match the hashes recorded at freezing, the deterministic refit reproduced stage 3, and the freeze verification finished without any refit. The test partition was not used.""")
    b.md("""## 2. Decisão de modelo pela regra aprovada

**Pergunta.** A regra aprovada seleciona o Poisson com offset fixo na partição principal, e por que o ganho de log-verossimilhança do NB2 não basta?""",
         """## 2. Model decision by the approved rule

**Question.** Does the approved rule select the fixed-offset Poisson on the main partition, and why is the log-likelihood gain of NB2 not enough?""")
    b.code('''decision = s4["model_decision"]
display(artifacts.localize(decision, LANG))
plots.show(plots.plot_stability(decision, LANG))''')
    b.md("""### Insights

No treino inteiro (partição principal), o desvio médio é {dev_g} (taxa global), {dev_p} (Poisson) e {dev_n} (NB2). A diferença pareada Poisson − taxa global é {d_pg} (IC 95% {d_pg_lo} a {d_pg_hi}; valores negativos favorecem o Poisson), e o erro absoluto da diferença de taxa global é {e_p} para o Poisson e {e_g} para a taxa global, de modo que as duas condições do parecer são cumpridas. O sinal do ganho de desvio é o mesmo nos três degraus ({d_pg1} com 100 mil, {d_pg2} com 200 mil e {d_pg} no treino inteiro). A condição de calibração global, porém, não se cumpre nos degraus menores: o erro do Poisson é {e_p1} contra {e_g1} (100 mil) e {e_p2} contra {e_g2} (200 mil). O NB2 melhora a log-verossimilhança frente ao Poisson em {ll} (IC {ll_lo} a {ll_hi}), mas seu desvio médio é maior em {dn} (IC {dn_lo} a {dn_hi}) e seu erro de taxa global é {e_n}, maior que o do Poisson; as condições do NB2 não são cumpridas e o ganho de verossimilhança, isolado, é insuficiente para selecioná-lo. Na partição agrupada, que é uma sensibilidade com população própria, as condições do NB2 são cumpridas (diferença de desvio {g_dn}, IC {g_dn_lo} a {g_dn_hi}; erro de taxa global {g_en} contra {g_ep}); esse resultado mede a generalização a perfis distintos e não altera a decisão da partição principal.""",
         """### Insights

On the full training set (main partition), the mean deviance is {dev_g} (global rate), {dev_p} (Poisson) and {dev_n} (NB2). The paired Poisson − global rate difference is {d_pg} (95% CI {d_pg_lo} to {d_pg_hi}; negative values favor Poisson), and the absolute overall-rate error is {e_p} for Poisson and {e_g} for the global rate, so both conditions of the opinion are met. The sign of the deviance gain is the same in the three steps ({d_pg1} with 100,000, {d_pg2} with 200,000 and {d_pg} on the full training set). The overall-calibration condition, however, is not met at the smaller steps: the Poisson error is {e_p1} against {e_g1} (100,000) and {e_p2} against {e_g2} (200,000). NB2 improves the log-likelihood over Poisson by {ll} (CI {ll_lo} to {ll_hi}), but its mean deviance is higher by {dn} (CI {dn_lo} to {dn_hi}) and its overall-rate error is {e_n}, larger than Poisson's; the NB2 conditions are not met and the likelihood gain alone is insufficient to select it. On the grouped partition, which is a sensitivity with its own population, the NB2 conditions are met (deviance difference {g_dn}, CI {g_dn_lo} to {g_dn_hi}; overall-rate error {g_en} against {g_ep}); this result measures generalization to distinct profiles and does not change the decision on the main partition.""")
    b.md("""## 3. Interpretação da exposição

**Pergunta.** Como se comparam, por faixa de exposição, a taxa observada e as previstas pelo Poisson com offset fixo e pelo diagnóstico com coeficiente livre de `log(Exposure)`? As taxas são `100 × Σμ / Σ Exposure` do grupo, calculadas das previsões diretas.""",
         """## 3. Interpretation of exposure

**Question.** How do the observed rate and the rates predicted by the fixed-offset Poisson and by the diagnostic with a free `log(Exposure)` coefficient compare by exposure band? Rates are `100 × Σμ / Σ Exposure` of the group, computed from direct predictions.""")
    b.code('''bands = s4["exposure_band_rates"]
cols = ["band", "rows", "exposure", "observed_rate_per_100",
        "predicted_rate_poisson", "diff_poisson", "diff_poisson_lo",
        "diff_poisson_hi", "predicted_rate_poisson_free_exposure",
        "diff_poisson_free_exposure", "diff_poisson_free_exposure_lo",
        "diff_poisson_free_exposure_hi", "low_support"]
display(artifacts.localize(bands[cols], LANG))
plots.show(plots.plot_exposure_rates(bands, LANG))''')
    b.md("""### Insights

Na faixa `(0; 0,02]`, a taxa observada é {obs1} por 100 apólices-ano, a prevista pelo Poisson com offset é {pp1} e a prevista pelo ajuste livre é {pf1}: a melhora é parcial e o desvio permanece expressivo. Em exatamente um ano, o observado é {obs_one}, o offset prevê {pp_one} e o ajuste livre, {pf_one}. O ajuste livre reduz os desvios na maioria das faixas (não em `(0,25; 0,50]`), mas o erro absoluto da diferença de taxa global sobe de {pois_abs} para {free_abs}. Os intervalos de 95% vêm do bootstrap pareado da etapa 3 e só existem onde há suporte. O ajuste livre é mantido como diagnóstico de associação com a duração observada no benchmark; a relação não sustenta extrapolação causal ou prospectiva da frequência para outra duração.""",
         """### Insights

In the `(0; 0.02]` band, the observed rate is {obs1} per 100 policy-years, the offset Poisson predicts {pp1} and the free fit predicts {pf1}: the improvement is partial and a substantial gap remains. At exactly one year, the observed value is {obs_one}, the offset predicts {pp_one} and the free fit {pf_one}. The free fit reduces the gaps in most bands (not in `(0.25; 0.50]`), but the absolute overall-rate error rises from {pois_abs} to {free_abs}. The 95% intervals come from the paired bootstrap of stage 3 and exist only where there is support. The free fit is kept as a diagnostic of association with the duration observed in the benchmark; the relation does not support a causal or prospective extrapolation of the frequency to another duration.""")
    b.md("""## 4. Diagnóstico formal de zeros

**Pergunta.** Quantos zeros são previstos, `Σ P(Y_i = 0)`, em comparação com os observados, no total e por faixa? Para os candidatos Poisson (incluindo a taxa global), `P(Y_i = 0) = exp(−μ_i)`; para o NB2, usa-se a expressão com a dispersão estimada.""",
         """## 4. Formal zero diagnostic

**Question.** How many zeros are predicted, `Σ P(Y_i = 0)`, compared with those observed, in total and by band? For the Poisson candidates (including the global rate), `P(Y_i = 0) = exp(−μ_i)`; for NB2, the expression with the estimated dispersion is used.""")
    b.code('''zeros = s4["zero_diagnostics"]
display(artifacts.localize(zeros, LANG))
plots.show(plots.plot_zero_diff(
    zeros, LANG, ("poisson", "nb2", "poisson_free_exposure")))
display(artifacts.zero_reconciliation_ok(s4))
display(artifacts.zero_gate_table(s4))''')
    b.md("""### Insights

As faixas reconciliam com o total para todos os candidatos (registros, exposição, zeros observados e previstos). No total, a diferença previstos − observados é {zg} para a taxa global (IC {zg_lo} a {zg_hi}), {zp} para o Poisson com offset (IC {zp_lo} a {zp_hi}) e {zn} para o NB2 (IC {zn_lo} a {zn_hi}); os três intervalos incluem zero, de modo que o critério numérico da etapa 2 não é cumprido no modelo de referência. O Poisson com offset superestima zeros nas faixas curtas e subestima nas longas ({p_bands} têm intervalo abaixo de zero), isto é, o sinal depende da duração. O diagnóstico com coeficiente livre cumpre a parte numérica: {zf} no total (IC {zf_lo} a {zf_hi}), com déficits em `(0,02; 0,08]` ({zf2}; IC {zf2_lo} a {zf2_hi}) e em `(0,25; 0,50]` ({zf4}; IC {zf4_lo} a {zf4_hi}), ambas com suporte. Essa diferença entre os modelos é registrada separadamente: o critério se aplica ao modelo de referência, e o resultado do ajuste livre é um diagnóstico.

Um déficit indica que foram previstos menos zeros do que os observados. No total da validação, a alta proporção de apólices sem sinistro foi compatível com o Poisson de offset fixo. O déficit no diagnóstico com coeficiente livre mostra que essa comparação é sensível à forma de modelar a exposição. Como o déficit global do modelo de referência não se distinguiu de zero e o contrato de dados não sustentou a hipótese de um segundo processo que sempre gerasse zero, não foi necessário ajustar ZIP ou ZINB nesta avaliação.""",
         """### Insights

The bands reconcile with the total for all candidates (records, exposure, observed and predicted zeros). Overall, the predicted − observed difference is {zg} for the global rate (CI {zg_lo} to {zg_hi}), {zp} for the offset Poisson (CI {zp_lo} to {zp_hi}) and {zn} for NB2 (CI {zn_lo} to {zn_hi}); all three intervals include zero, so the numerical criterion of stage 2 is not met by the reference model. The offset Poisson overestimates zeros in short bands and underestimates them in long bands ({p_bands} have an interval below zero), that is, the sign depends on duration. The free-coefficient diagnostic meets the numerical part: {zf} overall (CI {zf_lo} to {zf_hi}), with deficits in `(0.02; 0.08]` ({zf2}; CI {zf2_lo} to {zf2_hi}) and in `(0.25; 0.50]` ({zf4}; CI {zf4_lo} to {zf4_hi}), both with support. This difference between models is recorded separately: the criterion applies to the reference model, and the free-fit result is a diagnostic.

A deficit means that fewer zeros were predicted than observed. Across the validation set, the high share of policies with no claims was consistent with the fixed-offset Poisson model. The deficit in the free-coefficient diagnostic shows that this comparison is sensitive to how exposure is modeled. Since the overall deficit for the reference model was not distinguishable from zero and the data contract did not support a second process that always generates zeros, fitting ZIP or ZINB was unnecessary for this evaluation.""")
    b.md("""## 5. Erros nas médias e diferenças de zeros

**Pergunta.** Os erros nas contagens por duração coexistem com as diferenças de zeros? A coluna reescalada ajusta `μ` por faixa para somar as contagens observadas e recalcula os zeros previstos; é apenas um diagnóstico e nunca é usada em taxas ou previsões.""",
         """## 5. Mean errors and zero differences

**Question.** Do the count errors by duration coexist with the zero differences? The rescaled column adjusts `μ` by band to add up to the observed counts and recomputes the predicted zeros; it is only a diagnostic and is never used for rates or predictions.""")
    b.code('''display(artifacts.localize(s4["mean_vs_zero_by_band"], LANG))''')
    b.md("""### Insights

No Poisson com offset, os erros de contagem por faixa têm sinal oposto ao das diferenças de zeros: onde as contagens são subestimadas, os zeros são superestimados, e vice-versa. Quando `μ` é reescalado por faixa para igualar as contagens observadas, as diferenças de zeros por faixa se reduzem a poucas dezenas, com saldo de {resc_p} no Poisson e {resc_f} no diagnóstico livre. Esse saldo residual é pequeno e não tem intervalo associado; ele não permite concluir por um segundo processo gerador de zeros.""",
         """### Insights

In the offset Poisson, the count errors by band have the opposite sign to the zero differences: where counts are underestimated, zeros are overestimated, and vice versa. When `μ` is rescaled by band to match the observed counts, the zero differences by band shrink to a few tens, with a balance of {resc_p} for Poisson and {resc_f} for the free diagnostic. This residual balance is small and has no associated interval; it does not allow concluding for a second zero-generating process.""")
    b.md("""## 6. O que o contrato de dados sustenta sobre um segundo processo

**Pergunta.** O dicionário e o esquema sustentam uma hipótese plausível para um segundo processo gerador de zeros, requisito separado do critério numérico?""",
         """## 6. What the data contract supports about a second process

**Question.** Do the dictionary and schema support a plausible hypothesis for a second zero-generating process, a requirement separate from the numerical criterion?""")
    b.code('''display(artifacts.second_process_table(s4))''')
    b.md("""### Insights

O dicionário define `ClaimNb` como número de sinistros no período de exposição e `Exposure` como duração em anos (Dutang & Charpentier, 2018); não descreve apólices sem risco, não há datas nem identificador de unidade superior, e o instante de medição de `BonusMalus` é desconhecido. Assim, nenhuma hipótese de segundo processo é sustentada pelo contrato. O teste de Vuong não é usado como regra de seleção, pois seu emprego para testar inflação de zeros é questionado na literatura (Wilson, 2015). Como o modelo de referência não cumpre o critério numérico e não há hipótese documentada, a decisão desta rodada é encerrar com os modelos de contagem já avaliados, sem ajustar ZIP ou ZINB.""",
         """### Insights

The dictionary defines `ClaimNb` as the number of claims in the exposure period and `Exposure` as the duration in years (Dutang & Charpentier, 2018); it does not describe risk-free policies, there are neither dates nor a higher-level unit identifier, and the measurement time of `BonusMalus` is unknown. Hence, no hypothesis of a second process is supported by the contract. The Vuong test is not used as a selection rule, because its use to test zero inflation is questioned in the literature (Wilson, 2015). Since the reference model does not meet the numerical criterion and no hypothesis is documented, the decision of this round is to close with the count models already evaluated, without fitting ZIP or ZINB.""")
    b.md("""## 7. Modelo congelado e verificação

**Pergunta.** O que foi congelado, e a verificação restaura o artefato e reproduz a validação aprovada sem reajustar nem reaprender o pré-processamento?""",
         """## 7. Frozen model and verification

**Question.** What was frozen, and does the verification restore the artifact and reproduce the approved validation without refitting or relearning the preprocessing?""")
    b.code('''frozen = s4["frozen_model"]
display(pd.Series({
    "model": frozen["model"], "reference": frozen["reference"],
    "columns": len(frozen["estimator"]["columns"]),
    "n_fit": frozen["estimator"]["n_fit"],
    "global_rate": frozen["global_rate"]["rate"],
    "bonus_malus_included": frozen["estimator"]["bonus_malus_included"],
    "training_sample_sha256": frozen["training_sample"]["ids_sha256"]}))
display(pd.Series(frozen["hashes"]))
display(pd.Series({k: v for k, v in s4["verify_freeze"].items()
                   if not isinstance(v, dict)}))
display(pd.Series(s4["verify_freeze"]["frozen_reference"]))''')
    b.md("""### Insights

O artefato congela o Poisson com offset fixo, ajustado em {n_fit} registros do treino com {n_cols} colunas e sem `BonusMalus`, e a taxa global de {rate100} por 100 apólices-ano calculada no mesmo treino, com o pré-processador já ajustado, a ordem das colunas, os coeficientes e os hashes de origem. A verificação restaurou o estado persistido, sem reajustar o estimador nem reaprender o pré-processamento, e reproduziu {n_cmp} agregados de validação com diferença absoluta máxima de {maxdiff}, dentro da tolerância declarada (`rtol = 1e-9`, `atol = 1e-12`); os coeficientes são persistidos com 17 dígitos significativos e restaurados sem perda. O desvio por duração (seção 3) é uma limitação do Poisson selecionado: o offset fixo subestima as exposições muito curtas e superestima as longas.""",
         """### Insights

The artifact freezes the fixed-offset Poisson, fitted on {n_fit} training records with {n_cols} columns and without `BonusMalus`, and the global rate of {rate100} per 100 policy-years computed on the same training set, together with the already fitted preprocessor, the column order, the coefficients and the origin hashes. The verification restored the persisted state, without refitting the estimator or relearning the preprocessing, and reproduced {n_cmp} validation aggregates with a maximum absolute difference of {maxdiff}, within the declared tolerance (`rtol = 1e-9`, `atol = 1e-12`); coefficients are persisted with 17 significant digits and restored without loss. The deviation by duration (section 3) is a limitation of the selected Poisson: the fixed offset underestimates very short exposures and overestimates long ones.""")
    b.md("""## 8. Roteiro da etapa 5 e limitações

**Pergunta.** Como a avaliação final será aplicada, e o que deve ser declarado como limitação?""",
         """## 8. Stage 5 script and limitations

**Question.** How will the final evaluation be applied, and what must be stated as a limitation?""")
    b.code('''script = s4["stage5_script"]
display(pd.Series({k: v for k, v in script.items()
                   if k in ("model", "reference", "primary_metric",
                            "paired_comparison", "calibration",
                            "test_access", "outputs")}))
display(pd.Series(script["limitations_to_state"], name="limitation"))''')
    b.md("""### Insights

A etapa 5 aplicará o roteiro uma única vez à partição de teste, após o aceite, com o Poisson congelado e a taxa global de referência, o desvio médio de Poisson, a diferença pareada contra a taxa global, a calibração das previsões diretas por segmento com a faixa mais curta visível, as reconciliações e a verificação dos hashes. As limitações a declarar são o desvio por duração do offset fixo, a restrição ao benchmark, a exclusão de `BonusMalus` e a ausência de ZIP e ZINB. Os resultados são internos ao benchmark e `production_eligible` permanece falso.""",
         """### Insights

Stage 5 will apply the script once to the test partition, after acceptance, with the frozen Poisson and the global reference rate, the mean Poisson deviance, the paired difference against the global rate, the calibration of direct predictions by segment with the shortest band visible, the reconciliations and the hash verification. The limitations to state are the duration bias of the fixed offset, the restriction to the benchmark, the exclusion of `BonusMalus` and the absence of ZIP and ZINB. The results are internal to the benchmark and `production_eligible` remains false.""")
    return b


REFS = ["dutang2018", "wilson2015"]


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
        path = ROOT / "notebooks" / lang / "03_diagnostico_zeros.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, path)
        print("escrito", path)


if __name__ == "__main__":
    main()
