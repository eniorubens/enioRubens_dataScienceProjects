"""Construa 04_avaliacao_final.ipynb (PT-BR e EN-US).

Os numeros da narrativa sao lidos de ``outputs/etapa_05`` na construcao.
Ao ser executado, o notebook apenas carrega e exibe os agregados da
avaliacao oficial depois de verificar a origem; le apenas os agregados
gravados, sem ler o ARFF nem linhas de teste.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from notebook_common import EN, PT, Builder, num

ROOT = Path(__file__).resolve().parents[1]
O5 = ROOT / "outputs" / "etapa_05"


def _excluding_zero(frame: pd.DataFrame) -> list[str]:
    """Segmentos com IC inteiramente de um lado de zero."""
    ok = frame[(~frame["low_support"]) & frame["ci_low"].notna()]
    hit = ok[(ok["ci_low"] > 0) | (ok["ci_high"] < 0)]
    return hit["segment"].tolist()


def values(lang: str) -> dict:
    """Numeros citados na narrativa, formatados para o idioma."""
    f = lambda x, d=2: num(x, d, lang)  # noqa: E731
    met = pd.read_csv(O5 / "test_metrics.csv").set_index("candidate")
    cal = pd.read_csv(O5 / "test_calibration.csv")
    iv = pd.read_csv(O5 / "test_intervals.csv")
    iv = iv[iv["population"] == "test"].set_index("indicator")
    cmpb = pd.read_csv(O5 / "validation_vs_test_bands.csv").set_index("band")
    pop = json.loads((O5 / "test_population.json").read_text("utf-8"))
    rec = pd.read_csv(O5 / "test_reconciliation.csv")
    conc = json.loads((O5 / "conclusion.json").read_text("utf-8"))
    p = conc["primary"]
    c = conc["profile_cluster_bootstrap"]
    pc = cal[cal["candidate"] == "poisson"]
    veh = pc[pc["family"] == "vehicle_age"]
    reg = pc[pc["family"] == "region"]
    low_reg = reg[reg["low_support"]]["segment"].tolist()
    verdict = {"ganho_sustentado": ("sustenta um ganho", "supports a gain"),
               "inconclusivo": ("é inconclusivo", "is inconclusive"),
               "referencia_melhor": ("favorece a referência",
                                     "favors the reference")}[p["verdict"]]
    outside = [b for b, r in cmpb.iterrows()
               if not r["validation_diff_inside_test_ci"]
               and not r["test_low_support"]]
    inside_n = int((cmpb["validation_diff_inside_test_ci"]).sum())
    same_n = int(cmpb["same_sign"].sum())

    def b(band, col):
        return cmpb.loc[band, col]

    return {
        "n_test": f(pop["rows"], 0), "exp_test": f(pop["exposure"]),
        "claims_test": f(pop["observed_claims"], 0),
        "zeros_test": f(pop["observed_zeros"], 0),
        "dev_p": f(met.loc["poisson", "mean_poisson_deviance"], 5),
        "dev_g": f(met.loc["global", "mean_poisson_deviance"], 5),
        "d": f(p["d_dev_poisson_minus_global"], 5),
        "lo": f(p["ci_low"], 5), "hi": f(p["ci_high"], 5),
        "c_lo": f(c["ci_low"], 5), "c_hi": f(c["ci_high"], 5),
        "verdict_pt": verdict[0], "verdict_en": verdict[1],
        "obs_rate": f(met.loc["poisson", "observed_rate_per_100"], 3),
        "pred_p": f(met.loc["poisson", "predicted_rate_per_100"], 3),
        "pred_g": f(met.loc["global", "predicted_rate_per_100"], 3),
        "rd_p": f(met.loc["poisson", "rate_diff_pred_minus_obs"], 3),
        "rd_g": f(met.loc["global", "rate_diff_pred_minus_obs"], 3),
        "obs1": f(b("(0, 0.02]", "test_observed_rate")),
        "pred1": f(b("(0, 0.02]", "test_predicted_rate")),
        "diff1": f(b("(0, 0.02]", "test_diff"), 1),
        "lo1": f(b("(0, 0.02]", "test_ci_low"), 1),
        "hi1": f(b("(0, 0.02]", "test_ci_high"), 1),
        "val1": f(b("(0, 0.02]", "validation_diff"), 1),
        "obs_one": f(b("= 1", "test_observed_rate")),
        "pred_one": f(b("= 1", "test_predicted_rate")),
        "diff_one": f(b("= 1", "test_diff")),
        "n_same": f(same_n, 0), "n_inside": f(inside_n, 0),
        "n_bands": f(len(cmpb), 0),
        "outside": ", ".join(f"`{x}`" for x in outside) or "-",
        "veh_hit": ", ".join(f"`{x}`" for x in _excluding_zero(veh)) or "-",
        "reg_hit": ", ".join(f"`{x}`" for x in _excluding_zero(reg)) or "-",
        "low_reg": ", ".join(f"`{x}`" for x in low_reg) or "-",
        "n_checks": f(len(rec), 0),
        "ci_p_lo": f(iv.loc["ratediff|poisson|overall", "ci_low"], 3),
        "ci_p_hi": f(iv.loc["ratediff|poisson|overall", "ci_high"], 3),
        "ci_g_lo": f(iv.loc["ratediff|global|overall", "ci_low"], 3),
        "ci_g_hi": f(iv.loc["ratediff|global|overall", "ci_high"], 3),
        "g_clip": f(conc["abs_rate_error"]["global"], 3),
        "p_clip": f(conc["abs_rate_error"]["poisson"], 3),
    }


def build() -> Builder:
    """Defina as celulas do notebook de avaliacao final."""
    b = Builder()
    b.md("""# Avaliação final: frequência de sinistros por apólice

**O problema.** Uma seguradora de automóveis precisa estimar quantos sinistros esperar por 100 apólices-ano em perfis diferentes de veículo, condutor e região, e a duração da cobertura varia de poucos dias a mais de um ano. **Por que a calibração por segmento importa.** Um modelo pode acertar o total e, ainda assim, errar sistematicamente em grupos específicos, por exemplo nas coberturas muito curtas; um erro agregado pequeno pode esconder desvios grandes e de sinais opostos entre segmentos. **A pergunta.** No benchmark francês `freMTPL2freq`, o Poisson com offset de exposição fixo, congelado no treino, supera a taxa global de referência em registros reservados que nunca foram usados, e onde as taxas previstas se afastam das observadas por faixa de exposição, idade do veículo e região?

Este notebook apresenta a avaliação final única da partição de **teste**, executada uma vez por `python -m freq_sinistros stage5` depois de verificações de integridade. Cada seção segue a sequência pergunta → execução → insights. As conclusões são restritas ao benchmark. `production_eligible` permanece falso.""",
         """# Final evaluation: claim frequency per policy

**The problem.** A motor insurer needs to estimate how many claims to expect per 100 policy-years across different vehicle, driver and region profiles, and the coverage duration ranges from a few days to more than a year. **Why calibration by segment matters.** A model can get the total right and still be systematically wrong in specific groups, for example in very short coverages; a small aggregate error can hide large deviations of opposite signs across segments. **The question.** On the French `freMTPL2freq` benchmark, does the Poisson with a fixed exposure offset, frozen on the training data, outperform the global reference rate on held-out records never used before, and where do predicted rates depart from observed rates by exposure band, vehicle age and region?

This notebook presents the single final evaluation of the **test** partition, run once by `python -m freq_sinistros stage5` after integrity checks. Each section follows the sequence question → execution → insights. Conclusions are restricted to the benchmark. `production_eligible` remains false.""")
    b.code('''%matplotlib inline
import pandas as pd
from IPython.display import display

from freq_sinistros import artifacts, plots

LANG = "__LANG__"
ROOT = artifacts.project_root()
pd.options.display.float_format = "{:,.4f}".format
pd.options.display.max_columns = 30
out5 = artifacts.load_stage5(ROOT)''')
    b.md("""## 1. Origem e integridade

**Pergunta.** A avaliação usou o modelo, o roteiro e o código aprovados, com a população de teste prevista, e os agregados apresentados são os da execução oficial?""",
         """## 1. Origin and integrity

**Question.** Did the evaluation use the approved model, script and code, with the expected test population, and are the aggregates shown those of the official run?""")
    b.code('''display(pd.Series(artifacts.verify_stage5(ROOT)["checks"], name="ok"))
display(artifacts.gate_checks(out5))
display(artifacts.run_summary(out5))''')
    b.md("""### Insights

O portão de integridade foi satisfeito antes de as linhas de teste serem disponibilizadas ao avaliador (a verificação prévia do congelamento usa o leitor do ARFF integral, mas devolve somente a validação): os SHA-256 do modelo congelado e do roteiro conferem com os valores aprovados, o ARFF, as atribuições e os arquivos de origem conferem com o modelo, e os módulos de código conferem com os registrados. A execução oficial avaliou {n_test} registros de teste ({exp_test} apólices-ano, {claims_test} sinistros e {zeros_test} zeros), em uma única execução protegida por trava, criada depois do portão e antes da disponibilização do teste; o número de linhas conferido se refere às linhas disponibilizadas ao avaliador, não à leitura física do ARFF. Os agregados são verificados aqui sem reler o ARFF, e nenhuma linha de teste é persistida.""",
         """### Insights

The integrity gate was satisfied before the test rows were made available to the evaluator (the prior freeze check uses the whole-ARFF reader but returns only the validation partition): the SHA-256 of the frozen model and of the script match the approved values, the ARFF, the assignments and the source files match the model, and the code modules match those recorded. The official run evaluated {n_test} test records ({exp_test} policy-years, {claims_test} claims and {zeros_test} zeros), in a single run protected by a lock, created after the gate and before the test rows were made available; the row count checked refers to the rows made available to the evaluator, not to the physical reading of the ARFF. The aggregates are verified here without rereading the ARFF, and no test row is persisted.""")
    b.md("""## 2. Comparação primária

**Pergunta.** O Poisson congelado tem desvio médio de Poisson menor que a taxa global nas mesmas linhas de teste, com contagens e exposições originais? A diferença pareada Poisson − taxa global usa 1.000 réplicas bootstrap por `IDpol` (semente 20261121) e, como sensibilidade, por chave de perfil.""",
         """## 2. Primary comparison

**Question.** Does the frozen Poisson have a lower mean Poisson deviance than the global rate on the same test rows, with original counts and exposures? The paired Poisson − global rate difference uses 1,000 bootstrap replicates by `IDpol` (seed 20261121) and, as a sensitivity, by profile key.""")
    b.code('''display(artifacts.metrics_view(out5, LANG))
plots.show(plots.plot_paired_difference(out5["test_intervals"], LANG))''')
    b.md("""### Insights

O desvio médio é {dev_p} no Poisson e {dev_g} na taxa global. A diferença pareada é {d} (IC 95% {lo} a {hi}; valores negativos favorecem o Poisson), e o bootstrap por perfil dá {c_lo} a {c_hi}. Os dois intervalos ficam inteiramente abaixo de zero, de modo que o teste {verdict_pt} do Poisson sobre a taxa global na métrica primária. Os intervalos são condicionados aos modelos ajustados e não incluem a variabilidade do ajuste. No total, a taxa observada é {obs_rate} por 100 apólices-ano; a prevista é {pred_p} (Poisson) e {pred_g} (taxa global), com diferenças de {rd_p} e {rd_g}. Cada intervalo de 95% se refere à diferença prevista − observada de um candidato: de {ci_p_lo} a {ci_p_hi} no Poisson e de {ci_g_lo} a {ci_g_hi} na taxa global, e ambos incluem zero. Esses intervalos individuais não testam equivalência entre os candidatos nem comparam seus erros absolutos, cujos valores pontuais são {p_clip} (Poisson) e {g_clip} (taxa global). O ganho confirmado é no desvio médio.""",
         """### Insights

The mean deviance is {dev_p} for Poisson and {dev_g} for the global rate. The paired difference is {d} (95% CI {lo} to {hi}; negative values favor Poisson), and the profile bootstrap gives {c_lo} to {c_hi}. Both intervals lie entirely below zero, so the test {verdict_en} of Poisson over the global rate on the primary metric. The intervals are conditional on the fitted models and do not include the variability of the fit. Overall, the observed rate is {obs_rate} per 100 policy-years; the predicted rate is {pred_p} (Poisson) and {pred_g} (global rate), with differences of {rd_p} and {rd_g}. Each 95% interval refers to the predicted − observed difference of one candidate: {ci_p_lo} to {ci_p_hi} for Poisson and {ci_g_lo} to {ci_g_hi} for the global rate, and both include zero. These individual intervals do not test equivalence between the candidates nor compare their absolute errors, whose point values are {p_clip} (Poisson) and {g_clip} (global rate). The confirmed gain is in the mean deviance.""")
    b.md("""## 3. Calibração por faixa de exposição

**Pergunta.** Como as taxas previstas se comparam às observadas por faixa de exposição, e o comportamento observado em validação se repete no teste? As diferenças são prevista − observada, por 100 apólices-ano, calculadas com a exposição total de cada faixa e as previsões diretas; a faixa `(0; 0,02]` permanece visível e segmentos com menos de 500 registros ou 25 apólices-ano são sinalizados.""",
         """## 3. Calibration by exposure band

**Question.** How do predicted rates compare with observed rates by exposure band, and does the behavior observed in validation repeat on the test? Differences are predicted − observed, per 100 policy-years, computed with the total exposure of each band and direct predictions; the `(0; 0.02]` band stays visible and segments with fewer than 500 records or 25 policy-years are flagged.""")
    b.code('''display(artifacts.panel_view(out5, "exposure_band", "poisson", LANG))
plots.show(plots.plot_panel(out5["panel"], LANG, "exposure_band"))
display(out5["validation_vs_test_bands"])
plots.show(plots.plot_validation_vs_test(out5["validation_vs_test_bands"],
                                         LANG))''')
    b.md("""### Insights

O desvio nas exposições curtas persiste no teste. Em `(0; 0,02]`, a taxa observada é {obs1} por 100 apólices-ano e a prevista pelo Poisson, {pred1}, uma diferença de {diff1} (IC {lo1} a {hi1}); em validação, a diferença foi {val1}. Em exatamente um ano, o observado é {obs_one} e o previsto, {pred_one} (diferença {diff_one}). O sinal da diferença coincide com o da validação em {n_same} das {n_bands} faixas, e a diferença de validação cai dentro do intervalo de teste em {n_inside} delas; as faixas {outside} têm a diferença de validação fora do intervalo de teste. O offset fixo, portanto, subestima as exposições muito curtas e superestima as longas também no teste, e esse é um limite do modelo selecionado, que não foi recalibrado.""",
         """### Insights

The bias in short exposures persists on the test. In `(0; 0.02]`, the observed rate is {obs1} per 100 policy-years and the Poisson-predicted rate is {pred1}, a difference of {diff1} (CI {lo1} to {hi1}); in validation, the difference was {val1}. At exactly one year, the observed rate is {obs_one} and the predicted rate {pred_one} (difference {diff_one}). The sign of the difference matches that of validation in {n_same} of {n_bands} bands, and the validation difference falls inside the test interval in {n_inside} of them; the bands {outside} have the validation difference outside the test interval. The fixed offset, therefore, underestimates very short exposures and overestimates long ones on the test as well, and this is a limit of the selected model, which was not recalibrated.""")
    b.md("""## 4. Calibração por idade do veículo e região

**Pergunta.** Há desvios por idade do veículo ou região que os intervalos distinguem de zero? Segmentos de baixo suporte são mostrados sem intervalo.""",
         """## 4. Calibration by vehicle age and region

**Question.** Are there deviations by vehicle age or region that the intervals distinguish from zero? Low-support segments are shown without an interval.""")
    b.code('''for family in ("vehicle_age", "region"):
    display(artifacts.panel_view(out5, family, "poisson", LANG))
    plots.show(plots.plot_panel(out5["panel"], LANG, family))''')
    b.md("""### Insights

Por idade do veículo, os intervalos distinguem de zero apenas os segmentos {veh_hit}. Por região, apenas {reg_hit}; as regiões {low_reg} têm baixo suporte e não têm intervalo. Os demais segmentos têm intervalos que incluem zero. Como são muitas comparações simultâneas sem ajuste de multiplicidade, esses desvios são descritivos e não indicam, por si, falha de especificação.""",
         """### Insights

By vehicle age, the intervals distinguish from zero only the segments {veh_hit}. By region, only {reg_hit}; the regions {low_reg} have low support and no interval. The remaining segments have intervals that include zero. Since these are many simultaneous comparisons without multiplicity adjustment, these deviations are descriptive and do not by themselves indicate a specification failure.""")
    b.md("""## 5. Reconciliação

**Pergunta.** Cada segmentação soma os registros, a exposição e as contagens observadas e previstas ao total de cada candidato?""",
         """## 5. Reconciliation

**Question.** Does each segmentation add up records, exposure and observed and predicted counts to the total of each candidate?""")
    b.code('''display(out5["test_reconciliation"])''')
    b.md("""### Insights

As {n_checks} verificações (dois candidatos, três segmentações e quatro indicadores) foram satisfeitas: a soma dos segmentos coincide com os {n_test} registros, as {exp_test} apólices-ano, os {claims_test} sinistros observados e os previstos de cada candidato.""",
         """### Insights

The {n_checks} checks (two candidates, three segmentations and four indicators) were satisfied: the sum of the segments coincides with the {n_test} records, the {exp_test} policy-years, the {claims_test} observed claims and the predicted claims of each candidate.""")
    b.md("""## 6. Conclusão e limites

**Pergunta.** O que o teste sustenta, e o que não sustenta?""",
         """## 6. Conclusion and limits

**Question.** What does the test support, and what does it not support?""")
    b.code('''display(pd.Series(out5["conclusion"]["primary"], name="primary"))
display(pd.Series(out5["conclusion"]["profile_cluster_bootstrap"],
                  name="profile_cluster_bootstrap"))''')
    b.md("""### Insights

O teste sustenta um ganho do Poisson com offset fixo sobre a taxa global em desvio médio de Poisson, e a escolha congelada é preservada. O ganho confirmado é no desvio médio; ele não elimina o desvio sistemático nas exposições curtas. Na calibração total, o erro absoluto pontual do Poisson é maior que o da referência, mas os intervalos individuais não permitem concluir equivalência nem diferença entre os candidatos. Os resultados valem dentro do benchmark francês, sem evidência temporal, sem `BonusMalus` (instante de medição desconhecido, Dutang & Charpentier, 2018) e sem modelos inflacionados de zeros; os intervalos são condicionados aos modelos ajustados. O modelo não é elegível para produção.""",
         """### Insights

The test supports a gain of the fixed-offset Poisson over the global rate in mean Poisson deviance, and the frozen choice is preserved. The confirmed gain is in the mean deviance; it does not remove the systematic bias in short exposures. In total calibration, the point absolute error of Poisson is larger than the reference's, but the individual intervals do not allow concluding either equivalence or a difference between the candidates. The results hold within the French benchmark, with no temporal evidence, without `BonusMalus` (unknown measurement time, Dutang & Charpentier, 2018) and without zero-inflated models; intervals are conditional on the fitted models. The model is not eligible for production.""")
    return b


REFS = ["dutang2018"]


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
        path = ROOT / "notebooks" / lang / "04_avaliacao_final.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, path)
        print("escrito", path)


if __name__ == "__main__":
    main()
