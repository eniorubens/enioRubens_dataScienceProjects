"""Construa 01_eda_treino.ipynb (PT-BR e EN-US) a partir de um unico roteiro.

Os tamanhos das particoes e os resultados da EDA sao lidos de
``outputs/etapa_02`` e ``outputs/etapa_03`` durante a construcao. A EDA
e recalculada apenas com o treino quando o notebook e executado.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from notebook_common import EN, PT, Builder, num, pct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "etapa_03"


def values(lang: str) -> dict:
    """Numeros citados na narrativa, formatados para o idioma."""
    s = json.loads((OUT / "eda_summary.json").read_text("utf-8"))
    partition_path = ROOT / "outputs" / "etapa_02" / "partition_config.json"
    partition = json.loads(partition_path.read_text("utf-8"))
    sizes = {row["partition"]: row["ids"] for row in partition["sizes"]}
    dist = pd.read_csv(OUT / "eda_claimnb_distribution.csv").set_index(
        "claimnb")
    bands = pd.read_csv(OUT / "eda_exposure_bands.csv").set_index("band")
    veh = pd.read_csv(OUT / "eda_rate_vehicle_age.csv").set_index("segment")
    drv = pd.read_csv(OUT / "eda_rate_driver_age.csv").set_index("segment")
    reg = pd.read_csv(OUT / "eda_rate_region.csv")
    reg_ok = reg[~reg["low_support"]]
    lo, hi = reg_ok.loc[reg_ok["rate_per_100"].idxmin()], reg_ok.loc[
        reg_ok["rate_per_100"].idxmax()]
    sb = pd.read_csv(OUT / "eda_short_exposure_bands.csv").set_index("band")
    sc, off = s["short_exposure_check"], s["offset_check"]
    n = s["train_rows"]
    total = bands.loc["total"]
    one = bands.loc["= 1"]
    f = lambda x, d=2: num(x, d, lang)  # noqa: E731
    p = lambda x, d=1: pct(x, d, lang)  # noqa: E731
    return {
        "rows": f(n, 0), "exposure": f(s["train_exposure"]),
        "dataset_rows": f(sum(sizes.values()), 0),
        "validation_rows": f(sizes["validation"], 0),
        "test_rows": f(sizes["test"], 0),
        "claims": f(s["train_claims"], 0),
        "zeros_pct": p(s["train_zeros"] / n, 2),
        "rate": f(100 * s["train_claims"] / s["train_exposure"], 2),
        "c1_claims": p(dist.loc[1, "share_of_claims"], 1),
        "c2_claims": p(dist.loc[2, "share_of_claims"], 1),
        "tail_rows": f(s["extremes"]["claimnb_gt_4_rows"], 0),
        "tail_claims": f(s["extremes"]["claimnb_gt_4_claims"], 0),
        "one_rows": p(one["rows"] / total["rows"], 1),
        "one_exposure": p(one["exposure_valid"] / total["exposure_valid"], 1),
        "one_rate": f(one["rate_per_100_policy_years"], 2),
        "short_rows": f(bands.loc["(0, 0.02]", "rows"], 0),
        "short_rate": f(bands.loc["(0, 0.02]", "rate_per_100_policy_years"],
                        1),
        "gt1_rows": f(bands.loc["> 1", "rows"], 0),
        "gt1_rate": f(bands.loc["> 1", "rate_per_100_policy_years"], 2),
        "va01": f(veh.loc["0-1", "rate_per_100"], 1),
        "va1120": f(veh.loc["11-20", "rate_per_100"], 1),
        "da1824": f(drv.loc["18-24", "rate_per_100"], 1),
        "da3544": f(drv.loc["35-44", "rate_per_100"], 1),
        "reg_lo": f"{lo['segment']} ({f(lo['rate_per_100'], 1)})",
        "reg_hi": f"{hi['segment']} ({f(hi['rate_per_100'], 1)})",
        "oe_002": f(sb.loc["(0, 0.02]", "ratio_observed_expected"], 1),
        "oe_008": f(sb.loc["(0.02, 0.08]", "ratio_observed_expected"], 1),
        "oe_one": f(sb.loc["= 1", "ratio_observed_expected"], 2),
        "n_strata": f(sc["strata_supported_in_both"], 0),
        "n_above": f(sc["strata_short_rate_above_reference"], 0),
        "med_ratio": f(sc["median_ratio_short_over_ref"], 1),
        "min_ratio": f(sc["min_ratio_short_over_ref"], 1),
        "off_coef": f(off["coefficient_log_exposure"], 2),
        "off_lo": f(off["approx_95_low"], 2),
        "off_hi": f(off["approx_95_high"], 2),
        "unk_region": f(s["unknown_level_check"][
            "unknown_levels_heldout"]["Region"], 0),
        "unk_power": f(s["unknown_level_check"][
            "unknown_levels_heldout"]["VehPower"], 0),
        "heldout": f(s["unknown_level_check"]["heldout_train_rows"], 0),
    }


def build() -> Builder:
    """Defina as celulas do notebook de EDA."""
    b = Builder()
    b.md(
        "# EDA do treino: frequência de sinistros por apólice\n\n"
        "**O problema.** Quantos sinistros são esperados para diferentes "
        "perfis de veículo, condutor e região, considerando o tempo de "
        "cobertura? Essa pergunta é estudada com registros de uma carteira "
        "francesa de seguros de automóveis (Dutang & Charpentier, 2018).\n\n"
        "**Por que importa.** Cada apólice coberta por um ano contribui "
        "com uma apólice-ano de exposição. Dois sinistros observados em "
        "20 apólices-ano correspondem a 10 por 100 apólices-ano. Se os "
        "mesmos dois sinistros forem observados em 100 apólices-ano, a "
        "taxa será de 2 por 100. Assim, grupos com tempos de cobertura "
        "diferentes podem ser comparados na mesma escala. Mesmo quando a "
        "taxa prevista para a carteira está próxima da observada, as "
        "taxas de alguns segmentos podem ser subestimadas ou "
        "superestimadas.\n\n"
        "**O papel deste notebook.** Na partição de **treino**, são "
        "examinadas as contagens de sinistros (`ClaimNb`), a duração "
        "da cobertura (`Exposure`) e as taxas observadas por perfil e "
        "faixa de exposição. Cada taxa é calculada como "
        "`100 × Σ ClaimNb / Σ Exposure`. Essa análise prepara a "
        "comparação posterior entre as taxas observadas e previstas "
        "por segmento.",
        "# Training EDA: claim frequency per policy\n\n"
        "**The problem.** How many claims are expected for different "
        "vehicle, driver and regional profiles, given the length of "
        "coverage? This question is studied using records from a French "
        "motor insurance portfolio (Dutang & Charpentier, 2018).\n\n"
        "**Why it matters.** A policy covered for one year contributes "
        "one policy-year of exposure. Two claims observed over "
        "20 policy-years correspond to 10 per 100 policy-years. If the "
        "same two claims are observed over 100 policy-years, the rate "
        "is 2 per 100. Groups with different coverage durations can "
        "therefore be compared on the same scale. Even when the "
        "predicted portfolio rate is close to the observed rate, some "
        "segments can still be underpredicted or overpredicted.\n\n"
        "**The role of this notebook.** Claim counts (`ClaimNb`), "
        "coverage duration (`Exposure`), and observed rates by profile "
        "and exposure band are examined in the **training** partition. "
        "Each rate is calculated as `100 × Σ ClaimNb / Σ Exposure`. "
        "This analysis prepares the later comparison of observed and "
        "predicted rates by segment.",
    )
    b.code('''%matplotlib inline
import pandas as pd
from IPython.display import display

from freq_sinistros import artifacts, eda, plots
from freq_sinistros.data import load_partitions

LANG = "__LANG__"
ROOT = artifacts.project_root()
pd.options.display.float_format = "{:,.4f}".format
pd.options.display.max_columns = 20''')
    b.md("""## 1. Origem, unidade e partição

**Pergunta.** Qual arquivo, qual unidade de observação e qual partição sustentam esta análise?""",
         """## 1. Origin, unit and partition

**Question.** Which file, which observation unit and which partition support this analysis?""")
    b.code('''parts, provenance = load_partitions(ROOT, keep=("train",))
train = parts["train"]
display(artifacts.provenance_table(provenance, len(train), LANG))''')
    b.md("""### Insights

Foi usado o arquivo bruto do OpenML 41214, versão 1 (`freMTPL2freq`), conferido por SHA-256 contra o manifesto de aquisição. A auditoria confirmou {dataset_rows} valores distintos de `IDpol` e nenhum valor faltante. Cada linha representa um registro de cobertura, com duração medida em anos (Dutang & Charpentier, 2018).

Na etapa 2 do projeto, os registros foram divididos em aproximadamente **70% para treino, 15% para validação e 15% para teste**. A atribuição foi feita de forma determinística pelo hash SHA-256 de `IDpol`, independentemente da ordem das linhas. Foi verificado que cada identificador aparece em uma única partição e que todos foram atribuídos.

| Partição | Registros | Uso |
| --- | ---: | --- |
| Treino | {rows} | EDA, pré-processamento e ajuste dos modelos |
| Validação | {validation_rows} | Comparação e escolha do modelo |
| Teste | {test_rows} | Uma única avaliação após o congelamento do modelo |

**Por que a EDA foi feita depois da partição?** A auditoria de integridade foi realizada antes da divisão. A partição foi definida antes da EDA para que a exploração, a comparação de modelos e a avaliação final fossem conduzidas em conjuntos distintos (Wickham & Grolemund, 2017). A relação entre exposição e frequência de sinistros, por exemplo, foi explorada no treino. O teste foi reservado até o congelamento do modelo para evitar que seus resultados influenciassem decisões de modelagem e levassem a sobreajuste à própria amostra de teste (Dwork et al., 2015).

O leitor interpreta o ARFF integral em memória. Para a EDA deste notebook, são disponibilizadas apenas as linhas de treino.""",
         """### Insights

The raw file from OpenML 41214, version 1 (`freMTPL2freq`), was checked by SHA-256 against the acquisition manifest. The audit confirmed {dataset_rows} distinct `IDpol` values and no missing values. Each row represents a coverage record, with duration measured in years (Dutang & Charpentier, 2018).

In stage 2 of the project, the records were split approximately **70% for training, 15% for validation, and 15% for testing**. Assignments were made deterministically using a SHA-256 hash of `IDpol`, independently of row order. Every identifier was verified to appear in exactly one partition.

| Partition | Records | Use |
| --- | ---: | --- |
| Training | {rows} | EDA, preprocessing, and model fitting |
| Validation | {validation_rows} | Model comparison and selection |
| Test | {test_rows} | A single evaluation after the model was frozen |

**Why was EDA performed after splitting?** The data integrity audit was performed before the split. The data were split before EDA so that exploration, model comparison, and final evaluation could be carried out on separate sets (Wickham & Grolemund, 2017). The relationship between exposure and claim frequency, for example, was explored on the training set. The test set was held back until the model was frozen so that test results could not influence modeling decisions and lead to overfitting to the test sample itself (Dwork et al., 2015).

The reader parses the whole ARFF in memory. Only the training rows are made available to the EDA in this notebook.""")
    b.md("""## 2. Dicionário das variáveis

**Pergunta.** Quais são as 12 variáveis do `freMTPL2freq`, e como são diferenciados o alvo, a exposição, o identificador e as covariáveis?""",
         """## 2. Variable dictionary

**Question.** What are the 12 variables in `freMTPL2freq`, and how are the target, exposure, identifier and covariates distinguished?""")
    b.code('''display(artifacts.variable_dictionary_table(train.columns, LANG))''')
    b.md("""### Insights

Foi apresentado o dicionário das 12 variáveis na ordem do conjunto de dados. `ClaimNb` foi identificado como o alvo de contagem, `Exposure` como a exposição em anos usada no offset, e `IDpol` como identificador destinado à rastreabilidade, não à predição. As nove variáveis restantes foram classificadas como covariáveis do veículo, do condutor ou territoriais. A disponibilidade das covariáveis na estimação foi tratada como hipótese porque a fonte não informa a data de referência; para `BonusMalus`, o instante de medição permanece indeterminado e a variável não integra a especificação principal.""",
         """### Insights

The dictionary of all 12 variables was presented in data-set order. `ClaimNb` was identified as the count target, `Exposure` as exposure in years used in the offset, and `IDpol` as an identifier for traceability rather than prediction. The remaining nine variables were classified as vehicle, driver or territorial covariates. Covariate availability at estimation was treated as a hypothesis because the source does not report the reference date; for `BonusMalus`, the measurement time remains indeterminate and the variable is not included in the main specification.""")
    b.md("""## 3. Cálculo da EDA

**Pergunta.** Quais tabelas descritivas são obtidas do treino? A execução abaixo calcula todas elas de uma vez, incluindo a verificação das exposições curtas, o diagnóstico do offset e a cobertura de categorias.""",
         """## 3. EDA computation

**Question.** Which descriptive tables are obtained from the training data? The execution below computes all of them at once, including the short-exposure check, the offset diagnostic and the category coverage.""")
    b.code('''tables, summary = eda.compute_eda(train)
sorted(tables)''')
    b.md("""### Insights

Foram geradas nove tabelas e um resumo, todos derivados apenas das {rows} linhas de treino ({exposure} apólices-ano e {claims} sinistros; taxa de {rate} por 100 apólices-ano).""",
         """### Insights

Nine tables and one summary were generated, all derived only from the {rows} training rows ({exposure} policy-years and {claims} claims; rate of {rate} per 100 policy-years).""")
    b.md("""## 4. Distribuição de `ClaimNb`

**Pergunta.** Qual é a proporção de zeros e quão pesada é a cauda da contagem?""",
         """## 4. Distribution of `ClaimNb`

**Question.** What is the proportion of zeros and how heavy is the tail of the count?""")
    b.code('''display(artifacts.localize(tables["eda_claimnb_distribution"], LANG))
plots.show(plots.plot_claimnb(tables["eda_claimnb_distribution"], LANG))
display(artifacts.extremes_table(summary, LANG))''')
    b.md("""### Insights

Entre os registros de treino, {zeros_pct} têm `ClaimNb = 0`. As outras proporções usam um denominador diferente: do total de sinistros observado, {c1_claims} estão em apólices com um sinistro e {c2_claims}, em apólices com dois sinistros.

A cauda acima de quatro é rara: {tail_rows} registros somam {tail_claims} sinistros. Esses casos são válidos e foram preservados. Sua concentração em poucos perfis é tratada na sensibilidade por perfil da comparação de modelos.""",
         """### Insights

Among the training records, {zeros_pct} have `ClaimNb = 0`. The other percentages use a different denominator: of all observed claims, {c1_claims} are in policies with one claim and {c2_claims} in policies with two claims.

The tail above four is rare: {tail_rows} records add up to {tail_claims} claims. These cases are valid and were preserved. Their concentration in a small number of profiles is handled in the profile-based sensitivity of the model comparison.""")
    b.md("""## 5. Distribuição de `Exposure`

**Pergunta.** Como a duração da cobertura se distribui, e quanto pesam as exposições muito curtas, exatamente um ano e acima de um ano?""",
         """## 5. Distribution of `Exposure`

**Question.** How is the coverage duration distributed, and how much do very short exposures, exactly one year and more than one year weigh?""")
    b.code('''display(artifacts.localize(tables["eda_exposure_bands"], LANG))
plots.show(plots.plot_exposure(tables["eda_exposure_histogram"], LANG))''')
    b.md("""### Insights

Exposições de exatamente um ano são {one_rows} dos registros e {one_exposure} da exposição total. A faixa `(0; 0,02]` (cerca de uma semana) tem {short_rows} registros e a faixa acima de um ano, {gt1_rows}. A unidade é anos, conforme o dicionário (Dutang & Charpentier, 2018). Exposições acima de um ano são tratadas como cauda observada, não como erro, pois a fonte descreve as apólices como observadas predominantemente por um ano.""",
         """### Insights

Exposures of exactly one year are {one_rows} of the records and {one_exposure} of total exposure. The `(0; 0.02]` band (about one week) has {short_rows} records and the band above one year has {gt1_rows}. The unit is years, according to the dictionary (Dutang & Charpentier, 2018). Exposures above one year are handled as an observed tail, not as errors, because the source describes the policies as observed mostly over one year.""")
    b.md("""## 6. Frequência por faixa de exposição, idade e região

**Pergunta.** Como a taxa por 100 apólices-ano varia por faixa de exposição, idade do veículo, idade do condutor e região? Segmentos com menos de 500 registros ou menos de 25 apólices-ano são sinalizados como de baixo suporte.""",
         """## 6. Frequency by exposure band, age and region

**Question.** How does the rate per 100 policy-years vary by exposure band, vehicle age, driver age and region? Segments with fewer than 500 records or fewer than 25 policy-years are flagged as low support.""")
    b.code('''bands = eda.band_rate_table(tables["eda_exposure_bands"])
plots.show(plots.plot_rates(bands, LANG, "band"))
for name, key in (("age_veh", "eda_rate_vehicle_age"),
                  ("age_drv", "eda_rate_driver_age"),
                  ("region", "eda_rate_region")):
    table = tables[key]
    display(artifacts.localize(table, LANG))
    plots.show(plots.plot_rates(table, LANG, name))''')
    b.md("""### Insights

A taxa decresce com a duração da cobertura: {short_rate} por 100 apólices-ano em `(0; 0,02]`, {one_rate} em exatamente um ano e {gt1_rate} acima de um ano. Por idade do veículo, a taxa é {va01} em 0–1 ano e {va1120} em 11–20 anos; por idade do condutor, {da1824} em 18–24 e {da3544} em 35–44. Entre as regiões com suporte, as taxas vão de {reg_lo} a {reg_hi}. Nenhum segmento do treino é de baixo suporte nas faixas de exposição e de idade; as regiões de baixo suporte, quando existem, aparecem hachuradas. Todas as diferenças são marginais e descritivas.""",
         """### Insights

The rate decreases with coverage duration: {short_rate} per 100 policy-years in `(0; 0.02]`, {one_rate} at exactly one year and {gt1_rate} above one year. By vehicle age, the rate is {va01} at 0–1 years and {va1120} at 11–20 years; by driver age, {da1824} at 18–24 and {da3544} at 35–44. Among regions with support, rates range from {reg_lo} to {reg_hi}. No training segment is low support in the exposure and age bands; low-support regions, when present, appear hatched. All differences are marginal and descriptive.""")
    b.md("""## 7. As exposições curtas persistem dentro de perfis?

**Pergunta.** A taxa elevada das exposições curtas decorre da composição por idade ou persiste dentro de estratos com suporte? A taxa de referência de cada estrato (idade do condutor × idade do veículo) usa as linhas com `Exposure ≥ 0,75`.""",
         """## 7. Do short exposures persist within profiles?

**Question.** Is the elevated rate of short exposures due to the age composition, or does it persist within strata with support? The reference rate of each stratum (driver age × vehicle age) uses the rows with `Exposure ≥ 0.75`.""")
    b.code('''display(artifacts.localize(tables["eda_short_exposure_bands"], LANG))
plots.show(plots.plot_short_check(tables["eda_short_exposure_bands"], LANG))
display(pd.Series(summary["short_exposure_check"]))
display(pd.Series(summary["offset_check"]))''')
    b.md("""### Insights

Em {n_above} de {n_strata} estratos com suporte nos dois grupos, a taxa das linhas com `Exposure ≤ 0,08` supera a de referência (razão mediana {med_ratio}; mínima {min_ratio}). A razão observado/esperado é {oe_002} em `(0; 0,02]`, {oe_008} em `(0,02; 0,08]` e {oe_one} em exatamente um ano. O padrão, portanto, persiste dentro dos estratos de idade com suporte.

Essa comparação é descritiva. Os estratos controlam apenas a idade do condutor e a do veículo, não as demais covariáveis; além disso, a referência (`Exposure ≥ 0,75`) é uma escolha analítica. O resultado não exclui outras explicações nem indica causalidade.

O offset `log(Exposure)` com coeficiente 1 pressupõe sinistros proporcionais à exposição (Cameron & Trivedi, 2013). Como diagnóstico no treino, o coeficiente livre de `log(Exposure)` foi {off_coef} (intervalo aproximado de {off_lo} a {off_hi}), distante de 1. Na validação, será conferida a calibração por faixa de exposição dos três candidatos, sem recalibração pós-hoc. Se o padrão persistir, as faixas curtas serão subestimadas e as longas superestimadas.""",
         """### Insights

In {n_above} of {n_strata} strata with support in both groups, the rate of rows with `Exposure ≤ 0.08` exceeds the reference rate (median ratio {med_ratio}; minimum {min_ratio}). The observed/expected ratio is {oe_002} in `(0; 0.02]`, {oe_008} in `(0.02; 0.08]` and {oe_one} at exactly one year. The pattern therefore persists within the age strata with support.

This comparison is descriptive. The strata control only driver age and vehicle age, not the other covariates; moreover, the reference (`Exposure ≥ 0.75`) is an analytical choice. The result neither rules out other explanations nor indicates causality.

The `log(Exposure)` offset with coefficient 1 presupposes claims proportional to exposure (Cameron & Trivedi, 2013). As a training diagnostic, the free coefficient of `log(Exposure)` was {off_coef} (approximate interval {off_lo} to {off_hi}), far from 1. In validation, the calibration by exposure band of the three candidates will be checked without post hoc recalibration. If the pattern persists, short bands will be underpredicted and long bands overpredicted.""")
    b.md("""## 8. Cobertura de categorias e níveis desconhecidos

**Pergunta.** Como o pré-processamento trata níveis raros ou ausentes do ajuste? Um pré-processador foi ajustado em uma amostra de 100 mil linhas de treino e aplicado às demais linhas de treino.""",
         """## 8. Category coverage and unknown levels

**Question.** How does the preprocessing handle levels that are rare or absent from the fit? A preprocessor was fitted on a sample of 100,000 training rows and applied to the remaining training rows.""")
    b.code('''display(artifacts.coverage_table(summary, LANG))
display(pd.Series(summary["unknown_level_check"]["has_other"]))''')
    b.md("""### Insights

Níveis com menos de 500 linhas de ajuste são agrupados em `OTHER`. Entre as {heldout} linhas de treino fora da amostra, {unk_region} (`Region`) e {unk_power} (`VehPower`) foram mapeadas para `OTHER`, e a matriz de desenho permaneceu finita. Isso verifica o tratamento de níveis desconhecidos com dados de treino; a regra é aplicada a validação e teste sem reajuste.""",
         """### Insights

Levels with fewer than 500 fit rows are pooled into `OTHER`. Among the {heldout} training rows outside the sample, {unk_region} (`Region`) and {unk_power} (`VehPower`) were mapped to `OTHER`, and the design matrix remained finite. This verifies the handling of unknown levels with training data; the rule is applied to validation and test without refitting.""")
    b.md("""## 9. Limites da evidência

**Pergunta.** O que esta EDA não permite concluir?""",
         """## 9. Limits of the evidence

**Question.** What does this EDA not allow concluding?""")
    b.code('''pd.Series({"partition": "train", "rows": len(train),
           "has_dates": False, "has_higher_unit_id": False})''')
    b.md("""### Insights

O esquema não traz datas nem identificador de unidade superior; a disponibilidade das covariáveis na data de estimação é uma hipótese, e `BonusMalus`, cujo instante de medição é desconhecido, fica fora da especificação principal (Dutang & Charpentier, 2018). As taxas por segmento são marginais e não controlam as demais covariáveis. A evidência vale dentro do benchmark e é descritiva. O mesmo conjunto de dados foi estudado em outros trabalhos (Noll et al., 2020); a presente análise não os reproduz. A sequência de modelagem segue a literatura de modelos de contagem (Fávero & Belfiore, 2017).""",
         """### Insights

The schema has neither dates nor a higher-level unit identifier; covariate availability at the estimation date is a hypothesis, and `BonusMalus`, whose measurement time is unknown, is left out of the main specification (Dutang & Charpentier, 2018). Segment rates are marginal and do not control for the other covariates. The evidence holds within the benchmark and is descriptive. The same data set has been studied in other work (Noll et al., 2020); the present analysis does not reproduce it. The modeling sequence follows the count-model literature (Fávero & Belfiore, 2017).""")
    return b


REFS = ["cameron2013", "dutang2018", "dwork2015", "favero2017", "noll2020",
        "wickham2017"]


def main() -> None:
    """Grave os dois notebooks."""
    builder = build()
    for lang in (PT, EN):
        nb = builder.build(lang, REFS)
        v = values(lang)
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.format_map(_Safe(v))
        path = ROOT / "notebooks" / lang / "01_eda_treino.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, path)
        print("escrito", path)


class _Safe(dict):
    """Preserve chaves desconhecidas e blocos de codigo entre chaves."""

    def __missing__(self, key: str) -> str:
        """Devolva a propria chave entre chaves."""
        return "{" + key + "}"


if __name__ == "__main__":
    main()
