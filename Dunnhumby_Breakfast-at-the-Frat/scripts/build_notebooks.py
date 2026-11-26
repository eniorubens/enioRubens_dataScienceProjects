"""Build (and optionally execute) the bilingual notebooks from one specification.

Every notebook is described once, as a list of cells. Markdown cells carry both languages
(``md(pt=..., en=...)``); code cells are shared verbatim (``code(...)``) and ``lang_cell()``
emits the single ``LANG = "pt"/"en"`` line. The PT/EN code parity required by
``retail_multilevel.nbcontract`` therefore holds by construction.

Usage::

    python scripts/build_notebooks.py [--only 01] [--execute]

To add a notebook, write a function returning its cells and append a :class:`NotebookSpec`
to :data:`NOTEBOOKS`.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from retail_multilevel.references import REFERENCES, render_references

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
LANG_DIRS = {"pt": "pt_br", "en": "en_us"}
CELL_TIMEOUT = 600


@dataclass(frozen=True)
class Cell:
    """One notebook cell; ``kind`` is ``md``, ``code`` or ``lang``."""

    kind: str
    pt: str = ""
    en: str = ""
    source: str = ""


def md(pt: str, en: str) -> Cell:
    """Markdown cell with one text per language."""
    return Cell("md", pt=pt.strip("\n"), en=en.strip("\n"))


def code(source: str) -> Cell:
    """Code cell shared verbatim by both languages."""
    return Cell("code", source=source.strip("\n"))


def lang_cell() -> Cell:
    """Code cell holding the only language-dependent line, ``LANG = "pt"`` or ``"en"``."""
    return Cell("lang")


@dataclass(frozen=True)
class NotebookSpec:
    """A notebook: number prefix, file name per language and its cells."""

    number: str
    pt_name: str
    en_name: str
    cells: list[Cell]


CITE_SHAPE = re.compile(r"[A-Z][\w-]+(?: (?:&|et al\.)(?: [A-Z][\w-]+)?)?, (?:19|20)\d\d\b")


def cited_keys(texts: list[str], lang: str) -> set[str]:
    """Keys of :data:`REFERENCES` whose inline citation text occurs in ``texts`` (``lang`` = 'pt' or 'en')."""
    joined = "\n".join(texts)
    return {key for key, ref in REFERENCES.items() if ref[f"cite_{lang}"] in joined}


def check_citations(texts: list[str], lang: str) -> set[str]:
    """Return the cited keys of ``texts``; fail on any author-year string that is not in the vetted list."""
    known = {ref[f"cite_{lang}"] for ref in REFERENCES.values()}
    stray = {m for m in CITE_SHAPE.findall("\n".join(texts)) if m not in known}
    if stray:
        raise ValueError(f"citations outside docs/REFERENCIAS.md: {sorted(stray)}")
    return cited_keys(texts, lang)


def with_references(cells: list[Cell]) -> list[Cell]:
    """Append the final references cell; its keys are exactly the inline citations of ``cells``, per language."""
    keys = {}
    for lang in ("pt", "en"):
        texts = [getattr(c, lang) for c in cells if c.kind == "md"]
        keys[lang] = check_citations(texts, lang)
    if keys["pt"] != keys["en"]:
        raise ValueError(f"PT and EN cite different works: {sorted(keys['pt'] ^ keys['en'])}")
    ordered = sorted(keys["pt"])
    return [*cells, md(pt=render_references(ordered, "pt"), en=render_references(ordered, "en"))]


SETUP = """
import warnings

import pandas as pd
from IPython.display import display

from retail_multilevel import eda, viz
from retail_multilevel.i18n import t
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.split import model_sample

warnings.filterwarnings("ignore", category=FutureWarning)
viz.use_retina()
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:,.3f}".format)
panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
train = model_sample(panel, "train")
test = model_sample(panel, "test")
print(t("data_credit", LANG))
"""


def eda_cells() -> list[Cell]:
    """Cells of notebook 01, the exploratory analysis of the panel."""
    return [
        md(
            pt="""
# 01 · Análise exploratória do painel

*Dados: dunnhumby Source Files – Breakfast at the Frat.* Os dados brutos são licenciados e não são redistribuídos \
com o repositório; apenas resultados agregados são exibidos. Este notebook descreve o painel e não ajusta nenhum \
modelo. Toda a lógica está em `src/retail_multilevel`; as células apenas carregam os dados, chamam as funções e \
exibem tabelas e gráficos.
""",
            en="""
# 01 · Exploratory analysis of the panel

*Data: dunnhumby Source Files – Breakfast at the Frat.* The raw data are licensed and are not redistributed with \
the repository; only aggregated results are shown. This notebook describes the panel and fits no model. All logic \
lives in `src/retail_multilevel`; the cells only load the data, call the functions and display tables and charts.
""",
        ),
        lang_cell(),
        code(SETUP),
        md(
            pt="""
## Briefing: como funciona o negócio

Para contextualizar a atividade comercial, esta seção descreve a rede, os produtos, o que é registrado a cada \
semana e as alavancas de preço e promoção. As definições vêm do guia do usuário e da aba \
*Glossary* do arquivo da dunnhumby; os números vêm do painel, e as tabelas abaixo os exibem.

**Quem.** Os dados representam as vendas de uma rede varejista, que o guia não identifica pelo nome, em lojas de \
quatro estados dos EUA. O painel tem 77 lojas, embora a tabela de lojas do guia liste 79 registros. Cada loja \
recebeu da rede um segmento (*Value*, *Mainstream* ou *Upscale*), que o guia apresenta como o patamar de preço da \
loja, sem explicar como ele foi definido. Em duas lojas (4503 e 17627) o cadastro traz dois segmentos diferentes, \
e por isso o segmento delas é tratado como ambíguo.
""",
            en="""
## Briefing: how the business works

To set the commercial context, this section describes the chain, the products, what is recorded each week and the \
price and promotion levers. Definitions come from dunnhumby's user guide \
and the *Glossary* sheet of its file; the numbers come from the panel, and the tables below display them.

**Who.** The data represent the sales of a retail chain, which the guide does not identify by name, in stores in \
four US states. The panel has 77 stores, although the guide's store table lists 79 records. Each store was assigned \
a segment by the chain (*Value*, *Mainstream* or *Upscale*), which the guide presents as the store's pricing level \
without explaining how it was defined. For two stores (4503 and 17627) the store table carries two different \
segments, so their segment is treated as ambiguous.
""",
        ),
        code("""
stores = panel.drop_duplicates("STORE_NUM")
display(
    pd.crosstab(
        stores["STATE"],
        stores["SEGMENT"].fillna(t("segment_ambiguous", LANG)),
        margins=True,
        margins_name=t("all_rows", LANG),
    )
)
"""),
        md(
            pt="""
**O quê.** Quatro categorias são acompanhadas: enxaguantes bucais (`ORAL HYGIENE \
PRODUCTS`), pretzels (`BAG SNACKS`), pizza congelada (`FROZEN PIZZA`) e cereal em caixa (`COLD CEREAL`). São 55 \
produtos (UPCs): 43 de marcas nacionais e 12 de marca própria da rede (fabricante `PRIVATE LABEL`), 3 em cada \
categoria, além de 15 outros fabricantes.
""",
            en="""
**What.** Four categories are tracked: mouthwash (`ORAL HYGIENE PRODUCTS`), pretzels (`BAG SNACKS`), frozen pizza \
(`FROZEN PIZZA`) and boxed cereal (`COLD CEREAL`). There are 55 products (UPCs): 43 national brands and 12 \
private-label products of the retailer (manufacturer `PRIVATE LABEL`), 3 in each category, besides 15 other \
manufacturers.
""",
        ),
        code("""
products = panel.drop_duplicates("UPC")
by_category = pd.crosstab(
    [products["CATEGORY"], products["SUB_CATEGORY"]],
    products["PRIVATE_LABEL"].map({False: t("brand_national", LANG), True: t("brand_private", LANG)}),
    margins=True,
    margins_name=t("all_rows", LANG),
)
display(by_category)
display(pd.Series({t("n_manufacturers", LANG): products["MANUFACTURER"].nunique()}).to_frame("n"))
"""),
        md(
            pt="""
**O que é registrado.** Cada linha do painel é um produto (UPC) em uma loja em uma semana (156 semanas, com datas \
de fim de semana de 2009-01-14 a 2012-01-04, segundo a tabela de visão geral abaixo). Para cada linha, o guia define \
as variáveis descritas a seguir.

O volume aparece em duas medidas: `UNITS` conta as unidades vendidas, e `SPEND` traz o faturamento em dólares. O \
alcance da demanda também aparece em duas: `VISITS` conta quantas cestas de compra levaram o produto pelo menos uma \
vez, e `HHS` conta quantos domicílios diferentes o compraram. O preço, por fim, é descrito por `PRICE`, o que o \
cliente de fato pagou na loja, e por `BASE_PRICE`, o preço de referência do item; a diferença entre os dois mostra \
se houve desconto e de quanto.
""",
            en="""
**What is recorded.** Each row of the panel is one product (UPC) in one store in one week (156 weeks, with \
week-ending dates from 2009-01-14 to 2012-01-04, according to the overview table below). For each row, the guide \
defines the variables described below.

Volume appears in two measures: `UNITS` counts the units sold, and `SPEND` holds the revenue in dollars. The reach \
of demand also appears in two: `VISITS` counts how many shopping baskets contained the product at least once, and \
`HHS` counts how many different households bought it. Price, finally, is described by `PRICE`, what the customer \
actually paid in the store, and by `BASE_PRICE`, the item's reference price; the gap between the two shows whether \
there was a discount and how large it was.
""",
        ),
        md(
            pt="""
**As alavancas semanais.** O guia registra três formas de apoio promocional por produto, loja e semana. A primeira \
é a **TPR** (*temporary price reduction*, `TPR_ONLY`), um corte de preço por tempo limitado que o cliente só \
percebe pelo preço marcado na prateleira, sem destaque físico nem divulgação. A segunda é o **encarte** \
(*feature*, `FEATURE`), quando o produto é anunciado no folheto de ofertas da loja. A terceira é o **display** \
(`DISPLAY`), quando o produto ganha um ponto de exposição especial dentro da loja.

`TPR_ONLY` é codificada como exclusiva de encarte e display: a seção 4 mostra que as duas situações nunca \
coincidem nas linhas do painel. Já encarte e display podem ocorrer juntos.
""",
            en="""
**The weekly levers.** The guide records three forms of promotional support per product, store and week. The first \
is **TPR** (temporary price reduction, `TPR_ONLY`), a time-limited price cut that customers notice only through the \
price marked on the shelf, with no physical highlight and no advertising. The second is the **feature** \
(`FEATURE`), when the product is advertised in the store's weekly flyer. The third is the **display** (`DISPLAY`), \
when the product gets a special exhibition spot inside the store.

`TPR_ONLY` is coded as exclusive of feature and display: section 4 shows that the two situations never coincide \
in the rows of the panel. Feature and display, however, can occur together.
""",
        ),
        md(
            pt="""
**O dilema.** Preço menor ou promoção costuma vender mais unidades, mas com margem unitária menor e, em alguns \
casos, com custos da própria promoção. Se compensa depende de quão forte é a resposta da demanda. Duas medidas \
resumem essa resposta. A **elasticidade-preço** é a variação percentual das unidades associada a uma \
variação de 1% no preço (Tellis, 1988; Bijmolt et al., 2005): com elasticidade de −2, uma redução de 10% no preço \
eleva as unidades em cerca de 20% \
(regra prática: −2 × −10%), e, no modelo em logaritmos, com elasticidade constante, o valor exato é \
0,9<sup>−2</sup> − 1 ≈ 23,5%. Já o **lift de promoção** (Blattberg & Neslin, 1989) descreve o efeito das promoções \
e é medido pelo aumento \
das unidades em semanas com a promoção, em relação a semanas sem ela.

**Por que é difícil.** A resposta difere entre lojas e produtos. Muitos pares loja × produto têm poucas semanas \
ou pouca variação de preço, e estimar cada par separadamente pode dar elasticidades muito ruidosas, inclusive \
positivas, o que não faz sentido econômico (a seção 8 apresenta a evidência nas semanas de treino). Usar uma única elasticidade para tudo ignora as diferenças. Os \
**modelos multinível** (*partial pooling*, ou agrupamento parcial; Gelman & Hill, 2007; Raudenbush & Bryk, 2002) \
ficam no meio: cada par é estimado \
aproveitando a informação dos demais, e pares com pouca informação são puxados para a média do grupo.
""",
            en="""
**The trade-off.** A lower price or a promotion usually sells more units, but with a lower unit margin and, in \
some cases, with the cost of the promotion itself. Whether it pays depends on how strongly demand responds. Two \
measures summarise that response. The **price elasticity** is the percentage change in units associated with a \
1% change in price (Tellis, 1988; Bijmolt et al., 2005): with an elasticity of −2, a 10% price cut raises units by \
about 20% (rule of thumb: \
−2 × −10%), and, in the log model, under constant elasticity, the exact value is 0.9<sup>−2</sup> − 1 ≈ 23.5%. \
The **promotion lift** (Blattberg & Neslin, 1989), in turn, describes the effect of promotions and is measured by \
the increase in units in \
weeks with the promotion, relative to weeks without it.

**Why it is hard.** The response differs across stores and products. Many store × product pairs have few weeks \
or little price variation, and estimating each pair separately can give very noisy elasticities, even positive \
ones, which makes no economic sense (section 8 presents the evidence from the training weeks). Using a single elasticity for everything ignores the differences. \
**Multilevel models** (partial pooling; Gelman & Hill, 2007; Raudenbush & Bryk, 2002) sit in between: each pair is \
estimated borrowing information from the \
others, and pairs with little information are pulled toward the group average.
""",
        ),
        md(
            pt="""
## A pergunta-chave

**Quanto o preço e as promoções movem as vendas em cada loja e em cada produto, e onde vale baixar o preço, \
aumentá-lo ou promover?** O guia propõe perguntas dessa natureza (por exemplo, para quais produtos reduzir o preço \
para aumentar as vendas e para quais aumentá-lo para melhorar o lucro). O projeto as organiza em cinco perguntas \
analíticas.

A primeira trata de onde as vendas variam: quanto da variação está entre lojas, entre produtos e entre pares loja \
× produto, e quanto é movimento semana a semana. A segunda pergunta é qual é a elasticidade-preço média e qual é o \
efeito de encarte, display e TPR. A terceira diz respeito à heterogeneidade desses efeitos, isto é, como variam \
entre lojas e produtos e o que explica a variação (por exemplo, segmento da loja, categoria, marca própria).

A quarta pergunta compara métodos: se os modelos multinível preveem melhor, fora da amostra, do que um modelo \
único (OLS) e do que um modelo de *boosting* (GPBoost; Sigrist, 2022). Por fim, a quinta pergunta é como \
transformar as \
estimativas em decisões (baixar, manter ou aumentar preço, promover ou não). Os coeficientes estimados são \
associações condicionais em semanas com transações registradas. Uma recomendação de preço ou um ROI incremental \
exige identificação causal e dados de margem, de modo que a camada de decisão do projeto é tratada como análise de \
cenários.

## Guia de leitura

| Notebook | Conteúdo | Situação |
|---|---|---|
| 01 | Análise exploratória do painel (este) | Disponível |
| 02 | Modelo nulo: decomposição da variância | Disponível |
| 03 | Efeitos fixos de preço e promoção, com sensibilidades e diagnóstico residual | Disponível |
| 04 | Protocolo P1: inclinações aleatórias e comparação fora da amostra | Disponível |
| 05 | Contrastes planejados do desconto (segmento, marca própria e categoria) e sensibilidade a choques de categoria × semana | Disponível |
| 06 | Cenários de preço e promoção, com intervalos pela covariância conjunta e suporte observado | Disponível |

## Glossário

| Termo | Significado |
|---|---|
| UPC | Código universal do produto (identificador do produto) |
| Par | Combinação de uma loja com um produto (loja × produto) |
| Cesta (*visit*) | Cada compra em que o produto entrou ao menos uma vez |
| Preço base | Preço regular do item, sem desconto temporário; a comparação com o preço de gôndola dá o desconto (no painel, o preço só fica abaixo do base em semanas com promoção) |
| TPR | Corte de preço por tempo limitado, percebido só pelo preço marcado na prateleira |
| Encarte (*feature*) | Produto anunciado no folheto de ofertas da loja |
| Display | Produto com ponto de exposição especial dentro da loja |
| Elasticidade-preço | Variação percentual das unidades para 1% de variação no preço |
| Lift de promoção | Aumento das unidades em semanas com a promoção, frente a semanas sem ela |
| *Partial pooling* | Estimar cada grupo usando também a informação dos demais, o que aproxima as estimativas de pouca informação da média |
| ICC | Correlação prevista entre observações que compartilham um ou mais agrupamentos |
| BLUP | Desvio previsto de um grupo em relação à média, já encolhido em direção a zero |
| Treino / teste | As primeiras 104 semanas ajustam os modelos; as últimas 52 formam a janela de avaliação temporal retrospectiva, com exposição exploratória prévia |
""",
            en="""
## The key question

**How much do price and promotions move sales in each store and each product, and where is it worth cutting the \
price, raising it or promoting?** The guide poses questions of this nature (for example, for which products to \
lower the price to increase sales and for which to raise it to improve profit). The project organises them into \
five analytical questions.

The first concerns where sales vary: how much of the variation lies between stores, between products and between \
store × product pairs, and how much is week-to-week movement. The second asks what the average price elasticity is \
and what the effect of feature, display and TPR is. The third concerns the heterogeneity of these effects, that \
is, how they vary across stores and products and what explains the variation (for example, store segment, \
category, private label).

The fourth question compares methods: whether multilevel models predict better out of sample than a single model \
(OLS) and than a boosting model (GPBoost; Sigrist, 2022). Finally, the fifth is how the estimates can be turned \
into decisions \
(cut, keep or raise the price, promote or not). The estimated coefficients are conditional associations in weeks \
with recorded transactions. A pricing recommendation or an incremental ROI requires causal identification and \
margin data, so the decision layer of the project is treated as a scenario analysis.

## Reading guide

| Notebook | Content | Status |
|---|---|---|
| 01 | Exploratory analysis of the panel (this one) | Available |
| 02 | Null model: variance decomposition | Available |
| 03 | Fixed effects of price and promotion, with sensitivity analyses and residual diagnostics | Available |
| 04 | Protocol P1: random slopes and out-of-sample comparison | Available |
| 05 | Planned discount contrasts (segment, private label and category) and sensitivity to category × week shocks | Available |
| 06 | Price and promotion scenarios, with intervals from the joint covariance and observed support | Available |

## Glossary

| Term | Meaning |
|---|---|
| UPC | Universal product code (the product identifier) |
| Pair | A combination of one store and one product (store × product) |
| Basket (*visit*) | Each purchase in which the product appeared at least once |
| Base price | Regular price of the item, without a temporary discount; comparing it with the shelf price gives the discount (in the panel, price falls below base only in weeks with a promotion) |
| TPR | Time-limited price cut, noticed only through the price marked on the shelf |
| Feature | Product advertised in the store's weekly flyer |
| Display | Product given a special exhibition spot inside the store |
| Price elasticity | Percentage change in units for a 1% change in price |
| Promotion lift | Increase in units in weeks with the promotion, relative to weeks without it |
| Partial pooling | Estimating each group also using the information of the others, which pulls low-information estimates toward the average |
| ICC | Model-implied correlation between observations sharing one or more groups |
| BLUP | Predicted deviation of a group from the mean, already shrunk toward zero |
| Train / test | The first 104 weeks fit the models; the last 52 form the retrospective temporal evaluation window, with prior exploratory exposure |
""",
        ),
        md(
            pt="""
## Quais variáveis entram na análise?

Cada linha do painel agregado representa um UPC em uma loja em uma semana observada. A mesma loja e o mesmo \
produto aparecem em muitas linhas. Seus códigos identificam as unidades, enquanto preço e promoção descrevem as \
condições daquela semana. O quadro distingue a resposta, os preditores e as informações de contexto.

| Colunas no painel | O que representam | Papel na análise |
|---|---|---|
| `STORE_NUM`, `UPC`, `WEEK_END_DATE` | Código da loja, código do produto e data de fim da semana | Chave da observação semanal. `PAIR_ID` identifica a combinação loja × UPC ao longo do tempo. |
| `UNITS`, `LOG_UNITS` | Quantidade vendida e seu logaritmo natural | `LOG_UNITS` é a variável-resposta do modelo principal. |
| `PRICE`, `BASE_PRICE` | Preço de gôndola e preço regular, por unidade do produto | A comparação entre os dois dá o desconto. Ausências de preço regular são tratadas e sinalizadas na limpeza. |
| `FEATURE`, `DISPLAY`, `TPR_ONLY` | Encarte, exposição especial e redução temporária de preço sem encarte nem display, respectivamente | Indicadores 0/1 usados como preditores. O nome `FEATURE` nesta base significa encarte. |
| `WEEK`, `WEEK_OF_YEAR` | Índice sequencial da semana e número da semana no calendário ISO | Origem dos controles de tendência e sazonalidade. |
| `SEGMENT`, `CATEGORY`, `PRIVATE_LABEL` | Segmento da loja, categoria e indicador de marca própria | Moderadores da associação com desconto no notebook 05. Esses atributos são constantes dentro de um par. |
| `VISITS`, `HHS`, `SPEND` | Cestas que incluíram o produto, domicílios compradores e gasto total | Medidas observadas na mesma semana, usadas para descrição e controle de qualidade, fora dos preditores do C0. |
| `CITY`, `STATE`, `MSA_CODE`, `SALES_AREA`, `AVG_WEEKLY_BASKETS` | Localização e características da loja | Contexto do cadastro, fora dos preditores do C0. |

### Como os preditores do C0 são construídos

| Variável usada no modelo | Construção | Interpretação |
|---|---|---|
| `LOG_DISC` | `ln(PRICE / BASE_PRICE)` | É zero quando os preços coincidem e negativo quando há desconto. |
| `LOG_BASE_MEAN` | Média de `ln(BASE_PRICE)` do par na janela de ajuste | Representa o nível de preço regular do par. Em um par novo com UPC conhecido, é usada a média do UPC na janela de ajuste. |
| `LOG_BASE_DEV` | `ln(BASE_PRICE) − LOG_BASE_MEAN` | Mede o quanto o preço regular da semana se desvia da referência do par. |
| `FEATURE`, `DISPLAY`, `TPR_ONLY` | Indicadores registrados na semana | Identificam a modalidade de promoção com a profundidade do desconto incluída no mesmo modelo. |
| `TREND`, `SIN_1` a `SIN_3`, `COS_1` a `COS_3` | Índice da semana em anos e três pares de termos de Fourier do calendário (Hyndman & Athanasopoulos, 2021) | Representam a tendência e a sazonalidade anual. |

Por exemplo, um preço de 1,80 para um preço regular de 2,00 corresponde a um desconto de 10% e a `LOG_DISC` \
de aproximadamente −0,105. As chaves de loja, produto e par ligam a observação aos respectivos interceptos \
aleatórios. No C0, os coeficientes dos preditores da tabela são comuns aos grupos.
""",
            en="""
## Which variables are used in the analysis?

Each row in the aggregated panel represents a UPC at a store in an observed week. The same store and product \
appear in many rows. Their codes identify the units, while price and promotion describe conditions that week. \
The table distinguishes the outcome, predictors, and contextual information.

| Panel columns | What they represent | Role in the analysis |
|---|---|---|
| `STORE_NUM`, `UPC`, `WEEK_END_DATE` | Store code, product code, and week-ending date | Weekly observation key. `PAIR_ID` identifies the store × UPC combination over time. |
| `UNITS`, `LOG_UNITS` | Quantity sold and its natural logarithm | `LOG_UNITS` is the outcome of the main model. |
| `PRICE`, `BASE_PRICE` | Shelf price and regular price per unit | Their comparison gives the discount. Missing regular prices are handled and flagged during cleaning. |
| `FEATURE`, `DISPLAY`, `TPR_ONLY` | Flyer advertising, special in-store display, and a temporary price reduction with neither flyer nor display, respectively | Binary predictors. In this dataset, the field `FEATURE` means flyer advertising. |
| `WEEK`, `WEEK_OF_YEAR` | Sequential week index and ISO calendar week number | Source of the trend and seasonality controls. |
| `SEGMENT`, `CATEGORY`, `PRIVATE_LABEL` | Store segment, product category, and private-label indicator | Moderators of the discount association in notebook 05. These attributes are constant within a pair. |
| `VISITS`, `HHS`, `SPEND` | Baskets containing the product, purchasing households, and total spend | Measures observed in the same week, used for description and quality checks, outside C0's predictors. |
| `CITY`, `STATE`, `MSA_CODE`, `SALES_AREA`, `AVG_WEEKLY_BASKETS` | Store location and attributes | Store lookup context, outside C0's predictors. |

### How C0's predictors are constructed

| Model variable | Construction | Interpretation |
|---|---|---|
| `LOG_DISC` | `ln(PRICE / BASE_PRICE)` | Zero when prices match and negative when the product is discounted. |
| `LOG_BASE_MEAN` | Pair mean of `ln(BASE_PRICE)` in the fitting window | Represents the pair's regular price level. For a new pair with a known UPC, the UPC mean in the fitting window is used. |
| `LOG_BASE_DEV` | `ln(BASE_PRICE) − LOG_BASE_MEAN` | Measures how far the week's regular price is from the pair's reference. |
| `FEATURE`, `DISPLAY`, `TPR_ONLY` | Indicators recorded in the week | Identify the promotion type while discount depth is included in the same model. |
| `TREND`, `SIN_1` to `SIN_3`, `COS_1` to `COS_3` | Week index expressed in years and three pairs of calendar Fourier terms (Hyndman & Athanasopoulos, 2021) | Represent the trend and annual seasonality. |

For example, a price of 1.80 against a regular price of 2.00 corresponds to a 10% discount and a `LOG_DISC` \
of approximately −0.105. Store, product, and pair keys connect each observation to its random intercepts. \
In C0, the predictor coefficients in the table are shared across groups.
""",
        ),
        md(
            pt="""
## O painel tem dois ou três níveis?

Há **dois níveis na parte aninhada**: a observação semanal e o par loja × UPC, que é a unidade acompanhada ao \
longo do tempo. O modelo também agrupa os pares por loja e por UPC. Esses dois agrupamentos são **cruzados** \
(Baayen et al., 2008): a loja A vende mais de um UPC e o UPC A aparece em mais de uma loja.
""",
            en="""
## Does the panel have two or three levels?

There are **two levels in the nested part**: the weekly observation and the store × UPC pair, which is the unit \
followed over time. The model also groups pairs by store and by UPC. These two groupings are **crossed** \
(Baayen et al., 2008): store A sells more than one UPC, and UPC A appears in more than one store.
""",
        ),
        code("fig = viz.plot_panel_structure(LANG)"),
        md(
            pt="""
A figura usa **três faixas** para mostrar os agrupamentos cruzados, os pares e suas observações semanais. \
O C0 estima interceptos aleatórios de loja, UPC e par (`STORE_NUM:UPC`). Semanas do mesmo par compartilham \
os três interceptos. Pares distintos podem compartilhar apenas a loja ou apenas o UPC. A semana do calendário \
é comum aos pares observados naquela data e, no C0, entra por controles fixos de tendência e sazonalidade.

Essa estrutura motiva o uso do multinível. No modelo nulo M4 do notebook 02, ajustado apenas no treino, \
**57,2%** da variância de `LOG_UNITS` foi atribuída ao UPC, **9,6%** à loja, **8,5%** ao par e **24,6%** ao \
resíduo. Duas semanas do mesmo par compartilham os três componentes de agrupamento, o que implica correlação \
modelada de **75,4%** sob a hipótese residual do M4. O notebook 03 examina separadamente a dependência que \
permanece entre semanas após o ajuste.
""",
            en="""
The figure uses **three bands** to show crossed groups, pairs, and their weekly observations. C0 estimates random \
intercepts for store, UPC, and pair (`STORE_NUM:UPC`). Weeks within the same pair share all three intercepts. \
Different pairs may share only the store or only the UPC. The calendar week is shared by pairs observed on that \
date and enters C0 through fixed trend and seasonality controls.

This structure motivates multilevel modelling. In notebook 02's null model M4, fitted on training data only, \
**57.2%** of `LOG_UNITS` variance was attributed to UPC, **9.6%** to store, **8.5%** to the pair, and **24.6%** to \
the residual. Two weeks within the same pair share all three grouping components, implying a **75.4%** modelled \
correlation under M4's residual assumption. Notebook 03 separately examines the dependence that remains between \
weeks after fitting the model.
""",
        ),
        md(
            pt="""
## 1. Visão geral e limpeza

O painel limpo foi lido de `data/processed/panel.parquet`. A tabela abaixo resume o seu tamanho e o período \
coberto.
""",
            en="""
## 1. Overview and cleaning

The cleaned panel was read from `data/processed/panel.parquet`. The table below summarises its size and the period \
covered.
""",
        ),
        code('eda.panel_overview(panel).to_frame("panel")'),
        md(
            pt="""
As regras de limpeza (decisões D6 a D10 do registro de decisões) foram aplicadas antes desta análise. As contagens \
são de linhas do painel.

| Regra | Tratamento | Linhas afetadas |
|---|---|---|
| D6 | Lojas com segmento ambíguo (4503 e 17627) sinalizadas e excluídas dos modelos com segmento | 13.693 (2,6%) |
| D7 | Linhas com `PRICE` nulo ou ≤ 0 descartadas | 24 |
| D8 | `BASE_PRICE` nulo preenchido só com o histórico do par: 44 linhas com a semana anterior e 141 com o fallback `BASE_PRICE = PRICE`, que zera o desconto da linha e ocorre sempre em semanas com promoção (flags `BASE_IMPUTED` e `BASE_FALLBACK`) | 185 (44 + 141) |
| D9 | `OUTLIER_UPV` (unidades por visita > 5) excluído da amostra de modelagem; `OUTLIER_VPH` apenas sinalizado | 67 / 293 |
| D10 | Preço acima de 1,05 × base sinalizado (`PRICE_ABOVE_BASE`), mas mantido | 2.071 |

Das 524.926 linhas, 524.859 compõem a amostra de modelagem (`IN_MODEL`), isto é, todas exceto as 67 linhas de \
`OUTLIER_UPV`.
""",
            en="""
The cleaning rules (decisions D6 to D10 of the decision log) were applied before this analysis. The counts are \
panel rows.

| Rule | Treatment | Rows affected |
|---|---|---|
| D6 | Stores with an ambiguous segment (4503 and 17627) flagged and excluded from the segment models | 13,693 (2.6%) |
| D7 | Rows with null or non-positive `PRICE` dropped | 24 |
| D8 | Null `BASE_PRICE` filled using only the history of the pair: 44 rows with the previous week and 141 with the fallback `BASE_PRICE = PRICE`, which sets the discount of the row to zero and always occurs in promotion weeks (flags `BASE_IMPUTED` and `BASE_FALLBACK`) | 185 (44 + 141) |
| D9 | `OUTLIER_UPV` (units per visit > 5) excluded from the modelling sample; `OUTLIER_VPH` only flagged | 67 / 293 |
| D10 | Price above 1.05 × base flagged (`PRICE_ABOVE_BASE`) but kept | 2,071 |

Of the 524,926 rows, 524,859 form the modelling sample (`IN_MODEL`), that is, all except the 67 `OUTLIER_UPV` rows.
""",
        ),
        md(
            pt="""
## 2. Divisão temporal (D11, revisada)

As primeiras 104 semanas formam o treino, e somente elas foram usadas para ajustar modelos e selecionar \
especificações. As últimas 52 semanas (um ciclo sazonal completo, de 2011-01-12 a 2012-01-04) formam a **janela de \
avaliação temporal retrospectiva com exposição exploratória prévia**. Nenhum modelo foi ajustado nelas, mas esta \
análise exploratória (as unidades semanais, a visão geral e as taxas de promoção) e o OLS exploratório do \
`PROJECT_BRIEF.md`, com o perfil de variação de preço, usaram o painel inteiro, de modo que a janela não é um \
conjunto de teste intocado. As formulações motivadas por essa exploração, a decomposição de preço da D12 incluída, \
são decisões de desenvolvimento. Nenhuma comparação preditiva foi calculada nessa janela até aqui, e a comparação \
fora da amostra (Tashman, 2000) seguirá um protocolo registrado antes do ajuste dos candidatos (Nosek et al., \
2018; Weston et al., 2019). Nas tabelas e nos gráficos, o \
rótulo "teste" designa essa janela.
""",
            en="""
## 2. Temporal split (D11, revised)

The first 104 weeks form the training sample, and only they were used to fit models and select specifications. The \
last 52 weeks (one full seasonal cycle, from 2011-01-12 to 2012-01-04) form the **retrospective temporal \
evaluation window with prior exploratory exposure**. No model was fitted on them, but this exploratory analysis \
(the weekly units, the overview and the promotion rates) and the exploratory OLS of `PROJECT_BRIEF.md`, with the \
price-variation profile, used the whole panel, so the window is not an untouched test set. The formulations \
motivated by that exploration, the price decomposition of D12 included, are development decisions. No predictive \
comparison has been computed on this window so far, and the out-of-sample comparison (Tashman, 2000) will follow a \
protocol \
recorded before the candidates are fitted (Nosek et al., 2018; Weston et al., 2019). In the tables and charts, the \
label "test" designates this window.
""",
        ),
        code("""
weekly = eda.weekly_units(panel)
fig = viz.plot_weekly_units(weekly, LANG)
"""),
        code("""
split_sizes = pd.DataFrame(
    {"rows": [len(train), len(test)], "pairs": [train["PAIR_ID"].nunique(), test["PAIR_ID"].nunique()]},
    index=[t("legend_train", LANG), t("legend_test", LANG)],
)
display(split_sizes)
display(pd.Series({"pairs_only_in_test": len(set(test["PAIR_ID"]) - set(train["PAIR_ID"]))}).to_frame("n"))
"""),
        md(
            pt="""
A amostra de treino tem 347.092 linhas em 3.859 pares, e a de teste tem 177.767 linhas em 3.870 pares. Cinquenta \
pares aparecem apenas na janela de avaliação: para eles não há histórico de treino, e as previsões dependerão só \
dos efeitos de loja e de produto.
""",
            en="""
The training sample has 347,092 rows in 3,859 pairs, and the test sample has 177,767 rows in 3,870 pairs. Fifty \
pairs appear only in the evaluation window: they have no training history, so their predictions will rely only on \
the store and product effects.
""",
        ),
        md(
            pt="""
## 3. Cobertura dos pares

O histograma conta as semanas de treino de cada par loja × produto.
""",
            en="""
## 3. Pair coverage

The histogram counts the training weeks of each store × product pair.
""",
        ),
        code("""
coverage = eda.pair_coverage(train)
fig = viz.plot_pair_coverage(coverage, LANG)
coverage["n_weeks_train"].describe().to_frame()
"""),
        md(
            pt="""
A mediana é de 102 semanas de treino por par, e 1.537 pares têm as 104 semanas. O painel é, portanto, \
desbalanceado: 25% dos pares têm 86 semanas ou menos, e há pares com uma única semana. Esses pares têm pouca \
informação própria, o que é justamente onde o *partial pooling* ajuda.
""",
            en="""
The median is 102 training weeks per pair, and 1,537 pairs have all 104 weeks. The panel is therefore unbalanced: \
25% of the pairs have 86 weeks or fewer, and some pairs have a single week. Such pairs carry little information of \
their own, which is where partial pooling helps.
""",
        ),
        md(
            pt="""
## 4. Promoções

As taxas abaixo são a participação de linhas com cada promoção ligada, no painel inteiro, no total e por categoria. \
`BELOW_BASE` indica preço abaixo de 95% do preço base.
""",
            en="""
## 4. Promotions

The rates below are the share of rows with each promotion on, in the whole panel, overall and by category. \
`BELOW_BASE` marks a price below 95% of the base price.
""",
        ),
        code("""
display(eda.promo_rates(panel))
rates_by_category = eda.promo_rates(panel, "CATEGORY")
display(rates_by_category)
fig = viz.plot_promo_rates(pd.concat([eda.promo_rates(panel), rates_by_category]), LANG)
"""),
        code("""
tpr = panel["TPR_ONLY"].astype(bool).rename("TPR_ONLY")
feature_or_display = (panel["FEATURE"].astype(bool) | panel["DISPLAY"].astype(bool)).rename("FEATURE or DISPLAY")
display(pd.crosstab(tpr, feature_or_display))
below_base = (panel["PRICE"] < 0.95 * panel["BASE_PRICE"]).rename(t("promo_below_base", LANG))
any_promo = panel["ANY_PROMO"].astype(bool).rename(t("promo_any_promo", LANG))
display(pd.crosstab(below_base, any_promo))
"""),
        md(
            pt="""
Alguma promoção está ativa em 28,5% das linhas: encarte em 8,4%, display em 11,0% e TPR em 13,5%. A primeira \
tabela cruzada mostra que `TPR_ONLY` nunca coincide com encarte ou display (0 linhas), pela própria definição da \
variável. A segunda mostra que preço abaixo de 95% do base nunca ocorre sem promoção (0 linhas), embora 19.555 \
linhas com promoção não tenham redução de preço de 5% ou mais; as 129.830 linhas restantes com promoção têm preço \
abaixo do base. As taxas diferem entre categorias (por exemplo, pizza congelada tem a maior participação de \
encarte, 16,7%, e snacks a menor, 2,5%), de modo que o tipo de promoção depende do produto.
""",
            en="""
Some promotion is active in 28.5% of the rows: feature in 8.4%, display in 11.0% and TPR in 13.5%. The first \
cross-tabulation shows that `TPR_ONLY` never coincides with feature or display (0 rows), by definition of the \
variable. The second shows that a price below 95% of base never occurs without a promotion (0 rows), although \
19,555 rows with a promotion have no price cut of 5% or more; the remaining 129,830 promotion rows have a price \
below base. The rates differ across categories (for instance, frozen pizza has the highest feature share, 16.7%, \
and bag snacks the lowest, 2.5%), so the type of promotion depends on the product.
""",
        ),
        md(
            pt="""
## 5. Variação de preço

A variação de preço dentro de cada par (desvio-padrão de `LOG_PRICE`, na amostra de treino) é o sinal que \
identifica a elasticidade.
""",
            en="""
## 5. Price variation

The price variation within each pair (standard deviation of `LOG_PRICE`, in the training sample) is the signal that \
identifies the elasticity.
""",
        ),
        code("""
variation = eda.price_variation(train)
variation[["sd_log_price", "share_below_base"]].describe()
"""),
        md(
            pt="""
O desvio-padrão mediano de log(preço) dentro do par é de 0,111 (cerca de 11%), e os pares ficam em média 24% das \
semanas com preço abaixo do base. Há variação suficiente para estimar a elasticidade na maioria dos pares, mas \
ela se concentra nas semanas de promoção, o que dificulta separar o efeito do preço do efeito da própria \
promoção.
""",
            en="""
The median standard deviation of log(price) within a pair is 0.111 (about 11%), and pairs spend 24% of their \
weeks, on average, with a price below base. There is enough variation to estimate the elasticity in most pairs, but \
it is concentrated in promotion weeks, which makes it harder to separate the effect of price from the effect of the \
promotion itself.
""",
        ),
        md(
            pt="""
## 6. Participação descritiva na variância

Para cada agrupamento, calculou-se a fração da variância de `LOG_UNITS` que está entre os grupos (eta ao \
quadrado), na amostra de treino. O cálculo é descritivo e feito um agrupamento por vez; como loja, produto e par \
não são aninhados de forma exclusiva, as frações não somam 1.
""",
            en="""
## 6. Descriptive variance shares

For each grouping, the fraction of the variance of `LOG_UNITS` lying between groups (eta squared) was computed on \
the training sample. The computation is descriptive and done one grouping at a time; because store, product and \
pair are not mutually nested, the shares do not sum to 1.
""",
        ),
        code("""
shares = eda.variance_shares(train)
display(shares)
fig = viz.plot_variance_shares(shares, LANG, share_col="between_share")
"""),
        md(
            pt="""
O produto explica 53,3% da variância de log(unidades), a loja 6,7% e o par loja × produto 72,7% (o par contém \
os dois e a sua interação). Essas frações não vêm de um modelo e não separam efeitos: o par tem 3.859 médias e, \
por isso, tende a ser maior. Elas motivam o modelo nulo de efeitos aleatórios cruzados (Baayen et al., 2008) do \
notebook 02, que \
decompõe a variância em loja, produto, interação e resíduo de forma consistente.
""",
            en="""
The product explains 53.3% of the variance of log(units), the store 6.7% and the store × product pair 72.7% (the \
pair contains both and their interaction). These shares do not come from a model and do not separate effects: the \
pair has 3,859 means and therefore tends to be larger. They motivate the crossed random-effects null model (Baayen \
et al., 2008) of \
notebook 02, which decomposes the variance into store, product, interaction and residual consistently.
""",
        ),
        md(
            pt="""
## 7. Lift bruto de promoção

O lift bruto compara, dentro de cada par, o log de unidades em semanas com cada promoção ligada com o das semanas \
sem nenhuma promoção (amostra de treino).
""",
            en="""
## 7. Raw promotion lift

The raw lift compares, within each pair, the log of units in weeks with each promotion on against weeks with no \
promotion at all (training sample).
""",
        ),
        code("""
lift = eda.raw_promo_lift(train)
display(lift)
fig = viz.plot_raw_promo_lift(lift, LANG)
"""),
        md(
            pt="""
O lift bruto foi de 195% para encarte, 129% para display e 33% para TPR. Esses números são descritivos e não \
causais: cada promoção está confundida com a redução de preço e com as demais promoções, e encarte e display \
podem ser aplicados nos produtos e semanas de maior demanda. Separar o efeito do preço do efeito de cada promoção \
é tarefa dos modelos.
""",
            en="""
The raw lift was 195% for feature, 129% for display and 33% for TPR. These numbers are descriptive and not causal: \
each promotion is confounded with the price cut and with the other promotions, and feature and display may be \
applied to the products and weeks with the highest demand. Separating the effect of price from the effect of each \
promotion is the task of the models.
""",
        ),
        md(
            pt="""
## 8. Inclinações do desconto estimadas par a par

A seção de briefing afirma que estimar cada par loja × produto separadamente pode dar elasticidades ruidosas. Para verificar essa afirmação, ajustou-se um OLS para cada par nas semanas de treino (0 a 103), com a especificação `LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY`, em que `LOG_DISC = log(preço / preço base)`. Entram pares com ao menos 8 semanas, e um termo sem variação no par, ou colinear com os anteriores, deixa de ser estimado (o par fica sem inclinação quando é o `LOG_DISC` que não varia). O ajuste é feito por script (`scripts/fit_pair_ols_slopes.py`), que grava a execução lida abaixo, e o notebook apenas lê os resultados. Como `LOG_DISC` é negativo nas semanas de desconto, uma inclinação negativa significa que mais desconto vem com mais unidades; uma inclinação positiva significa o oposto.
""",
            en="""
## 8. Discount slopes estimated pair by pair

The briefing section states that estimating each store × product pair separately can give noisy elasticities. To check that statement, one OLS was fitted for each pair on the training weeks (0 to 103), with the specification `LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY`, where `LOG_DISC = log(price / base price)`. Pairs with at least 8 weeks enter, and a term without variation in the pair, or collinear with the previous ones, is not estimated (the pair is left without a slope when it is `LOG_DISC` that does not vary). The fit is done by a script (`scripts/fit_pair_ols_slopes.py`), which stores the run read below, and the notebook only reads the results. Because `LOG_DISC` is negative in discount weeks, a negative slope means that more discount comes with more units; a positive slope means the opposite.
""",
        ),
        code("""
from retail_multilevel.i18n import localize_columns, stratum_label
from retail_multilevel.runs import load_run

pair_run = load_run("pair_ols_slopes", "20261126-104604-bd48a521")
print(pair_run.run_id)
pair_slopes = pair_run.load_table("slopes")
pair_summary = pair_run.load_table("summary")
pair_summary["stratum"] = [stratum_label(x, LANG) for x in pair_summary["stratum"]]
with pd.option_context("display.float_format", "{:,.2f}".format):
    display(localize_columns(pair_summary, LANG))
fig = viz.plot_pair_slopes(pair_slopes, LANG)
"""),
        md(
            pt="""
Das 3.859 combinações de treino, 3.788 tiveram a inclinação do desconto estimada: 42 pares têm menos de 8 semanas e 29 foram ajustados sem variação de desconto estimável. A mediana das inclinações é de −1,65, mas a dispersão é grande, com o percentil 5 em −5,40 e o percentil 95 em +3,18. Um total de 751 pares (19,8%) tem inclinação positiva. A participação é de 36,2% nos pares com menos de 26 semanas (25 de 69), de 27,3% nos de 26 a 77 semanas (148 de 543) e de 18,2% nos de 78 semanas ou mais (578 de 3.176). Ela diminui com o histórico, mas ainda alcança quase um par em cada cinco entre os mais longos. Inclinações exatamente nulas, que ocorrem em pares com unidades constantes, e coeficientes nulos a menos de arredondamento não são contados como positivos.

Uma inclinação positiva do desconto não tem sentido econômico: ela diz que a redução de preço vem acompanhada de menos unidades. Em um par isolado, isso sinaliza ruído, pelo número reduzido de semanas com desconto, ou confusão com outros fatores, como a época em que as promoções ocorrem. É esse comportamento que motiva o *partial pooling* (Gelman & Hill, 2007) testado nos notebooks 02 a 04, em que cada par toma emprestada informação dos demais em vez de ser estimado sozinho.
""",
            en="""
Of the 3,859 training pairs, 3,788 had the discount slope estimated: 42 pairs have fewer than 8 weeks and 29 were fitted without estimable discount variation. The median slope is −1.65, but the dispersion is large, with the 5th percentile at −5.40 and the 95th percentile at +3.18. A total of 751 pairs (19.8%) have a positive slope. The share is 36.2% among pairs with fewer than 26 weeks (25 of 69), 27.3% among those with 26 to 77 weeks (148 of 543) and 18.2% among those with 78 weeks or more (578 of 3,176). It decreases with history, but it still reaches almost one pair in five among the longest ones. Slopes that are exactly zero, which occur in pairs with constant units, and coefficients that are zero up to rounding are not counted as positive.

A positive discount slope has no economic sense: it says that a price cut comes with fewer units. In a single pair, this signals noise, because of the small number of discount weeks, or confounding with other factors, such as the timing of promotions. This behaviour motivates the partial pooling (Gelman & Hill, 2007) tested in notebooks 02 to 04, in which each pair borrows information from the others instead of being estimated alone.
""",
        ),
        md(
            pt="""
## 9. Resumo

O painel tem 524.926 linhas, 77 lojas, 55 produtos, 3.909 pares e 156 semanas, e a amostra de modelagem exclui 67 \
linhas. As últimas 52 semanas formam a janela de avaliação retrospectiva, com exposição exploratória prévia, e 50 \
pares só aparecem nesse período. O painel de treino é desbalanceado (mediana de 102 semanas por par), o que \
favorece o *partial pooling*. Estimadas par a par, 19,8% das inclinações do desconto são positivas, o que reforça essa motivação.

Alguma promoção está ativa em 28,5% das linhas; `TPR_ONLY` nunca coincide com encarte ou display, e o preço abaixo \
do base só ocorre sob promoção. O lift bruto de promoção é alto, mas está confundido com o preço. Quanto à \
variância de log(unidades), ela se encontra sobretudo entre produtos e pares, e as frações descritivas não são \
aditivas.

O notebook 02 lê os modelos nulos ajustados por script (`runs.load_run("null_models", "20261124-114105-b15be79c")`) e decompõe a variância \
entre loja, produto, interação e resíduo.
""",
            en="""
## 9. Summary

The panel has 524,926 rows, 77 stores, 55 products, 3,909 pairs and 156 weeks, and the modelling sample excludes \
67 rows. The last 52 weeks form the retrospective evaluation window, with prior exploratory exposure, and 50 pairs \
appear only in that period. The training panel is unbalanced (median of 102 weeks per pair), which favours partial \
pooling. Estimated pair by pair, 19.8% of the discount slopes are positive, which reinforces this motivation.

Some promotion is active in 28.5% of the rows; `TPR_ONLY` never coincides with feature or display, and a price \
below base occurs only under promotion. The raw promotion lift is large but is confounded with price. As for the \
variance of log(units), it lies mostly between products and pairs, and the descriptive shares are not additive.

Notebook 02 reads the null models fitted by script (`runs.load_run("null_models", "20261124-114105-b15be79c")`) and decomposes the variance \
into store, product, interaction and residual.
""",
        ),
    ]


SETUP_02 = """
import warnings

import pandas as pd
from IPython.display import display

from retail_multilevel import viz
from retail_multilevel.i18n import t
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import load_run

warnings.filterwarnings("ignore", category=FutureWarning)
viz.use_retina()
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:,.4f}".format)
run = load_run("null_models", "20261124-114105-b15be79c")
print(t("data_credit", LANG))
print("run:", run.run_id, "| finished_at:", run.metadata["finished_at"])
sample = run.load_json("sample")
display(pd.Series(sample).to_frame("training sample"))
"""


def null_model_cells() -> list[Cell]:
    """Cells of notebook 02, the null models read from the stored run."""
    return [
        md(
            pt="""
# 02 · Modelo nulo: decomposição da variância

Na estratégia de seleção por etapas (*step-up*) de modelos multinível (Raudenbush & Bryk, 2002), o passo inicial é \
o modelo nulo, sem \
preditores, que decompõe a variância da resposta entre os níveis da hierarquia. Aqui, porém, a hierarquia é \
**cruzada** (Baayen et al., 2008): as semanas estão aninhadas em pares loja × produto, e cada par pertence a uma \
loja e a um produto, que \
se cruzam entre si. Isso difere das hierarquias puramente aninhadas, de dois ou três níveis, em que cada unidade \
pertence a um único grupo do nível acima. Somente a amostra de treino (as 104 semanas iniciais, D11 \
revisada) foi usada, e nenhum ajuste tocou as 52 semanas finais.

Este notebook **não ajusta modelos**: os ajustes foram feitos por `scripts/fit_null_models.py` e gravados em \
`outputs/runs/null_models`. O notebook apenas lê essa execução, o que torna as versões em português e em inglês \
idênticas em resultados.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 02 · Null model: variance decomposition

In the step-up selection strategy for multilevel models (Raudenbush & Bryk, 2002), the first step is the null \
model, with no predictors, \
which splits the variance of the response across the levels of the hierarchy. Here, however, the hierarchy is \
**crossed** (Baayen et al., 2008): weeks are nested in store × product pairs, and each pair belongs to one store \
and one product, which \
cross each other. This differs from purely nested hierarchies, with two or three levels, in which each unit belongs \
to a single group of the level above. Only the training sample (the initial 104 weeks, D11 \
revised) was used, and no fit touched the final 52 weeks.

This notebook **fits no models**: the fits were made by `scripts/fit_null_models.py` and stored in \
`outputs/runs/null_models`. The notebook only reads that run, so the Portuguese and English versions show \
identical results.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_02),
        md(
            pt="""
A execução lida está identificada acima (identificador e horário de término), e a amostra de treino tem 347.092 \
linhas, 77 lojas, 55 produtos e 3.859 pares, de 2009-01-14 a 2011-01-05.
""",
            en="""
The run that was read is identified above (identifier and finish time), and the training sample has 347,092 rows, \
77 stores, 55 products and 3,859 pairs, from 2009-01-14 to 2011-01-05.
""",
        ),
        md(
            pt="""
## 1. Os modelos M0 a M4

Cinco modelos foram ajustados, todos por REML (Bates et al., 2015) e com a mesma parte fixa (apenas o intercepto), \
de modo que as \
verossimilhanças REML são comparáveis (Pinheiro & Bates, 2000). M0 é a regressão de nível único (OLS); de M1 a M4, \
efeitos aleatórios de \
loja, de produto, de ambos e, por fim, de ambos mais a interação loja × produto (o par) foram adicionados.
""",
            en="""
## 1. Models M0 to M4

Five models were fitted, all by REML (Bates et al., 2015) and with the same fixed part (intercept only), so that \
the REML likelihoods \
are comparable (Pinheiro & Bates, 2000). M0 is the single-level regression (OLS); from M1 to M4, random effects \
were added for store, for \
product, for both and, finally, for both plus the store × product interaction (the pair).
""",
        ),
        code("""
fits = run.load_table("fits")
display(fits[["name", "formula"]])
"""),
        md(
            pt="""
## 2. Estatísticas de ajuste

A tabela traz o número de parâmetros, a log-verossimilhança REML, o AIC, o BIC, o tempo de ajuste e o número de \
mensagens de convergência.
""",
            en="""
## 2. Fit statistics

The table gives the number of parameters, the REML log-likelihood, the AIC, the BIC, the fitting time and the \
number of convergence messages.
""",
        ),
        code("""
display(fits[["name", "npar", "loglik", "aic", "bic", "seconds", "n_messages"]])
"""),
        md(
            pt="""
A log-verossimilhança sobe de −563.135 (M0) para −346.786 (M4), e AIC e BIC diminuem a cada modelo adicionado (AIC \
de 1.126.273 para 693.583). Nenhum ajuste gerou mensagens de convergência (0 em todos). O M4, o maior, levou cerca \
de 29 segundos.
""",
            en="""
The log-likelihood rises from −563,135 (M0) to −346,786 (M4), and AIC and BIC fall with every added model (AIC \
from 1,126,273 to 693,583). No fit produced convergence messages (0 in all). M4, the largest, took about 29 \
seconds.
""",
        ),
        md(
            pt="""
## 3. Testes de razão de verossimilhanças

Cada teste compara um modelo restrito com o modelo completo, que tem um componente de variância a mais. Sob a \
hipótese nula, esse componente é zero, isto é, está na **fronteira** do espaço de parâmetros (σ² = 0), e a \
estatística não segue um χ² com 1 grau de liberdade. A distribuição de referência é uma mistura 50:50 de χ²₀ e χ²₁ \
(Self & Liang, 1987; Stram & Lee, 1994), de modo que `p_value` é a metade de `p_naive`.
""",
            en="""
## 3. Likelihood-ratio tests

Each test compares a restricted model with the full model, which has one more variance component. Under the null \
hypothesis that component is zero, that is, on the **boundary** of the parameter space (σ² = 0), and the statistic \
does not follow a χ² with 1 degree of freedom. The reference distribution is a 50:50 mixture of χ²₀ and χ²₁ (Self \
& Liang, 1987; Stram & Lee, 1994), so `p_value` is half of `p_naive`.
""",
        ),
        code("""
lr = run.load_table("lr")


def format_p(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2e}"


display(lr.assign(p_value=lr["p_value"].map(format_p), p_naive=lr["p_naive"].map(format_p)))
"""),
        md(
            pt="""
Todas as estatísticas são enormes (de 23.545 para M0 contra M1 a 322.868 para M1 contra M3), e todos os valores p \
são menores que 1e-300. Cada componente de variância (loja, produto e interação loja × produto) é, portanto, \
claramente diferente de zero. Com 347.092 linhas, porém, o teste tem poder para detectar qualquer variância, e o \
que importa é o tamanho dos componentes, mostrado a seguir.
""",
            en="""
All statistics are huge (from 23,545 for M0 against M1 to 322,868 for M1 against M3), and every p-value is below \
1e-300. Each variance component (store, product and store × product interaction) is therefore clearly different \
from zero. With 347,092 rows, however, the test has the power to detect any variance, and what matters is the size \
of the components, shown next.
""",
        ),
        md(
            pt="""
## 4. Correlação intraclasse (ICC)

O gráfico mostra a participação de cada componente na variância total de log(unidades) (Raudenbush & Bryk, 2002), \
por modelo de M1 a M4. \
Diferentemente da seção descritiva do notebook 01, as participações aqui vêm de um modelo e somam 1.
""",
            en="""
## 4. Intraclass correlation (ICC)

The chart shows the share of each component in the total variance of log(units) (Raudenbush & Bryk, 2002), for \
each model from M1 to M4. \
Unlike the descriptive section of notebook 01, the shares here come from a model and sum to 1.
""",
        ),
        code("""
icc = run.load_table("icc")
fig = viz.plot_icc_by_model(icc, LANG)
icc4 = icc[icc["model"] == "M4_crossed_pair"].reset_index(drop=True)
display(icc4)
"""),
        code("""
share = icc4.set_index("group")["share"]
derived = pd.Series(
    {
        t("corr_same_pair", LANG): share[["STORE_NUM", "UPC", "STORE_NUM:UPC"]].sum(),
        t("corr_same_store", LANG): share["STORE_NUM"],
    }
)
derived.to_frame("correlation")
"""),
        md(
            pt="""
No M4, o produto responde por 57,2% da variância, o resíduo por 24,6%, a loja por 9,6% e a interação loja × produto \
por 8,5%. A correlação entre duas semanas do mesmo par é (loja + produto + par) / total = 75,4%, e a correlação \
entre dois produtos diferentes na mesma loja é a participação da loja, 9,6%.

Do ponto de vista do negócio, a maior parte da variação no volume vem de qual produto se trata. Apenas o \
componente intra-par (resíduo), de cerca de 25%, é movimento semana a semana, e é nele que preço e promoção \
precisam atuar. Esse é o foco dos próximos notebooks. Os componentes mudam de um modelo para outro porque os \
modelos parciais atribuem ao resíduo a variância que os componentes omitidos explicariam (por exemplo, o resíduo \
cai de 93,3% em M1 para 24,6% em M4).
""",
            en="""
In M4, the product accounts for 57.2% of the variance, the residual for 24.6%, the store for 9.6% and the store × \
product interaction for 8.5%. The correlation between two weeks of the same pair is (store + product + pair) / \
total = 75.4%, and the correlation between two different products in the same store is the store share, 9.6%.

From a business point of view, most of the variation in volume comes from which product it is. Only the within-pair \
component (the residual), about 25%, is week-to-week movement, and that is where price and promotion have to act. \
This frames the next notebooks. The components change from one model to another because the partial models assign \
to the residual the variance that the omitted components would explain (for instance, the residual falls from \
93.3% in M1 to 24.6% in M4).
""",
        ),
        md(
            pt="""
## 5. Efeitos aleatórios previstos (BLUPs) do M4

Os gráficos mostram o intercepto aleatório previsto de cada loja e de cada produto, com intervalos de 95% \
baseados no desvio-padrão condicional. Os produtos são identificados pela descrição do cadastro.
""",
            en="""
## 5. Predicted random effects (BLUPs) of M4

The charts show the predicted random intercept of each store and each product, with 95% intervals based on the \
conditional standard deviation. Products are identified by their catalogue description.
""",
        ),
        code("""
fig = viz.plot_caterpillar(run.load_table("ranef_store"), LANG, "STORE_NUM")
"""),
        code("""
products = pd.read_parquet(PROCESSED_DIR / "panel.parquet", columns=["UPC", "DESCRIPTION"]).drop_duplicates("UPC")
labels = dict(zip(products["UPC"].astype(str), products["DESCRIPTION"], strict=True))
fig = viz.plot_caterpillar(run.load_table("ranef_product"), LANG, "UPC", labels=labels)
"""),
        md(
            pt="""
Os BLUPs (Henderson, 1975; Robinson, 1991) não são médias observadas: cada desvio observado do grupo é \
multiplicado por um fator de encolhimento \
(*shrinkage*), que num modelo de um único agrupamento é σ²ᵤ / (σ²ᵤ + σ²ₑ/n). Quanto menos observações um grupo \
tem, menor é esse fator e mais o seu valor é puxado para a média geral, e esse \
encolhimento, o *partial pooling*, é o tema central do projeto. Os intervalos são estreitos em relação à dispersão \
entre grupos (desvio-padrão condicional de cerca de 0,07 para lojas e 0,06 para produtos), o que indica que os \
grupos são bem distinguíveis entre si, em especial os produtos.
""",
            en="""
BLUPs (Henderson, 1975; Robinson, 1991) are not observed means: each group's observed deviation is multiplied by a \
shrinkage factor, which in a \
single-grouping model is σ²ᵤ / (σ²ᵤ + σ²ₑ/n). The fewer observations a group has, the smaller this factor and the \
more its value is pulled toward the overall mean, and this \
shrinkage, partial pooling, is the central theme of the project. The intervals are narrow relative to the spread \
between groups (conditional standard deviation of about 0.07 for stores and 0.06 for products), which indicates \
that the groups are clearly distinguishable from each other, especially the products.
""",
        ),
        md(
            pt="""
## 6. Verificação cruzada com o lme4

O modelo M1 (efeito aleatório só de loja) foi ajustado também com o `MixedLM` do statsmodels (Seabold & Perktold, \
2010), uma implementação \
independente em Python, e comparado ao lme4 (R; Bates et al., 2015).
""",
            en="""
## 6. Cross-check against lme4

Model M1 (random effect of store only) was also fitted with statsmodels `MixedLM` (Seabold & Perktold, 2010), an \
independent implementation \
in Python, and compared with lme4 (R; Bates et al., 2015).
""",
        ),
        code("""
display(run.load_table("crosscheck"))
"""),
        md(
            pt="""
A variância residual difere em 1,3e-6 (diferença relativa), e a variância de loja difere em 0,6% (0,1001 contra \
0,1007). O statsmodels só foi usado nesta verificação de um único agrupamento. O `MixedLM` não tem sintaxe direta \
para efeitos cruzados (eles exigem um grupo único com `vc_formula`), e por isso o lme4 foi escolhido como motor \
dos demais modelos (decisão D1). O desempenho do `MixedLM` com efeitos cruzados neste volume não foi testado.
""",
            en="""
The residual variance differs by 1.3e-6 (relative difference), and the store variance differs by 0.6% (0.1001 \
against 0.1007). Statsmodels was used only in this single-grouping check. `MixedLM` has no direct syntax for \
crossed effects (they require a single group with `vc_formula`), so lme4 was chosen as the engine for the \
remaining models (decision D1). The performance of `MixedLM` with crossed effects at this size was not tested.
""",
        ),
        md(
            pt="""
## 7. Conclusões e próximo passo

A variação de log(unidades) está sobretudo entre produtos (57,2% no M4); lojas respondem por 9,6%, a interação \
loja × produto por 8,5% e o resíduo, por 24,6%. Todos os componentes são significativos pelo teste de razão de \
verossimilhanças com correção de fronteira, mas o relevante é o seu tamanho, não o valor p.

Essa estrutura se traduz em correlações: duas semanas do mesmo par são correlacionadas em 75,4%, enquanto produtos \
diferentes na mesma loja o são em 9,6%. Por fim, o ajuste do statsmodels coincide com o do lme4 no modelo de um \
único agrupamento.

O notebook 03 acrescentará os efeitos fixos de preço e promoção. Depois, serão estudadas inclinações aleatórias, \
cuja especificação exige cuidado no `MixedLM`. Com `groups='g'` e `re_formula='x'`, um componente \
`vc_formula={'h': '0 + x'}` não cria uma inclinação por nível de `h`: ele gera a coluna `x` dentro de cada grupo \
`g`, isto é, uma segunda variância de inclinação no nível de `g`, duplicada da que `re_formula` já estima, e não \
inclui intercepto por `h`. O resultado são componentes não identificáveis, que tendem a falhar na convergência ou a \
ter variância nula. A especificação por nível de `h` seria `vc_formula={'h': '0 + C(h)', 'h_x': '0 + C(h):x'}`, e no \
lme4 seria `(1 + x | g) + (1 + x | g:h)`. Neste projeto, as inclinações aleatórias foram especificadas diretamente \
no lme4.
""",
            en="""
## 7. Conclusions and next step

The variation in log(units) lies mostly between products (57.2% in M4); stores account for 9.6%, the store × \
product interaction for 8.5% and the residual for 24.6%. All components are significant under the \
boundary-corrected likelihood-ratio test, but what matters is their size, not the p-value.

This structure translates into correlations: two weeks of the same pair are correlated at 75.4%, whereas different \
products in the same store are correlated at 9.6%. Finally, the statsmodels fit agrees with lme4 on the \
single-grouping model.

Notebook 03 will add the fixed effects of price and promotion. Random slopes will be studied afterwards, and their \
specification needs care in `MixedLM`. With `groups='g'` and `re_formula='x'`, a component \
`vc_formula={'h': '0 + x'}` does not create a slope per level of `h`: it generates the column `x` within each group \
`g`, that is, a second slope variance at the level of `g`, duplicating the one `re_formula` already estimates, and \
it includes no intercept per `h`. The result is unidentifiable components, which tend to fail to converge or to get \
zero variance. The specification per level of `h` would be `vc_formula={'h': '0 + C(h)', 'h_x': '0 + C(h):x'}`, \
and in lme4 it would be `(1 + x | g) + (1 + x | g:h)`. In this project, random slopes were specified directly in \
lme4.
""",
        ),
    ]


SETUP_03 = """
import warnings

import pandas as pd
from IPython.display import display

from retail_multilevel import interpret, viz
from retail_multilevel.i18n import t
from retail_multilevel.paths import PROCESSED_DIR
from retail_multilevel.runs import load_run
from retail_multilevel.split import model_sample

warnings.filterwarnings("ignore", category=FutureWarning)
viz.use_retina()
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:,.4f}".format)
run = load_run("fixed_effects", "20261124-114202-1fda5a5d")
print(t("data_credit", LANG))
print("run:", run.run_id, "| finished_at:", run.metadata["finished_at"])
sample = run.load_json("sample")
display(pd.Series(sample).to_frame("training sample"))
"""


def fixed_effects_cells() -> list[Cell]:
    """Cells of notebook 03, the fixed effects of price and promotion read from the stored runs."""
    return [
        md(
            pt="""
# 03 · Efeitos fixos de preço e promoção

Este é o segundo passo da seleção de modelos por etapas (*step-up*; Raudenbush & Bryk, 2002). Depois do modelo \
nulo do notebook 02, preditores \
do nível 1 \
(a semana) são acrescentados à parte fixa, e os interceptos aleatórios de loja, de produto e do par loja × produto \
(o M4) são mantidos. Somente a amostra de treino foi usada para ajustar os modelos: as 104 semanas iniciais (D11 \
revisada). As 52 semanas finais formam uma janela de avaliação temporal retrospectiva com exposição exploratória \
prévia, e nenhum dos ajustes deste notebook as utilizou.

Este notebook **não ajusta modelos**: os ajustes foram feitos por scripts (`scripts/fit_fixed_effects.py` e os das \
sensibilidades) e gravados em `outputs/runs`. O notebook apenas lê essas execuções, o que torna as versões em \
português e em inglês idênticas em resultados. Os coeficientes apresentados são associações condicionais em \
semanas com transações registradas, e não efeitos causais.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 03 · Fixed effects of price and promotion

This is the second step of step-up model selection (Raudenbush & Bryk, 2002). After the null model of notebook 02, \
level-1 predictors \
(the week) are added to the fixed part, while the random intercepts for store, product and the store × product \
pair (M4) are kept. Only the training sample was used to fit the models: the initial 104 weeks (D11, revised). The \
final 52 weeks form a retrospective temporal evaluation window with prior exploratory exposure, and none of the \
fits in this notebook used them.

This notebook **fits no models**: the fits were made by scripts (`scripts/fit_fixed_effects.py` and those of the \
sensitivity analyses) and stored in `outputs/runs`. The notebook only reads those runs, so the Portuguese and \
English versions show identical results. The coefficients presented are conditional associations in weeks with \
recorded transactions, not causal effects.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_03),
        md(
            pt="""
A execução lida está identificada acima (identificador e horário de término). A amostra de treino tem 347.092 \
linhas e 3.859 pares loja × produto. O log do desconto (`LOG_DISC`, a seguir) tem média de −0,069 e desvio-padrão \
de 0,147, e 25,0% das semanas têm `LOG_DISC` abaixo de −0,05, isto é, um desconto de cerca de 5% ou mais sobre o \
preço regular. Há, portanto, muita variação de desconto de onde extrair a associação. Das linhas de treino, 111 \
usam o preço-base de fallback (`BASE_PRICE = PRICE`), tratadas na seção 8.
""",
            en="""
The run that was read is identified above (identifier and finish time). The training sample has 347,092 rows and \
3,859 store × product pairs. The log of the discount (`LOG_DISC`, below) has a mean of −0.069 and a standard \
deviation of 0.147, and 25.0% of the weeks have `LOG_DISC` below −0.05, that is, a discount of about 5% or more \
off the regular price. There is, therefore, plenty of discount variation from which to extract the association. Of \
the training rows, 111 use the fallback base price (`BASE_PRICE = PRICE`), which section 8 addresses.
""",
        ),
        md(
            pt="""
## 1. A especificação

A resposta é log(unidades), de modo que cada coeficiente é uma variação logarítmica, convertida em variação \
percentual mais adiante. A parte fixa foi construída com base nas decisões D12 a D14, que são decisões de \
desenvolvimento, motivadas pela exploração inicial do painel (D11 revisada). O preço não entra como um único \
termo, porque o log do preço varia sobretudo entre produtos, e essa diferença se confunde com as características \
do produto. Por isso, o log do preço é decomposto em três partes. A primeira é a profundidade do desconto \
temporário, `LOG_DISC = log(preço / preço base)`, cujo coeficiente é a elasticidade ao desconto (Tellis, 1988; \
Bijmolt et al., 2005). A segunda é o \
desvio do preço regular em relação à média do próprio par, `LOG_BASE_DEV`, cujo coeficiente é a elasticidade ao \
preço regular dentro do par ao longo do tempo. A terceira é a própria média do par, `LOG_BASE_MEAN` (um termo de \
Mundlak), que absorve a associação entre o nível de preço e o par e impede que essa diferença entre pares \
contamine os coeficientes dentro do par (Mundlak, 1978). As médias de par foram calculadas só com o treino, e o \
preço-base ausente \
foi preenchido apenas com o histórico do par (D8 revisada), para que nada posterior ao instante da observação \
entre na especificação.

Somente o preço regular foi decomposto em termos dentro e entre pares. O desconto e as variáveis de promoção \
entram como estão, e a leitura dentro do par desses regressores é examinada por uma sensibilidade com efeitos \
fixos de par e de semana (seção 9). As três variáveis de promoção (`FEATURE`, `DISPLAY` e `TPR_ONLY`) são \
indicadores de 0 ou 1. Como o desconto já entra pela sua profundidade, esses indicadores medem a associação \
**além** da redução de preço, com o desconto mantido constante. Essa separação resolve em parte a quase \
colinearidade entre desconto e promoção observada na limpeza dos dados, porque o preço só fica abaixo do base em \
semanas com promoção, mas a sobreposição entre tipos de promoção e faixas de desconto continua limitada (seção 4). \
A sazonalidade é representada por três pares de termos de Fourier (Hyndman & Athanasopoulos, 2021) da semana do \
ano (`SIN_k` e `COS_k`, com k de 1 \
a 3), globais, e a tendência por uma variável linear em anos (`TREND`). Dessa forma, evitam-se 52 variáveis \
binárias de semana, pesadas e ruidosas para dois anos de treino.

Os testes de razão de verossimilhanças (LR) que comparam parte fixa diferente usam ajustes por **máxima \
verossimilhança (ML)**, porque a verossimilhança restrita (REML) depende da parte fixa e, por isso, não é \
comparável entre esses modelos (Pinheiro & Bates, 2000). Os coeficientes e as variâncias do modelo final, por sua \
vez, vêm de um novo \
ajuste por REML (Bates et al., 2015), que estima as variâncias com menos viés. Os quatro modelos, de F0 a F3, são \
mostrados a seguir.
""",
            en="""
## 1. The specification

The response is log(units), so each coefficient is a log-change, converted to a percentage change further on. The \
fixed part was built from decisions D12 to D14, which are development decisions motivated by the initial \
exploration of the panel (D11, revised). Price does not enter as a single term, because the log price varies \
mostly between products, and that difference is entangled with the characteristics of the product. The log price \
is therefore decomposed into three parts. The first is the depth of the temporary discount, `LOG_DISC = log(price \
/ base price)`, whose coefficient is the elasticity to the discount (Tellis, 1988; Bijmolt et al., 2005). The \
second is the deviation of the regular \
price from the mean of its own pair, `LOG_BASE_DEV`, whose coefficient is the elasticity to the regular price \
within the pair over time. The third is the pair mean itself, `LOG_BASE_MEAN` (a Mundlak term), which absorbs the \
association between the price level and the pair and keeps this difference between pairs from contaminating the \
within-pair coefficients (Mundlak, 1978). The pair means were computed on the training rows only, and a missing \
base price was \
filled using only the history of the pair (D8, revised), so that nothing posterior to the time of the observation \
enters the specification.

Only the regular price was decomposed into within-pair and between-pair terms. The discount and the promotion \
variables enter as they are, and the within-pair reading of those regressors is examined by a sensitivity analysis \
with pair and week fixed effects (section 9). The three promotion variables (`FEATURE`, `DISPLAY` and `TPR_ONLY`) \
are 0/1 indicators. Because the discount already enters through its depth, these indicators measure the \
association **beyond** the price cut, with the discount held constant. This separation partly resolves the \
near-collinearity between discount and promotion seen in the data cleaning, because the price only falls below \
base in weeks with a promotion, but the overlap between promotion types and discount bands remains limited \
(section 4). Seasonality is represented by three pairs of Fourier terms (Hyndman & Athanasopoulos, 2021) of the \
week of the year (`SIN_k` and \
`COS_k`, with k from 1 to 3), global, and the trend by a linear variable in years (`TREND`). In this way, 52 \
weekly dummy variables, heavy and noisy for two years of training, are avoided.

The likelihood-ratio (LR) tests that compare different fixed parts use **maximum likelihood (ML)** fits, because \
the restricted likelihood (REML) depends on the fixed part and is therefore not comparable across these models \
(Pinheiro & Bates, 2000). \
The coefficients and variances of the final model, in turn, come from a new REML fit (Bates et al., 2015), which \
estimates the \
variances with less bias. The four models, F0 to F3, are shown next.
""",
        ),
        code("""
fits = run.load_table("fits")
display(fits[["name", "formula"]])
"""),
        md(
            pt="""
## 2. O step-up dos efeitos fixos

Cada passo acrescenta um bloco de termos à parte fixa: o preço (F1), as promoções (F2) e a sazonalidade com \
tendência (F3). A tabela traz o número de parâmetros, a log-verossimilhança ML, o AIC, o BIC e o número de \
mensagens de convergência.
""",
            en="""
## 2. The fixed-effects step-up

Each step adds a block of terms to the fixed part: price (F1), promotions (F2) and seasonality with trend (F3). \
The table gives the number of parameters, the ML log-likelihood, the AIC, the BIC and the number of convergence \
messages.
""",
        ),
        code("""
display(fits[["name", "nobs", "npar", "loglik", "aic", "bic", "n_messages"]])
"""),
        md(
            pt="""
A log-verossimilhança ML sobe de −346.785 (F0) para −295.885 (F1), −275.145 (F2) e −274.532 (F3), e o AIC cai de \
693.581 para 591.787, 550.312 e 549.101, com 5, 8, 11 e 18 parâmetros. O BIC também diminui a cada passo (de \
693.635 para 549.294), de modo que a penalização por parâmetros adicionais não reverte nenhum dos ganhos. Nenhum \
ajuste gerou mensagens de convergência.
""",
            en="""
The ML log-likelihood rises from −346,785 (F0) to −295,885 (F1), −275,145 (F2) and −274,532 (F3), and the AIC \
falls from 693,581 to 591,787, 550,312 and 549,101, with 5, 8, 11 and 18 parameters. The BIC also decreases at \
every step (from 693,635 to 549,294), so the penalty for additional parameters does not reverse any of the gains. \
No fit produced convergence messages.
""",
        ),
        code("""
lr = run.load_table("lr")


def format_p(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2e}"


display(lr.assign(p_value=lr["p_value"].map(format_p), p_naive=lr["p_naive"].map(format_p)))
"""),
        md(
            pt="""
Aqui, ao contrário do notebook 02, os testes não envolvem variâncias na fronteira do espaço de parâmetros: são \
efeitos fixos, e a referência é um χ² comum, razão pela qual `p_value` e `p_naive` coincidem. As estatísticas são \
de 101.800 (F0 contra F1, 3 graus de liberdade), 41.481 (F1 contra F2, 3 graus) e 1.225 (F2 contra F3, 7 graus). \
Os dois primeiros valores p são menores que 1e-300 e o terceiro é de 2,69e-260.

Com 347.092 linhas, porém, qualquer termo teria valor p desprezível, de modo que a significância, por si só, \
informa pouco. O critério relevante é a magnitude: o bloco de preço é o de maior contribuição (101.800), seguido \
pelo de promoções (41.481), enquanto a sazonalidade, com 1.225, é cerca de 83 vezes menor que a do preço.
""",
            en="""
Here, unlike in notebook 02, the tests do not involve variances on the boundary of the parameter space: they \
concern fixed effects, and the reference is an ordinary χ², which is why `p_value` and `p_naive` coincide. The \
statistics are 101,800 (F0 against F1, 3 degrees of freedom), 41,481 (F1 against F2, 3 degrees) and 1,225 (F2 \
against F3, 7 degrees). The first two p-values are below 1e-300 and the third is 2.69e-260.

With 347,092 rows, however, any term would have a negligible p-value, so significance alone says little. The \
relevant criterion is magnitude: the price block makes the largest contribution (101,800), followed by the \
promotion block (41,481), while seasonality, at 1,225, is about 83 times smaller than price.
""",
        ),
        md(
            pt="""
## 3. Os coeficientes e o que significam

O gráfico mostra os coeficientes do modelo final (F3, por REML) com intervalos de 95% de Wald, na escala \
logarítmica, sem o intercepto e sem os termos de Fourier. Os intervalos são tão estreitos que mal se distinguem do \
ponto, mas são condicionais à especificação do F3 e subestimam a incerteza, como mostram as seções 9 e 10.
""",
            en="""
## 3. The coefficients and what they mean

The chart shows the coefficients of the final model (F3, by REML) with 95% Wald intervals, on the log scale, \
without the intercept and the Fourier terms. The intervals are so narrow that they are barely distinguishable from \
the point, but they are conditional on the F3 specification and understate the uncertainty, as sections 9 and 10 \
show.
""",
        ),
        code("""
fixef = run.load_table("fixef")
fig = viz.plot_coefficients(fixef, LANG)
display(fixef[~fixef["term"].str.startswith(("SIN_", "COS_"))])
"""),
        md(
            pt="""
Como a resposta está em log, um coeficiente de log-desconto é uma elasticidade, e a tabela a seguir converte cada \
coeficiente (e os seus limites de confiança, pela mesma transformação) em variação percentual das unidades: uma \
redução de 10% no preço dentro do desconto, um aumento de 10% no preço regular, e a presença de cada promoção ou \
de mais um ano. Cada linha mantém constantes as demais covariáveis.
""",
            en="""
Because the response is in logs, a log-discount coefficient is an elasticity, and the following table converts \
each coefficient (and its confidence limits, through the same transformation) into a percentage change in units: a \
10% price cut through the discount, a 10% rise in the regular price, and the presence of each promotion or of one \
more year. Each row holds the other covariates constant.
""",
        ),
        code("""
with pd.option_context("display.float_format", "{:,.1f}".format):
    display(interpret.effects_table(fixef, LANG))
"""),
        md(
            pt="""
O coeficiente do desconto é −1,685 (IC 95% de Wald: −1,706 a −1,664), de modo que um desconto 10% mais profundo \
está associado a cerca de 19,4% mais unidades (IC 95%: 19,2% a 19,7%). O preço regular, medido dentro do par, tem \
coeficiente −1,592, e um preço regular 10% maior está associado a cerca de 14,1% menos unidades. O encarte tem \
coeficiente 0,497, ou seja, cerca de 64,4% mais unidades, e o display, 0,409, ou cerca de 50,5% mais, ambos com o \
desconto e as demais covariáveis mantidos constantes. A tendência é de −0,021 por ano, cerca de 2,1% menos \
unidades a cada ano.

**TPR.** O coeficiente de `TPR_ONLY` é −0,041, ou seja, −4,0% nas unidades (IC 95%: −4,7% a −3,3%). Esse valor é a \
transformação do coeficiente da flag com a profundidade do desconto e as demais covariáveis mantidas constantes, \
isto é, a comparação entre duas semanas com o mesmo desconto, uma marcada como TPR e outra sem promoção. Essa \
comparação praticamente não existe nos dados de treino: as 47.610 linhas `TPR_ONLY` têm todas desconto (nenhuma \
abaixo de 5%, como mostra a tabela cruzada da seção 4), e nenhuma das 247.331 linhas sem promoção tem preço abaixo \
do base. Comparar TPR com uma semana sem promoção na mesma faixa de desconto é, portanto, uma extrapolação, que \
depende da forma funcional aditiva do modelo. O coeficiente, por si só, não permite afirmar que a TPR reduz as \
vendas nem que o seu valor vem apenas do desconto. A leitura de negócio exige cenários que combinem a profundidade \
do desconto e o tipo de promoção, apresentados na seção 4.

Os coeficientes de encarte e de display contrastam com os lifts brutos (Blattberg & Neslin, 1989) do notebook 01 \
(encarte +195%, display \
+129% e TPR +33%). Naquele notebook, as comparações eram descritivas e misturavam a promoção com o corte de preço \
que a acompanha. Aqui, a profundidade do desconto, o nível de preço regular, a tendência e a sazonalidade são \
controlados, e o que resta para o encarte (+64%) e o display (+50%) é uma fração do lift bruto: uma parte do que \
parecia efeito da promoção acompanha o corte de preço.

O coeficiente de `LOG_BASE_MEAN` (−0,880, IC 95%: −1,086 a −0,673) merece uma leitura distinta. Ele compara pares \
com preços regulares médios diferentes e, por isso, é uma associação **entre pares**, de natureza composicional \
(pares com preço regular médio mais alto tendem a vender menos unidades por semana), e não o efeito de alterar o \
preço de um item.
""",
            en="""
The discount coefficient is −1.685 (95% Wald CI: −1.706 to −1.664), so a 10% deeper discount is associated with \
about 19.4% more units (95% CI: 19.2% to 19.7%). The regular price, measured within the pair, has a coefficient of \
−1.592, and a 10% higher regular price is associated with about 14.1% fewer units. Feature has a coefficient of \
0.497, that is, about 64.4% more units, and display 0.409, or about 50.5% more, both with the discount and the \
other covariates held constant. The trend is −0.021 per year, about 2.1% fewer units each year.

**TPR.** The coefficient of `TPR_ONLY` is −0.041, that is, −4.0% in units (95% CI: −4.7% to −3.3%). This value is \
the transformation of the flag coefficient with the depth of the discount and the other covariates held constant, \
that is, the comparison between two weeks with the same discount, one flagged as TPR and one with no \
promotion. That comparison practically does not exist in the training data: all 47,610 `TPR_ONLY` rows have a \
discount (none below 5%, as the cross-tabulation of section 4 shows), and none of the 247,331 rows without a \
promotion has a price below base. Comparing TPR with a week without promotion in the same discount band is \
therefore an extrapolation, which depends on the additive functional form of the model. The coefficient alone does \
not allow stating that TPR reduces sales or that its value comes only from the discount. The business reading \
requires scenarios that combine the depth of the discount and the promotion type, presented in section 4.

The feature and display coefficients contrast with the raw lifts (Blattberg & Neslin, 1989) of notebook 01 \
(feature +195%, display +129% and \
TPR +33%). In that notebook, the comparisons were descriptive and mixed the promotion with the price cut that \
accompanies it. Here, the depth of the discount, the regular price level, the trend and seasonality are \
controlled, and what remains for feature (+64%) and display (+50%) is a fraction of the raw lift: a part of what \
looked like a promotion effect accompanies the price cut.

The coefficient of `LOG_BASE_MEAN` (−0.880, 95% CI: −1.086 to −0.673) deserves a different reading. It compares \
pairs with different average regular prices and is therefore an association **between pairs**, compositional in \
nature (pairs with a higher average regular price tend to sell fewer units per week), and not the effect of \
changing the price of an item.
""",
        ),
        md(
            pt="""
## 4. Cenários conjuntos de desconto e promoção

Como o desconto e o tipo de promoção variam juntos nos dados, a leitura de negócio se apoia em cenários que \
combinam os dois, e não em contrastes isolados de um coeficiente. A tabela cruzada a seguir mostra a sobreposição \
nas linhas de treino: o número de linhas de cada tipo de promoção em cada faixa de desconto (`1 − PRICE / \
BASE_PRICE`).
""",
            en="""
## 4. Joint discount-and-promotion scenarios

Because the discount and the promotion type vary together in the data, the business reading rests on scenarios \
that combine the two, not on isolated contrasts of one coefficient. The cross-tabulation below shows the overlap \
in the training rows: the number of rows of each promotion type in each discount band (`1 − PRICE / BASE_PRICE`).
""",
        ),
        code("""
panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
train = model_sample(panel, "train")
display(interpret.promo_by_discount_band(train, LANG))
"""),
        md(
            pt="""
Nenhuma das 247.331 linhas sem promoção tem desconto, e nenhuma das 47.610 linhas de TPR tem desconto de até 5%: \
155 ficam entre 5% e 10%, e a maior parte, 26.592, entre 10% e 20%. Encarte e display ocorrem, ao contrário, em \
todas as faixas, inclusive sem redução de preço (793 linhas só de encarte e 8.489 só de display).

Cada cenário abaixo é medido em relação a uma semana regular, sem desconto e sem promoção, por `exp(β_desconto · \
log(1 − d) + β_promoção) − 1`, com as demais covariáveis mantidas constantes. As duas últimas colunas contam as \
linhas de treino que realizam exatamente a combinação de promoção do cenário com desconto a até 3 pontos \
percentuais do valor do cenário, o que indica se o cenário está dentro das faixas observadas. Os valores são \
estimativas pontuais: a tabela de coeficientes não guarda as covariâncias, e os intervalos individuais não podem \
ser combinados.
""",
            en="""
None of the 247,331 rows without a promotion has a discount, and none of the 47,610 TPR rows has a discount of up \
to 5%: 155 fall between 5% and 10%, and most of them, 26,592, between 10% and 20%. Feature and display, by \
contrast, occur in every band, including with no price cut (793 feature-only rows and 8,489 display-only rows).

Each scenario below is measured relative to a regular week, with no discount and no promotion, by `exp(β_discount \
· log(1 − d) + β_promotion) − 1`, with the other covariates held constant. The last two columns count the training \
rows that realise exactly the promotion combination of the scenario with a discount within 3 percentage points of \
the scenario value, which indicates whether the scenario lies within the observed ranges. The values are point \
estimates: the coefficient table does not store the covariances, and the individual intervals cannot be combined.
""",
        ),
        code("""
scenarios = interpret.DEFAULT_SCENARIOS
scenario_tab = interpret.scenario_table(
    interpret.scenario_effects(fixef, scenarios), interpret.scenario_support(train, scenarios, tol=0.03), LANG
)
with pd.option_context("display.float_format", "{:,.1f}".format):
    display(scenario_tab)
"""),
        md(
            pt="""
Uma TPR com 10% de desconto está associada a 14,6% mais unidades que a semana regular (9.393 linhas de treino, \
2,7%), e com 20% de desconto a 39,8% mais (10.185 linhas, 2,9%); os dois cenários estão dentro das faixas \
observadas. O encarte com 20% de desconto (+139,5%, 2.574 linhas), o display com 20% (+119,2%, 2.272 linhas) e o \
encarte com display e 25% de desconto (+301,8%, 2.767 linhas) também têm suporte, embora pequeno, de 0,7% a 0,8% \
das linhas cada um. Sem desconto, o display (+50,5%, 8.622 linhas, 2,5%) e o encarte (+64,4%, 1.055 linhas, 0,3%) \
têm o suporte mais baixo entre os cenários, sobretudo o encarte.

Os valores dependem da forma funcional. A elasticidade é constante em log(1 − d), as flags e o desconto são \
aditivos na escala logarítmica, e encarte e display não têm interação estimada: o valor de 301,8% para encarte com \
display é o produto dos efeitos estimados em separado (e^0,497 × e^0,409 × 0,75^−1,685 − 1), embora haja linhas \
dessa combinação em todas as faixas de desconto. O contraste entre TPR e display com o mesmo desconto de 20% \
(+39,8% contra +119,2%) usa linhas observadas nos dois cenários, mas continua sendo uma associação condicional, \
sujeita à seleção de quando e onde cada promoção é aplicada. Nenhum ranking de rentabilidade decorre desses \
números: a margem e o custo das promoções não estão nos dados, e uma recomendação de preço ou um ROI incremental \
exigiria identificação causal e dados de margem. A camada de decisão do projeto é, por isso, uma análise de \
cenários.
""",
            en="""
A TPR with a 10% discount is associated with 14.6% more units than the regular week (9,393 training rows, 2.7%), \
and with a 20% discount with 39.8% more (10,185 rows, 2.9%); both scenarios lie within the observed ranges. \
Feature with a 20% discount (+139.5%, 2,574 rows), display with 20% (+119.2%, 2,272 rows) and feature with display \
at a 25% discount (+301.8%, 2,767 rows) also have support, although small, of 0.7% to 0.8% of the rows each. \
Without a discount, display (+50.5%, 8,622 rows, 2.5%) and feature (+64.4%, 1,055 rows, 0.3%) have the lowest \
support among the scenarios, especially feature.

The values depend on the functional form. The elasticity is constant in log(1 − d), the flags and the discount are \
additive on the log scale, and feature and display have no estimated interaction: the 301.8% for feature with \
display is the product of the separately estimated effects (e^0.497 × e^0.409 × 0.75^−1.685 − 1), although there \
are rows of that combination in every discount band. The contrast between TPR and display at the same 20% discount \
(+39.8% against +119.2%) uses rows observed in both scenarios, but remains a conditional association, subject to \
the selection of when and where each promotion is applied. No profitability ranking follows from these numbers: \
the margin and the cost of promotions are not in the data, and a pricing recommendation or an incremental ROI \
would require causal identification and margin data. The decision layer of the project is therefore a scenario \
analysis.
""",
        ),
        md(
            pt="""
## 5. Sazonalidade e tendência

Os seis termos de Fourier não têm leitura direta individualmente. O perfil sazonal que eles implicam é calculado a \
seguir, como efeito percentual sobre as unidades em relação à semana média do ano.
""",
            en="""
## 5. Seasonality and trend

The six Fourier terms have no direct reading individually. The seasonal profile they imply is computed next, as a \
percentage effect on units relative to the average week of the year.
""",
        ),
        code("""
profile = interpret.seasonal_profile(fixef)
fig = viz.plot_seasonality(profile, LANG)
display(profile.loc[[profile["effect_pct"].idxmin(), profile["effect_pct"].idxmax()]])
display(fixef[fixef["term"].str.startswith(("SIN_", "COS_"))])
"""),
        md(
            pt="""
A amplitude da sazonalidade é pequena frente às associações de preço e de promoção: a semana mais fraca do ano (a \
semana 29) tem cerca de 4,2% menos unidades que a semana média, e a mais forte (a semana 2), cerca de 6,6% mais. A \
diferença entre os extremos é de cerca de 11 pontos percentuais, a ser comparada ao valor de 19,4% de um desconto \
10% mais profundo. Alguns termos individuais têm coeficientes pequenos (por exemplo, `SIN_1`, −0,002, cujo \
intervalo inclui o zero), mas o bloco como um todo é claramente diferente de zero pelo teste de razão de \
verossimilhanças (1.225). A tendência, por fim, é de uma queda de cerca de 2,1% por ano (IC 95%: −2,5% a −1,8%) \
nas duas temporadas de treino, após o controle de preço, promoção e sazonalidade.
""",
            en="""
The amplitude of seasonality is small compared with the price and promotion associations: the weakest week of the \
year (week 29) has about 4.2% fewer units than the average week, and the strongest (week 2) about 6.6% more. The \
difference between the extremes is about 11 percentage points, to be compared with the 19.4% associated with a 10% \
deeper discount. Some individual terms have small coefficients (for example, `SIN_1`, −0.002, whose interval \
includes zero), but the block as a whole is clearly different from zero by the likelihood-ratio test (1,225). The \
trend, finally, is a decline of about 2.1% per year (95% CI: −2.5% to −1.8%) over the two training seasons, after \
controlling for price, promotion and seasonality.
""",
        ),
        md(
            pt="""
## 6. Quanto da variância preço e promoção explicam: pseudo-R² por nível

Em modelos multinível, a redução proporcional de cada componente de variância, do modelo nulo para o modelo final, \
informa em que nível os preditores atuam (Raudenbush & Bryk, 2002). O cálculo usa os ajustes por REML de F0 e F3, \
e a redução é 1 − \
variância final / variância do modelo nulo.
""",
            en="""
## 6. How much variance price and promotion explain: pseudo-R² per level

In multilevel models, the proportional reduction of each variance component, from the null model to the final \
model, indicates the level at which the predictors act (Raudenbush & Bryk, 2002). The calculation uses the REML \
fits of F0 and F3, and the \
reduction is 1 − final variance / null-model variance.
""",
        ),
        code("""
pseudo = run.load_table("pseudo_r2")
fig = viz.plot_pseudo_r2(pseudo, LANG)
display(pseudo.assign(group=pseudo["group"].map(lambda g: viz.group_label(g, LANG))))
"""),
        md(
            pt="""
O resíduo, que representa o movimento semana a semana dentro do par, cai 34,4% (de 0,415 para 0,272), de modo que \
preço e promoção explicam cerca de um terço da variância semanal dentro do par. A variância de produto cai 29,3% \
(de 0,965 para 0,682): como o nível de preço regular distingue os produtos, isso é consistente com o preço médio \
do par explicando parte da diferença entre produtos. Já as variâncias de loja (−2,6%) e de par (−4,2%) aumentam \
levemente.

Esses valores negativos pequenos são um artefato conhecido do pseudo-R² em modelos multinível: quando um preditor \
absorve variância em um nível, a variância pode ser redistribuída entre os níveis, e o componente de outro nível \
pode crescer um pouco sem que isso indique um modelo pior. Não se deve interpretá-los como perda de informação, \
apenas como indicação de que preço e promoção atuam sobretudo no resíduo e no produto, e muito pouco na loja e no \
par.
""",
            en="""
The residual, which represents the week-to-week movement within the pair, falls by 34.4% (from 0.415 to 0.272), so \
price and promotion explain about a third of the weekly variance within the pair. The product variance falls by \
29.3% (from 0.965 to 0.682): because the regular price level distinguishes products, this is consistent with the \
average price of the pair explaining part of the difference between products. The store (−2.6%) and pair (−4.2%) \
variances, in contrast, increase slightly.

These small negative values are a known artefact of pseudo-R² in multilevel models: when a predictor absorbs \
variance at one level, the variance can be reallocated between levels, and the component of another level can grow \
a little without indicating a worse model. They should not be read as a loss of information, only as an indication \
that price and promotion act mostly on the residual and on the product, and very little on the store and on the \
pair.
""",
        ),
        md(
            pt="""
## 7. Nível único contra multinível

A mesma parte fixa do F3 foi ajustada de duas formas de nível único (Gelman & Hill, 2007): um OLS agrupado, que \
ignora a hierarquia, e \
um OLS com efeitos fixos de produto (variáveis binárias de UPC). O gráfico compara as estimativas dos quatro \
coeficientes mais relevantes (ponto ± 1,96 erro-padrão) nos três métodos, e a tabela traz os valores.
""",
            en="""
## 7. Single-level versus multilevel

The same fixed part as F3 was fitted in two single-level ways (Gelman & Hill, 2007): a pooled OLS, which ignores \
the hierarchy, and an \
OLS with product fixed effects (UPC dummy variables). The chart compares the estimates of the four most relevant \
coefficients (point ± 1.96 standard errors) across the three methods, and the table gives the values.
""",
        ),
        code("""
comparison = interpret.method_comparison_table(run.load_table("ols_pooled"), fixef)
fig = viz.plot_method_comparison(comparison, LANG)
estimates = comparison.pivot(index="term", columns="method", values="estimate")
display(estimates.loc[viz.COMPARISON_TERMS, ["ols_pooled", "ols_upc_fe", "multilevel"]])
"""),
        md(
            pt="""
A elasticidade ao desconto é de −0,359 no OLS agrupado, −1,405 no OLS com efeitos fixos de produto e −1,685 no \
multinível. O OLS agrupado atenua a elasticidade em cerca de 4,7 vezes e, ao mesmo tempo, infla os efeitos de \
encarte (0,908 contra 0,497 no multinível) e de display (0,651 contra 0,409). O OLS com efeitos fixos de produto \
se aproxima do multinível no desconto (−1,405) e no encarte (0,454), mas ainda diverge no display (0,551) e no \
preço regular dentro do par (−1,459 contra −1,592). Nenhum dos intervalos dos dois OLS contém a estimativa \
multinível do desconto, e esses intervalos, além disso, supõem linhas independentes, hipótese que o próprio \
notebook 02 mostrou ser inadequada (correlação de 75,4% entre semanas do mesmo par).

Esses resultados são consistentes com a ideia de que o OLS agrupado mistura variação entre pares e variação dentro \
do par: se descontos e promoções se concentram em pares cujos níveis de vendas já diferem, essa composição se \
confunde com a associação do desconto. Essa é uma explicação compatível com os números, e não uma demonstração. O \
que os números mostram, de forma concreta, é que ignorar a hierarquia altera a resposta: a elasticidade ao \
desconto estimada, a base de qualquer análise de preço, muda de −0,36 para −1,69 conforme o método.
""",
            en="""
The elasticity to the discount is −0.359 in the pooled OLS, −1.405 in the OLS with product fixed effects and \
−1.685 in the multilevel model. The pooled OLS attenuates the elasticity by about a factor of 4.7 and, at the same \
time, inflates the effects of feature (0.908 against 0.497 in the multilevel model) and display (0.651 against \
0.409). The OLS with product fixed effects comes closer to the multilevel model on the discount (−1.405) and \
feature (0.454), but still diverges on display (0.551) and on the within-pair regular price (−1.459 against \
−1.592). None of the intervals of the two OLS models contains the multilevel estimate of the discount, and these \
intervals, moreover, assume independent rows, a hypothesis that notebook 02 itself showed to be inadequate (a \
correlation of 75.4% between weeks of the same pair).

These results are consistent with the idea that the pooled OLS mixes variation between pairs and variation within \
the pair: if discounts and promotions are concentrated in pairs whose sales levels already differ, this \
composition is entangled with the association of the discount. This is an explanation compatible with the numbers, \
not a demonstration. What the numbers show, concretely, is that ignoring the hierarchy changes the answer: the \
estimated elasticity to the discount, the basis of any price analysis, changes from −0.36 to −1.69 depending on \
the method.
""",
        ),
        md(
            pt="""
## 8. Sensibilidade: sem as linhas de preço-base de fallback

A regra D8 revisada preenche o preço-base ausente apenas com a semana anterior do mesmo par. Quando não há \
histórico, vale `BASE_PRICE = PRICE`, e o desconto da linha é zerado. A tabela a seguir conta as linhas de \
fallback do painel e do treino e as distribui por tipo de promoção.
""",
            en="""
## 8. Sensitivity: without the fallback base-price rows

The revised D8 rule fills a missing base price only with the previous week of the same pair. When there is no \
history, `BASE_PRICE = PRICE` holds, and the discount of the row is set to zero. The following table counts the \
fallback rows of the panel and of the training sample and distributes them by promotion type.
""",
        ),
        code("""
fallback = train[train["BASE_FALLBACK"]]
counts = {"fallback_rows_panel": int(panel["BASE_FALLBACK"].sum()), "fallback_rows_train": len(fallback)}
display(pd.Series(counts).to_frame("n"))
display(interpret.promo_by_discount_band(fallback, LANG))
"""),
        md(
            pt="""
O painel tem 141 linhas de fallback, das quais 111 estão no treino. Todas estão em semanas com promoção: 18 só com \
encarte, 69 só com display e 24 com os dois, nenhuma com TPR, e todas registram desconto zero, ainda que um \
desconto possa ter ocorrido. O preço-base parece faltar justamente em semanas promocionais, e esse é o custo da \
regra. A sensibilidade reajusta o F3 por REML sem essas 111 linhas (execução `fixed_effects_no_fallback`), e a \
tabela a seguir a coloca ao lado do F3.
""",
            en="""
The panel has 141 fallback rows, of which 111 are in the training sample. All of them are in promotion weeks: 18 \
with feature only, 69 with display only and 24 with both, none with TPR, and all record a zero discount, although \
a discount may have occurred. The base price appears to be missing precisely in promotion weeks, and that is the \
cost of the rule. The sensitivity analysis refits F3 by REML without those 111 rows (run \
`fixed_effects_no_fallback`), and the following table places it next to F3.
""",
        ),
        code("""
run_nf = load_run("fixed_effects_no_fallback", "20261124-114601-c4b16e68")
print("run:", run_nf.run_id, "| nobs:", run_nf.load_json("sample")["nobs"])
nofb = interpret.compare_estimates(fixef, run_nf.load_table("fixef"))
main_terms = ~nofb["term"].str.startswith(("SIN_", "COS_"))
display(nofb[main_terms].rename(columns={"estimate_other": "estimate_no_fallback", "se_other": "se_no_fallback"}))
"""),
        md(
            pt="""
Sem as linhas de fallback (346.981 linhas), as estimativas mudam muito pouco: o desconto vai de −1,6852 para \
−1,6848, o preço regular dentro do par, de −1,5921 para −1,5910 (diferença de 0,0011, a maior entre os termos de \
preço e promoção), o encarte fica em 0,4972, o display passa de 0,4087 para 0,4090 e a TPR de −0,0412 para \
−0,0413. A maior diferença é a do termo de média do par, de −0,8797 para −0,8710 (0,0086). O efeito da regra é, \
portanto, desprezível para as conclusões, e os erros-padrão não mudam de forma perceptível. Isso não elimina a \
limitação: nessas 111 linhas, o desconto realizado fica registrado como zero em semanas de encarte ou display.
""",
            en="""
Without the fallback rows (346,981 rows), the estimates change very little: the discount goes from −1.6852 to \
−1.6848, the within-pair regular price from −1.5921 to −1.5910 (a difference of 0.0011, the largest among the \
price and promotion terms), feature stays at 0.4972, display goes from 0.4087 to 0.4090 and TPR from −0.0412 to \
−0.0413. The largest difference is that of the pair-mean term, from −0.8797 to −0.8710 (0.0086). The effect of the \
rule is therefore negligible for the conclusions, and the standard errors do not change appreciably. This does not \
remove the limitation: in those 111 rows, the realised discount is recorded as zero in feature or display weeks.
""",
        ),
        md(
            pt="""
## 9. Sensibilidade: leitura dentro do par com efeitos fixos de par e de semana

Como somente o preço regular foi decomposto em termos dentro e entre pares, o desconto e as variáveis de promoção \
ainda misturam variação dentro e entre pares. Para examinar a leitura dentro do par, foi ajustado um OLS com \
efeitos fixos de par e de semana, estimado por transformação *within* (Wooldridge, 2010), com erros-padrão \
agrupados por par (CR1; Cameron & Miller, 2015), \
nas mesmas 347.092 linhas de treino e 3.859 pares (execução `fe_sensitivity`). Os efeitos de semana absorvem os \
choques comuns, a sazonalidade e a tendência. A tabela põe as estimativas e os erros-padrão ao lado dos do F3.
""",
            en="""
## 9. Sensitivity: within-pair reading with pair and week fixed effects

Because only the regular price was decomposed into within-pair and between-pair terms, the discount and the \
promotion variables still mix within-pair and between-pair variation. To examine the within-pair reading, an OLS \
with pair and week fixed effects was fitted, estimated by the *within* transformation (Wooldridge, 2010), with \
pair-clustered \
standard errors (CR1; Cameron & Miller, 2015), on the same 347,092 training rows and 3,859 pairs (run \
`fe_sensitivity`). The week effects \
absorb common shocks, seasonality and trend. The table places the estimates and standard errors next to those of \
F3.
""",
        ),
        code("""
run_fe = load_run("fe_sensitivity", "20261124-135553-5c4b797b")
print("run:", run_fe.run_id, "| nobs:", run_fe.load_json("sample")["nobs"])
within = interpret.compare_estimates(
    fixef, run_fe.load_table("fe_two_way"), se_col="se_cluster", term_map={"LOG_BASE_PRICE": "LOG_BASE_DEV"}
)
display(within.rename(columns={"estimate_other": "estimate_fe", "se_other": "se_cluster_fe"}))
"""),
        md(
            pt="""
As estimativas pontuais coincidem com as do F3 em até 0,02: o desconto é de −1,703 contra −1,685 (a maior \
diferença, de −0,018), o preço regular de −1,598 contra −1,592, o encarte de 0,492 contra 0,497, o display de \
0,405 contra 0,409 e a TPR de −0,043 contra −0,041. A leitura dentro do par, com controle de tempo, confirma, \
portanto, as magnitudes do F3 também para o desconto e as flags, que não foram decompostos. A estimativa do \
desconto (−1,703) fica próxima da do multinível, e não da do OLS agrupado (−0,359).

Os erros-padrão agrupados por par são de 2,2 a 4,1 vezes maiores que os de Wald do F3 (a razão é de 4,1 para o \
desconto e de 2,2 para o display). Os intervalos de Wald do F3 são, assim, condicionais à especificação do F3 e \
estreitos demais para a incerteza que os dados comportam, o que a seção seguinte detalha.
""",
            en="""
The point estimates coincide with those of F3 within 0.02: the discount is −1.703 against −1.685 (the largest \
difference, −0.018), the regular price −1.598 against −1.592, feature 0.492 against 0.497, display 0.405 against \
0.409 and TPR −0.043 against −0.041. The within-pair reading, with time control, therefore confirms the F3 \
magnitudes also for the discount and the flags, which were not decomposed. The discount estimate (−1.703) is close \
to the multilevel one, and not to the pooled OLS one (−0.359).

The pair-clustered standard errors are 2.2 to 4.1 times larger than the Wald ones of F3 (the ratio is 4.1 for the \
discount and 2.2 for display). The Wald intervals of F3 are thus conditional on the F3 specification and too \
narrow for the uncertainty the data carry, which the next section details.
""",
        ),
        md(
            pt="""
## 10. Dependência residual: defasagens e choques comuns de semana

Os intervalos de Wald do F3 supõem resíduos independentes depois de considerados os efeitos aleatórios. A hipótese \
foi verificada reajustando o F3 por REML no treino (execução `residual_diagnostics`) e examinando os resíduos \
condicionais, isto é, aqueles de que se descontaram os efeitos fixos e os efeitos aleatórios previstos. O primeiro \
gráfico traz a correlação, dentro do mesmo par, entre o resíduo da semana t e o da semana t + k, com os pares \
agrupados.
""",
            en="""
## 10. Residual dependence: lags and common week shocks

The Wald intervals of F3 assume independent residuals once the random effects are accounted for. The assumption \
was checked by refitting F3 by REML on the training sample (run `residual_diagnostics`) and examining the \
conditional residuals, that is, those from which the fixed effects and the predicted random effects were \
subtracted. The first chart shows the correlation, within the same pair, between the residual of week t and that \
of week t + k, with the pairs pooled.
""",
        ),
        code("""
run_diag = load_run("residual_diagnostics", "20261124-115221-9c67822a")
print("run:", run_diag.run_id, "| nobs:", run_diag.load_json("key_numbers")["nobs"])
lag_corr = run_diag.load_table("lag_corr")
fig = viz.plot_residual_lag_corr(lag_corr, LANG)
display(lag_corr)
display(run_diag.load_table("consecutive"))
display(run_diag.load_table("lag1_by_category"))
"""),
        md(
            pt="""
A correlação entre semanas consecutivas do mesmo par é de 0,196 (327.045 comparações), e as defasagens de 2, 3 e 4 \
semanas têm 0,117, 0,092 e 0,096. A dependência decai, mas não desaparece em quatro semanas. Entre linhas \
consecutivas observadas do par, qualquer que seja o intervalo de calendário, a correlação é de 0,192 (343.233 \
comparações). A correlação de defasagem 1 varia por categoria, de 0,143 em higiene oral a 0,253 em snacks, com \
cereal em caixa e pizza congelada em valores intermediários.
""",
            en="""
The correlation between consecutive weeks of the same pair is 0.196 (327,045 comparisons), and the lags of 2, 3 \
and 4 weeks have 0.117, 0.092 and 0.096. The dependence decays but does not vanish within four weeks. Between \
consecutive observed rows of the pair, whatever the calendar gap, the correlation is 0.192 (343,233 comparisons). \
The lag-1 correlation varies by category, from 0.143 in oral hygiene to 0.253 in bag snacks, with boxed cereal and \
frozen pizza at intermediate values.
""",
        ),
        code("""
shocks = run_diag.load_table("week_shocks")
fig = viz.plot_week_shocks(shocks, LANG)
with pd.option_context("display.float_format", "{:,.6f}".format):
    display(run_diag.load_table("week_summary").iloc[0].to_frame("value"))
largest = shocks.loc[shocks["mean_resid"].abs().idxmax()]
week_end = panel.loc[panel["WEEK"] == largest["WEEK"], "WEEK_END_DATE"].iloc[0].date().isoformat()
outside = int((shocks["mean_resid"].abs() > 1.96 * shocks["expected_sd"]).sum())
display(
    pd.Series(
        {
            "weeks_outside_band": outside,
            "weeks": len(shocks),
            "largest_shock_week": int(largest["WEEK"]),
            "largest_shock_week_end": week_end,
            "largest_shock_mean_resid": largest["mean_resid"],
        },
        dtype=object,
    ).to_frame("value")
)
"""),
        md(
            pt="""
O segundo gráfico mostra o resíduo médio de cada semana de treino com a faixa de 95% esperada se as linhas fossem \
independentes. A variância das médias semanais é de 0,007861, contra 0,000081 esperado sob independência, uma \
razão de cerca de 97. A participação das semanas na soma de quadrados dos resíduos é de 2,8%, contra 0,03% \
esperado, e a da combinação categoria × semana é de 6,4%, contra 0,12% esperado. Há, portanto, choques comuns a \
todas as linhas de uma semana, em especial por categoria. A autocorrelação de defasagem 1 das médias semanais é de \
−0,005, ou seja, esses choques não persistem de uma semana para a seguinte. Em 79 das 104 semanas, a média fica \
fora da faixa. A maior é a da semana 61 (semana encerrada em 2010-03-17), com resíduo médio de −0,582, contra um \
desvio-padrão esperado de 0,009.

Os resíduos marginais, que não descontam os efeitos aleatórios, também foram guardados na execução. Eles refletem \
sobretudo os efeitos de loja, produto e par, e não informam sobre os intervalos de Wald.

Em conjunto, os resíduos condicionais do F3 não são independentes: há correlação serial positiva dentro do par e \
choques comuns de semana e de categoria. Essa dependência diz respeito à incerteza, e não às estimativas pontuais, \
que coincidem com as de um estimador de efeitos fixos de par e de semana (seção 9). Os intervalos de Wald do F3 \
são condicionais à especificação do F3 e estreitos demais. Depois deste notebook, foram calculados erros-padrão \
agrupados por par e por semana (Cameron et al., 2011) no estimador de efeitos fixos (execução `fe_sensitivity`, \
registro de decisões, \
emenda E4): eles ficam de 4,1 a 7,4 vezes os de Wald do F3, com o agrupamento por semana dominante, e os sinais \
das estimativas se mantêm.
""",
            en="""
The second chart shows the mean residual of each training week with the 95% band expected if the rows were \
independent. The variance of the weekly means is 0.007861, against 0.000081 expected under independence, a ratio \
of about 97. The share of the sum of squares of the residuals due to weeks is 2.8%, against 0.03% expected, and \
that of the category × week combination is 6.4%, against 0.12% expected. There are, therefore, shocks common to \
all the rows of a week, especially by category. The lag-1 autocorrelation of the weekly means is −0.005, that is, \
these shocks do not persist from one week to the next. In 79 of the 104 weeks, the mean falls outside the band. \
The largest is that of week 61 (week ending 2010-03-17), with a mean residual of −0.582, against an expected \
standard deviation of 0.009.

The marginal residuals, which do not subtract the random effects, were also stored in the run. They mostly reflect \
the store, product and pair effects, and are not informative about the Wald intervals.

Taken together, the conditional residuals of F3 are not independent: there is positive serial correlation within \
the pair and shocks common to the week and the category. This dependence concerns the uncertainty, not the point \
estimates, which coincide with those of a pair and week fixed-effects estimator (section 9). The Wald intervals of \
F3 are conditional on the F3 specification and too narrow. After this notebook, standard errors clustered by pair \
and by week (Cameron et al., 2011) were computed on the fixed-effects estimator (run `fe_sensitivity`, decision \
register, amendment E4): \
they are 4.1 to 7.4 times the F3 Wald ones, with week clustering dominant, and the signs of the estimates hold.
""",
        ),
        md(
            pt="""
## 11. Limitações

Preço e promoção são decididos pela própria rede, e não atribuídos ao acaso. Se as promoções são planejadas para \
semanas de demanda já alta (ou se os descontos são aplicados em resposta a estoques e à concorrência), parte da \
associação estimada reflete essa seleção, e os coeficientes descrevem associações condicionais, não efeitos \
causais (endogeneidade; Villas-Boas & Winer, 1999). Uma recomendação de preço ou um ROI incremental exigiria \
identificação causal e dados de \
margem, que o projeto não tem. A especificação log-log também impõe uma elasticidade constante: um desconto de 10% \
e um de 30% atuam com a mesma elasticidade, o que é uma aproximação, e a sobreposição entre tipos de promoção e \
faixas de desconto (seção 4) limita as comparações entre promoções.

A dependência residual foi medida (seção 10): há correlação serial de 0,196 entre semanas consecutivas do mesmo \
par e choques comuns de semana. Ela não foi modelada, e os intervalos de Wald do F3 são condicionais à sua \
especificação e estreitos demais, como indicam os erros-padrão agrupados da seção 9, de 2,2 a 4,1 vezes maiores. \
Além disso, a formulação (a decomposição de preço da D12 e as demais) foi motivada pela exploração inicial do \
painel, que incluiu as 52 semanas finais, e é uma decisão de desenvolvimento. Nenhum modelo foi ajustado nessas \
semanas, e nenhuma comparação preditiva foi calculada nelas.

Somente as semanas observadas foram modeladas (D3), de modo que as estimativas são condicionais a haver venda no \
par naquela semana, e não incluem semanas sem registro, que poderiam ser vendas zero ou ruptura. Por fim, o modelo \
estima uma única elasticidade para todas as lojas e produtos, ao passo que o interesse do projeto está justamente \
em como o efeito de preço varia entre eles. Essa última limitação motiva o notebook 04.
""",
            en="""
Price and promotion are decided by the retailer itself, and are not assigned at random. If promotions are planned \
for weeks of already high demand (or if discounts are applied in response to inventory and competition), part of \
the estimated association reflects this selection, and the coefficients describe conditional associations, not \
causal effects (endogeneity; Villas-Boas & Winer, 1999). A pricing recommendation or an incremental ROI would \
require causal identification \
and margin data, which the project does not have. The log-log specification also imposes a constant elasticity: a \
10% discount and a 30% discount act with the same elasticity, which is an approximation, and the overlap between \
promotion types and discount bands (section 4) limits the comparisons between promotions.

Residual dependence was measured (section 10): there is a serial correlation of 0.196 between consecutive weeks of \
the same pair and shocks common to the week. It was not modelled, and the Wald intervals of F3 are conditional on \
its specification and too narrow, as the clustered standard errors of section 9, 2.2 to 4.1 times larger, \
indicate. In addition, the formulation (the price decomposition of D12 and the others) was motivated by the \
initial exploration of the panel, which included the final 52 weeks, and is a development decision. No model was \
fitted on those weeks, and no predictive comparison was computed on them.

Only the observed weeks were modelled (D3), so the estimates are conditional on there being a sale in the pair \
that week, and do not include weeks without a record, which could be zero sales or a stock-out. Finally, the model \
estimates a single elasticity for all stores and products, whereas the interest of the project lies precisely in \
how the price effect varies among them. This last limitation motivates notebook 04.
""",
        ),
        md(
            pt="""
## 12. Conclusões e próximo passo

Com a hierarquia mantida (interceptos aleatórios de loja, de produto e de par), um desconto 10% mais profundo está \
associado a cerca de 19,4% mais unidades (elasticidade de −1,685), e um preço regular 10% maior a cerca de 14,1% \
menos. O encarte (+64,4%) e o display (+50,5%) estão associados a mais unidades além do desconto, e a flag de TPR \
tem coeficiente de −0,041 (−4,0%), valor que depende de comparar semanas sem promoção com o mesmo desconto, \
comparação que os dados quase não contêm. Para a leitura de negócio, os cenários conjuntos de desconto e promoção \
da seção 4 (por exemplo, TPR com 10% de desconto, +14,6%, e com 20%, +39,8%) têm suporte nos dados, e os que \
combinam encarte ou display sem desconto têm suporte menor. A tendência é de −2,1% por ano, e a sazonalidade, de \
−4,2% a +6,6% sobre a semana média, é pequena frente às associações de preço. Todos esses valores são associações \
condicionais em semanas com transações registradas.

Preço e promoção explicam 34,4% da variância semanal dentro do par e 29,3% da variância entre produtos, e os \
pequenos valores negativos de pseudo-R² nos níveis de loja e de par são um artefato conhecido. O OLS agrupado \
estima uma elasticidade ao desconto de −0,36, contra −1,69 do multinível. A exclusão das 111 linhas de fallback \
quase não altera as estimativas, e a regressão com efeitos fixos de par e de semana reproduz as estimativas \
pontuais do F3 em até 0,02, com erros-padrão agrupados de 2,2 a 4,1 vezes maiores. Os resíduos condicionais têm \
correlação de 0,196 entre semanas consecutivas e choques comuns de semana. A dependência afeta a incerteza, e não \
as estimativas pontuais, e os intervalos de Wald do F3 são condicionais à especificação e estreitos demais; os \
erros-padrão agrupados por par e por semana, registrados na emenda E4, ficam de 4,1 a 7,4 vezes maiores.

O notebook 04 apresenta os resultados do protocolo pré-registrado P1, aprovado com emendas, sobre se deixar a \
elasticidade ao desconto variar entre lojas e produtos melhora a previsão fora da amostra.
""",
            en="""
## 12. Conclusions and next step

With the hierarchy kept (random intercepts for store, product and pair), a 10% deeper discount is associated with \
about 19.4% more units (elasticity of −1.685), and a 10% higher regular price with about 14.1% fewer. Feature \
(+64.4%) and display (+50.5%) are associated with more units beyond the discount, and the TPR flag has a \
coefficient of −0.041 (−4.0%), a value that depends on comparing weeks with no promotion at the same discount, a \
comparison that the data hardly contain. For the business reading, the joint discount-and-promotion scenarios of \
section 4 (for example, TPR with a 10% discount, +14.6%, and with 20%, +39.8%) have support in the data, and those \
that combine feature or display without a discount have lower support. The trend is −2.1% per year, and \
seasonality, from −4.2% to +6.6% around the average week, is small compared with the price associations. All these \
values are conditional associations in weeks with recorded transactions.

Price and promotion explain 34.4% of the weekly variance within the pair and 29.3% of the variance between \
products, and the small negative pseudo-R² values at the store and pair levels are a known artefact. The pooled \
OLS estimates a discount elasticity of −0.36, against −1.69 for the multilevel model. Excluding the 111 fallback \
rows barely changes the estimates, and the regression with pair and week fixed effects reproduces the F3 point \
estimates within 0.02, with clustered standard errors 2.2 to 4.1 times larger. The conditional residuals have a \
correlation of 0.196 between consecutive weeks and shocks common to the week. The dependence affects the \
uncertainty, not the point estimates, and the Wald intervals of F3 are conditional on the specification and too \
narrow; the standard errors clustered by pair and by week, recorded in amendment E4, are 4.1 to 7.4 times larger.

Notebook 04 presents the results of the pre-registered protocol P1, approved with amendments, on whether letting \
the discount elasticity vary across stores and products improves out-of-sample prediction.
""",
        ),
    ]


SETUP_04 = """
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from retail_multilevel import interpret, viz
from retail_multilevel.i18n import t
from retail_multilevel.runs import load_run

warnings.filterwarnings("ignore", category=FutureWarning)
viz.use_retina()
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:,.4f}".format)
folds = load_run("p1_folds", "20261124-135426-7e46f16b")
tuning = load_run("gpboost_tuning", "20261124-135420-206b7827")
final = load_run("p1_final", "20261124-144615-0aba3b28")
print(t("data_credit", LANG))
stored = {r.metadata["experiment"]: r for r in (folds, tuning, final)}
display(
    pd.DataFrame(
        {name: {"run_id": r.run_id, "finished_at": r.metadata["finished_at"]} for name, r in stored.items()}
    ).T
)
"""


def protocol_p1_cells() -> list[Cell]:
    """Cells of notebook 04, the pre-registered protocol P1 read from the stored runs."""
    return [
        md(
            pt="""
# 04 · Protocolo P1: inclinações aleatórias e comparação fora da amostra

Este notebook apresenta o protocolo pré-registrado P1 a partir de artefatos gravados. Nenhum modelo é ajustado ou \
pontuado aqui: os ajustes e a pontuação final foram feitos por scripts, uma vez cada, e gravados em `outputs/runs`. \
Para o P1, são lidas três execuções (`p1_folds`, `gpboost_tuning` e `p1_final`), nas duas versões de idioma. \
As métricas do P1 são erros de previsão de log(unidades), \
condicionais ao preço e à promoção realizados na semana. A seção 9 lê uma quarta execução com um teste \
inferencial complementar, feito somente no treino depois da conclusão do P1.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 04 · Protocol P1: random slopes and out-of-sample comparison

This notebook presents the pre-registered protocol P1 from stored artefacts. No model is fitted or scored here: \
the fits and the final scoring were made by scripts, once each, and stored in `outputs/runs`. Both language \
versions read the same three P1 runs (`p1_folds`, `gpboost_tuning` and `p1_final`). The P1 metrics are \
prediction errors of log(units), conditional on the price and promotion \
realised in the week. Section 9 reads a fourth run for a supplementary inferential test conducted only on the \
training data after P1 was completed.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_04),
        md(
            pt="""
As três execuções lidas estão identificadas acima (identificador e horário de término). Os identificadores são \
fixados no código, de modo que reexecutar o notebook lê sempre os mesmos artefatos.

## 1. A pergunta e o pré-registro

A pergunta do protocolo é se permitir que a elasticidade ao desconto varie entre lojas e produtos (inclinações \
aleatórias de `LOG_DISC`; Gelman & Hill, 2007) melhora a previsão fora da amostra, em especial para os pares loja \
× produto com pouco \
histórico. A razão é que, num par curto, uma elasticidade estimada só com as semanas do próprio par é instável, e o \
encolhimento (Robinson, 1991) em direção à média da loja e do produto poderia estabilizá-la. O notebook 03 estimou \
uma única \
elasticidade comum a todos os pares, e é essa limitação que o P1 examina.

O protocolo foi pré-registrado (Nosek et al., 2018; Weston et al., 2019): foi escrito e aprovado, com seis emendas \
(E1 a E6), antes de qualquer candidato ser \
ajustado, e fixou de antemão os candidatos, a ordem de comparação, a regra de parada, a métrica e as populações. As \
emendas definiram, entre outros pontos, que o candidato inicial C0 é o F3 do notebook 03 (E1), como a cobertura do \
OLS por par é medida (E2), a regra sequencial de parada (E3) e o congelamento anterior à avaliação final (E6). \
Nenhum limiar nem fórmula foi alterado depois de ver os resultados das dobras internas.

Todos os candidatos compartilham a parte fixa do F3 (decisões D12 a D14) e as mesmas linhas.

| Candidato | Especificação além da parte fixa do F3 | Situação neste notebook |
|---|---|---|
| C0 | Interceptos aleatórios de loja, de produto e do par, com uma elasticidade ao desconto comum | Ajustado \
nas dobras e na janela final |
| C1 | C0 mais inclinações aleatórias independentes de `LOG_DISC` por produto e por loja | Ajustado nas dobras e \
rejeitado |
| C2 | C1 com intercepto e inclinação correlacionados, por produto e por loja | Não ajustado (a sequência parou) |
| C3 | O melhor entre C1 e C2, mais inclinação aleatória de `LOG_DISC` por par | Não ajustado (a sequência parou) |
| C4 | O melhor anterior, mais inclinações aleatórias de `FEATURE` e `DISPLAY` por produto | Não ajustado (a \
sequência parou) |

Cada candidato é comparado com o último modelo aceito, na ordem C1, C2, C3, C4. Para ser aceito, precisa convergir \
sem singularidade (Bates et al., 2015) e, em caso de aviso, com uma nova tentativa com o otimizador `nloptwrap`, \
ter MAE por par menor nas \
duas dobras internas e reduzir o MAE relativo em pelo menos 0,5% na média das dobras. Se falhar em qualquer \
condição, o último modelo aceito é mantido e a sequência termina, de modo que um candidato posterior não é ajustado \
quando o anterior falha.

A seleção usa duas dobras internas (Tashman, 2000) dentro das 104 semanas de treino. Na dobra A, o ajuste usa as \
semanas 0 a 51 e a \
validação usa as semanas 52 a 77. Na dobra B, o ajuste usa as semanas 0 a 77 e a validação usa as semanas 78 a 103. \
Todas as transformações aprendidas, como as médias de par, são recalculadas dentro de cada janela de ajuste. Só \
depois da seleção o modelo escolhido é reajustado nas semanas 0 a 103 e pontuado nas semanas 104 a 155.
""",
            en="""
The three runs that were read are identified above (identifier and finish time). The identifiers are fixed in the \
code, so re-running the notebook always reads the same artefacts.

## 1. The question and the pre-registration

The question of the protocol is whether letting the discount elasticity vary across stores and products (random \
slopes of `LOG_DISC`; Gelman & Hill, 2007) improves out-of-sample prediction, especially for store × product pairs \
with little history. \
The reason is that, in a short pair, an elasticity estimated only from the weeks of the pair itself is unstable, \
and shrinkage (Robinson, 1991) towards the mean of the store and of the product could stabilise it. Notebook 03 \
estimated a single \
elasticity common to all pairs, and it is this limitation that P1 examines.

The protocol was pre-registered (Nosek et al., 2018; Weston et al., 2019): it was written and approved, with six \
amendments (E1 to E6), before any candidate \
was fitted, and it fixed in advance the candidates, the order of comparison, the stopping rule, the metric and the \
populations. The amendments defined, among other points, that the initial candidate C0 is the F3 of notebook 03 \
(E1), how the coverage of the per-pair OLS is measured (E2), the sequential stopping rule (E3) and the freeze that \
precedes the final evaluation (E6). No threshold or formula was changed after the results of the internal folds \
were seen.

All candidates share the fixed part of F3 (decisions D12 to D14) and the same rows.

| Candidate | Specification beyond the fixed part of F3 | Status in this notebook |
|---|---|---|
| C0 | Random intercepts for store, product and pair, with a common discount elasticity | Fitted on the folds \
and on the final window |
| C1 | C0 plus independent random slopes of `LOG_DISC` by product and by store | Fitted on the folds and \
rejected |
| C2 | C1 with correlated intercept and slope, by product and by store | Not fitted (the sequence stopped) |
| C3 | The better of C1 and C2, plus a random slope of `LOG_DISC` by pair | Not fitted (the sequence stopped) |
| C4 | The best earlier model, plus random slopes of `FEATURE` and `DISPLAY` by product | Not fitted (the \
sequence stopped) |

Each candidate is compared with the last accepted model, in the order C1, C2, C3, C4. To be accepted, it has to \
converge without singularity (Bates et al., 2015) and, in case of a warning, with a new attempt using the \
`nloptwrap` optimiser, have a \
lower per-pair MAE in both internal folds and reduce the relative MAE by at least 0.5% on average over the folds. \
If it fails any condition, the last accepted model is kept and the sequence ends, so a later candidate is not \
fitted when the earlier one fails.

The selection uses two internal folds (Tashman, 2000) within the 104 training weeks. In fold A, the fit uses weeks \
0 to 51 and the \
validation uses weeks 52 to 77. In fold B, the fit uses weeks 0 to 77 and the validation uses weeks 78 to 103. All \
learned transformations, such as the pair means, are recomputed within each fit window. Only after the selection \
is the chosen model refitted on weeks 0 to 103 and scored on weeks 104 to 155.
""",
        ),
        md(
            pt="""
## 2. Métrica, populações, referências e congelamento

A métrica principal é o erro absoluto médio (MAE) de `LOG_UNITS`, calculado primeiro dentro de cada par e depois \
promediado com peso igual por par. Cada par conta, portanto, uma vez, independentemente do número de semanas, e o \
MAE por linha (peso igual por linha) é relatado como medida complementar. As previsões são condicionais ao preço e \
à promoção realizados na semana, e a população são as linhas do modelo, uma coorte retrospectiva, porque o filtro \
das linhas usa informação conhecida só depois da venda.

Três populações são comparadas, e os pares são divididos em estratos de histórico, contados na janela de ajuste \
(menos de 26, de 26 a 77 e 78 semanas ou mais).

| População | Definição |
|---|---|
| (a) Pares vistos | Pares presentes na janela de ajuste. É a população da comparação principal |
| (b) Linhas cobertas pelo OLS por par | Subconjunto de (a) em que o OLS por par consegue prever (emenda E2), \
com todos os modelos avaliados nas mesmas linhas |
| (c) Pares inéditos | Pares sem histórico na janela de ajuste, relatados à parte, com efeito de par igual a zero |

As referências são o OLS agrupado (a parte fixa do F3, sem efeitos aleatórios), o OLS por par (ajustado com \
`LOG_DISC`, `FEATURE`, `DISPLAY` e `TPR_ONLY`, só nos pares com pelo menos 8 semanas na janela de ajuste) e o \
GPBoost (Sigrist, 2022), um desafiante secundário de aprendizado de máquina com efeitos de grupo de loja, de \
produto e de par. Os \
hiperparâmetros do GPBoost foram calibrados somente nas dobras internas, e ele não participa da sequência de \
seleção.

**Congelamento e estatuto da janela final.** Antes da avaliação final, o modelo selecionado (C0), os parâmetros do \
GPBoost, as versões do código e o painel foram congelados e registrados no registro de decisões, e a avaliação das \
semanas 104 a 155 foi prevista uma única vez, com o resultado publicado em qualquer direção. A tabela a seguir \
compara os hashes de congelamento (12 primeiros caracteres do SHA-256) com os calculados no momento da pontuação. \
Essa janela, porém, é uma janela de avaliação temporal **retrospectiva, com exposição exploratória prévia** (D11 \
revisada): o painel inteiro foi explorado antes do protocolo, e a formulação do modelo, como a decomposição de preço \
da D12, é uma decisão de desenvolvimento. Ela não é um conjunto de teste intocado nem cego.
""",
            en="""
## 2. Metric, populations, references and the freeze

The primary metric is the mean absolute error (MAE) of `LOG_UNITS`, computed first within each pair and then \
averaged with equal weight per pair. Each pair therefore counts once, regardless of the number of weeks, and the \
MAE per row (equal weight per row) is reported as a complementary measure. The predictions are conditional on the \
price and the promotion realised in the week, and the population is the model rows, a retrospective cohort, \
because the row filter uses information known only after the sale.

Three populations are compared, and the pairs are divided into history strata, counted in the fit window (fewer \
than 26, 26 to 77, and 78 weeks or more).

| Population | Definition |
|---|---|
| (a) Seen pairs | Pairs present in the fit window. This is the population of the main comparison |
| (b) Rows covered by the per-pair OLS | Subset of (a) in which the per-pair OLS can predict (amendment E2), with \
all models evaluated on the same rows |
| (c) Unseen pairs | Pairs with no history in the fit window, reported separately, with the pair effect set to \
zero |

The references are the pooled OLS (the fixed part of F3, without random effects), the per-pair OLS (fitted with \
`LOG_DISC`, `FEATURE`, `DISPLAY` and `TPR_ONLY`, only in pairs with at least 8 weeks in the fit window) and \
GPBoost (Sigrist, 2022), a secondary machine-learning challenger with group effects for store, product and pair. \
The GPBoost \
hyperparameters were tuned only on the internal folds, and it does not take part in the selection sequence.

**The freeze and the status of the final window.** Before the final evaluation, the selected model (C0), the \
GPBoost parameters, the code versions and the panel were frozen and recorded in the decision register, and the \
evaluation of weeks 104 to 155 was planned to be made once, with the result published in either direction. The \
table below compares the freeze hashes (first 12 characters of the SHA-256) with those computed at scoring time. \
That window, however, is a **retrospective temporal evaluation window with prior exploratory exposure** (D11, \
revised): the whole panel was explored before the protocol, and the formulation of the model, such as the price \
decomposition of D12, is a development decision. It is not an untouched or blind test set.
""",
        ),
        code("""
cfg = final.metadata["config"]
hashes = pd.DataFrame({"frozen": cfg["frozen_hashes"], "at_scoring": cfg["current_hashes"]})
hashes["identical"] = hashes["frozen"] == hashes["at_scoring"]
display(hashes)
display(pd.Series(final.load_json("summary")).to_frame("summary"))
"""),
        md(
            pt="""
## 3. Seleção nas dobras internas

A tabela mostra o registro da sequência de seleção. O C1 foi comparado com o C0, e o resultado determinou a parada \
da sequência.
""",
            en="""
## 3. Selection on the internal folds

The table shows the log of the selection sequence. C1 was compared with C0, and the result determined that the \
sequence stopped.
""",
        ),
        code("""
folds_m = folds.load_table("metrics")
log = interpret.selection_table(folds.load_table("selection_log"), LANG)
display(log.set_index(t("col_candidate", LANG)).T)
display(folds.load_table("convergence"))
display(folds.load_table("coverage").T.rename(columns={0: "A", 1: "B"}))
"""),
        md(
            pt="""
Os quatro ajustes (C0 e C1 em cada dobra) convergiram com o otimizador bobyqa, sem mensagens e sem singularidade, \
em 19 a 85 s, com 18 parâmetros no C0 e 20 no C1. O C1 teve MAE por par de 0,4294 na dobra A, contra 0,4230 do C0, \
um ganho de −1,52%, isto é, uma piora. Na dobra B, o MAE foi de 0,4188 contra 0,4211, um ganho de 0,54%. O ganho \
médio, de −0,49%, ficou abaixo do limiar de 0,5%, e a regra rejeitou o C1. O C2, o C3 e o C4 não foram ajustados.

A cobertura das dobras mostra que o OLS por par cobriu 98,9% dos pares e 97,8% das linhas na dobra A, e 99,2% e \
97,8% na dobra B. Os pares inéditos são só 4 na dobra A (32 linhas) e 13 na dobra B (80 linhas), pequenos demais \
para qualquer conclusão, e na dobra A, 1.710 linhas de validação não têm UPC na janela de ajuste e entram só na \
contagem de cobertura.
""",
            en="""
The four fits (C0 and C1 in each fold) converged with the bobyqa optimiser, without messages and without \
singularity, in 19 to 85 s, with 18 parameters for C0 and 20 for C1. C1 had a per-pair MAE of 0.4294 in fold A, \
against 0.4230 for C0, a gain of −1.52%, that is, a deterioration. In fold B, the MAE was 0.4188 against 0.4211, a \
gain of 0.54%. The mean gain, −0.49%, fell below the 0.5% threshold, and the rule rejected C1. C2, C3 and C4 were \
not fitted.

The coverage of the folds shows that the per-pair OLS covered 98.9% of the pairs and 97.8% of the rows in fold A, \
and 99.2% and 97.8% in fold B. The unseen pairs are only 4 in fold A (32 rows) and 13 in fold B (80 rows), too few \
for any conclusion, and in fold A, 1,710 validation rows have no UPC in the fit window and enter only the coverage \
count.
""",
        ),
        code("""
fig = viz.plot_fold_mae(folds_m, LANG)
display(interpret.metric_table(folds_m, "a_seen", LANG, models=["C0", "C1", "OLS_pooled"]))
print(t("pop_b_ols_covered", LANG))
display(interpret.metric_table(folds_m, "b_ols_covered", LANG, models=["C0", "C1", "OLS_pair", "OLS_pooled"]))
"""),
        md(
            pt="""
O gráfico e a primeira tabela mostram o MAE por par nos pares vistos, por dobra e por estrato de histórico. O \
OLS agrupado teve MAE de 0,9149 na dobra A e de 0,9036 na dobra B, mais que o dobro do C0. O C1 foi melhor que o \
C0 no estrato com menos de 26 semanas nas duas dobras (dobra A: 0,3686 contra 0,3828; dobra B: 0,4956 contra \
0,4999), na direção da hipótese. Ele foi pior no estrato de 26 a 77 semanas da dobra A (0,4338 contra 0,4259), que \
concentra 3.425 dos 3.671 pares dessa dobra. O estrato de 78 semanas ou mais está vazio na dobra A por construção, \
porque a janela de ajuste tem só 52 semanas. Na população (b), o C0 teve MAE por par menor que o do OLS por par nas \
duas dobras (0,4230 contra 0,4690 na dobra A, e 0,4190 contra 0,4549 na dobra B).
""",
            en="""
The chart and the first table show the per-pair MAE on the seen pairs, by fold and history stratum. The pooled OLS \
had an MAE of 0.9149 in fold A and 0.9036 in fold B, more than twice that of C0. C1 was better than C0 in the \
stratum with fewer than 26 weeks in both folds (fold A: 0.3686 against 0.3828; fold B: 0.4956 against 0.4999), in \
the direction of the hypothesis. It was worse in the 26 to 77 weeks stratum of fold A (0.4338 against 0.4259), \
which holds 3,425 of the 3,671 pairs of that fold. The stratum of 78 weeks or more is empty in fold A by \
construction, because the fit window has only 52 weeks. In population (b), C0 had a lower per-pair MAE than the \
per-pair OLS in both folds (0.4230 against 0.4690 in fold A, and 0.4190 against 0.4549 in fold B).
""",
        ),
        code("""
weekly_folds = folds.load_table("weekly_diff")
display(interpret.weekly_summary_table(weekly_folds, LANG))
display(weekly_folds.loc[weekly_folds.groupby("fold")["mae_diff"].idxmin(), ["fold", "WEEK", "mae_diff"]])
"""),
        md(
            pt="""
A série semanal da diferença entre o C1 e o C0 (negativa quando o C1 é melhor) reforça a inconsistência entre as \
dobras. Na dobra A, o C1 foi melhor em 5 das 26 semanas, com diferença média de +0,0082, isto é, pior. Na dobra B, \
foi melhor em 19 das 26 semanas, com média de −0,0022. O maior ganho do C1 na dobra A, de −0,0548, ocorreu na \
semana 61, a do choque comum de semana investigado no notebook 03. Um ganho que muda de sinal entre as dobras é o \
caso em que a regra mantém o modelo mais simples.
""",
            en="""
The weekly series of the difference between C1 and C0 (negative when C1 is better) reinforces the inconsistency \
between the folds. In fold A, C1 was better in 5 of the 26 weeks, with a mean difference of +0.0082, that is, \
worse. In fold B, it was better in 19 of the 26 weeks, with a mean of −0.0022. The largest gain of C1 in fold A, \
−0.0548, occurred in week 61, the common week shock investigated in notebook 03. A gain that changes sign between \
the folds is the case in which the rule keeps the simpler model.
""",
        ),
        md(
            pt="""
## 4. Calibração do GPBoost

O GPBoost foi calibrado com Optuna (Akiba et al., 2019) e amostrador TPE (Bergstra et al., 2011) em 50 tentativas, \
e o objetivo foi a média, nas duas dobras, do MAE por \
par dos pares vistos. Por tempo de execução, cada tentativa usou uma subamostra fixa de 50% dos pares da janela de \
ajuste e no máximo 300 rodadas, sem parada antecipada; os dois limites constam da configuração gravada da execução \
(`fit_frac` e `max_rounds`). A coluna de rodadas da tabela registra o valor amostrado pela busca, antes do limite.
""",
            en="""
## 4. GPBoost tuning

GPBoost was tuned with Optuna (Akiba et al., 2019) and the TPE sampler (Bergstra et al., 2011) over 50 trials, and \
the objective was the mean, over the two folds, of the \
per-pair MAE of the seen pairs. To limit the run time, each trial used a fixed subsample of 50% of the pairs of the \
fit window and at most 300 rounds, without early stopping; both limits are stored in the run's recorded \
configuration (`fit_frac` and `max_rounds`). The rounds column of the table records the value sampled by the \
search, before the cap.
""",
        ),
        code("""
best = tuning.load_json("best_params")
trials = tuning.load_table("trials")
space = trials.filter(like="params_").agg(["min", "max"]).T
space.index = space.index.str.removeprefix("params_")
space["selected"] = pd.Series({**best["params"], "num_boost_round": best["num_boost_round"]})
display(space)
print("trials:", len(trials), "| complete:", int((trials["state"] == "COMPLETE").sum()))
print("objective of the selected trial:", round(best["objective"], 4))
display(interpret.gpboost_fold_table(folds_m, tuning.load_table("fold_metrics"), LANG))
"""),
        md(
            pt="""
Os parâmetros selecionados ficaram perto das bordas do espaço de busca: 60 rodadas (o menor valor amostrado foi \
55, e o espaço começa em 50), `min_data_in_leaf` de 459 (o maior valor amostrado foi 483, e o espaço termina em \
500) e `feature_fraction` de 0,601 (o menor valor amostrado foi 0,6007, e o espaço começa em 0,6). A parte de \
árvores fica, portanto, pouco ajustada. Isso é uma limitação, e o espaço não foi reaberto, porque essa seria uma \
decisão tomada depois de ver o resultado.

Reajustado na janela inteira de cada dobra, o GPBoost teve MAE por par de 0,4148 na dobra A e de 0,4133 na dobra B, \
contra 0,4230 e 0,4211 do C0, diferenças de −1,95% e −1,86%. Essa comparação favorece o GPBoost, porque os \
hiperparâmetros dele foram escolhidos nessas mesmas dobras e o C0 não tem hiperparâmetros. Os valores das dobras não \
são, portanto, uma comparação justa entre os dois modelos.
""",
            en="""
The selected parameters were close to the edges of the search space: 60 rounds (the smallest sampled value was 55, \
and the space starts at 50), `min_data_in_leaf` of 459 (the largest sampled value was 483, and the space ends at \
500) and `feature_fraction` of 0.601 (the smallest sampled value was 0.6007, and the space starts at 0.6). The tree \
part is therefore under-tuned. This is a limitation, and the space was not reopened, because that would be a \
decision taken after seeing the result.

Refitted on the whole window of each fold, GPBoost had a per-pair MAE of 0.4148 in fold A and 0.4133 in fold B, \
against 0.4230 and 0.4211 for C0, differences of −1.95% and −1.86%. This comparison favours GPBoost, because its \
hyperparameters were chosen on those same folds and C0 has no hyperparameters. The fold values are therefore not a \
fair comparison between the two models.
""",
        ),
        md(
            pt="""
## 5. A janela final (semanas 104 a 155)

O C0 foi reajustado nas semanas 0 a 103 (347.092 linhas) e pontuou 177.767 linhas das semanas 104 a 155, a janela \
retrospectiva com exposição exploratória prévia. O ajuste convergiu com bobyqa, sem mensagens e sem singularidade, \
em cerca de 32 s e com 18 parâmetros. As tabelas a seguir apresentam a convergência, a cobertura do OLS por par e o \
MAE por par e por linha em cada população.
""",
            en="""
## 5. The final window (weeks 104 to 155)

C0 was refitted on weeks 0 to 103 (347,092 rows) and scored 177,767 rows of weeks 104 to 155, the retrospective \
window with prior exploratory exposure. The fit converged with bobyqa, without messages and without singularity, in \
about 32 s and with 18 parameters. The tables below present the convergence, the coverage of the per-pair OLS and \
the MAE per pair and per row in each population.
""",
        ),
        code("""
final_m = final.load_table("metrics")
display(final.load_table("convergence"))
display(final.load_table("coverage").T.rename(columns={0: "final"}))
for pop in ("a_seen", "b_ols_covered", "c_unseen"):
    print(t("pop_" + pop, LANG), "|", t("metric_mae_pair", LANG))
    display(interpret.metric_table(final_m, pop, LANG))
"""),
        code("""
for pop in ("a_seen", "b_ols_covered", "c_unseen"):
    print(t("pop_" + pop, LANG), "|", t("metric_mae_row", LANG))
    display(interpret.metric_table(final_m, pop, LANG, value="mae_row"))
"""),
        md(
            pt="""
Nos pares vistos (população a), o MAE por par foi de 0,4541 para o C0, 0,4520 para o GPBoost e 0,9172 para o OLS \
agrupado, e o MAE por linha foi de 0,4446, 0,4478 e 0,8823. Por estrato, o GPBoost teve o menor MAE por par nos \
pares com menos de 26 semanas (0,4376 contra 0,5430 do C0) e nos de 26 a 77 semanas (0,4791 contra 0,4922), e o C0 \
teve o menor nos de 78 semanas ou mais (0,4448 contra 0,4478).

O OLS por par teve cobertura em 3.779 dos 3.820 pares vistos (98,9%) e em 174.206 das 176.529 linhas (98,7%). \
Faltou estimativa de `LOG_DISC` em 23 pares, de `FEATURE` em 825, de `DISPLAY` em 905 e de `TPR_ONLY` em 382. Nessas \
linhas cobertas (população b), o MAE por par foi de 0,4510 para o C0, 0,4506 para o GPBoost, 0,4941 para o OLS por \
par e 0,9131 para o OLS agrupado.

Nos 50 pares inéditos (população c, 1.238 linhas, nenhuma sem covariável), o MAE por par foi de 0,6260 para o C0, \
0,6876 para o GPBoost e 0,9925 para o OLS agrupado. O OLS por par não tem previsão para esses pares.
""",
            en="""
On the seen pairs (population a), the per-pair MAE was 0.4541 for C0, 0.4520 for GPBoost and 0.9172 for the pooled \
OLS, and the MAE per row was 0.4446, 0.4478 and 0.8823. By stratum, GPBoost had the lowest per-pair MAE in pairs \
with fewer than 26 weeks (0.4376 against 0.5430 for C0) and in those with 26 to 77 weeks (0.4791 against 0.4922), \
and C0 had the lowest in those with 78 weeks or more (0.4448 against 0.4478).

The per-pair OLS had coverage in 3,779 of the 3,820 seen pairs (98.9%) and in 174,206 of the 176,529 rows (98.7%). \
An estimate was missing for `LOG_DISC` in 23 pairs, for `FEATURE` in 825, for `DISPLAY` in 905 and for `TPR_ONLY` \
in 382. On those covered rows (population b), the per-pair MAE was 0.4510 for C0, 0.4506 for GPBoost, 0.4941 for \
the per-pair OLS and 0.9131 for the pooled OLS.

On the 50 unseen pairs (population c, 1,238 rows, none without a covariate), the per-pair MAE was 0.6260 for C0, \
0.6876 for GPBoost and 0.9925 for the pooled OLS. The per-pair OLS has no prediction for these pairs.
""",
        ),
        code("""
wd = final.load_table("weekly_diff")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
viz.plot_weekly_error_diff(wd, LANG, ax=axes[0])
viz.plot_weekly_error_diff(wd[wd["compared_to"] != "OLS_pooled"], LANG, ax=axes[1])
display(interpret.weekly_summary_table(wd, LANG))
"""),
        md(
            pt="""
A série semanal mostra a diferença de erro entre o C0 e cada referência, definida na execução como a média, nas \
linhas da semana, do erro absoluto do C0 menos o da referência (negativa quando o C0 é melhor). Ela pesa linhas, e \
não pares, e é descritiva. O C0 foi melhor que o OLS agrupado nas 52 semanas (diferença média de −0,4372, de \
−0,4901 a −0,3245) e melhor que o OLS por par nas 52 semanas (média de −0,0339, de −0,0755 a −0,0004). Contra o \
GPBoost, o C0 foi melhor em 30 das 52 semanas, com média de −0,0032, num intervalo de −0,0659 a +0,0229.
""",
            en="""
The weekly series shows the error difference between C0 and each reference, defined in the run as the mean, over \
the rows of the week, of the absolute error of C0 minus that of the reference (negative when C0 is better). It \
weights rows, not pairs, and is descriptive. C0 was better than the pooled OLS in all 52 weeks (mean difference of \
−0.4372, from −0.4901 to −0.3245) and better than the per-pair OLS in all 52 weeks (mean of −0.0339, from −0.0755 \
to −0.0004). Against GPBoost, C0 was better in 30 of the 52 weeks, with a mean of −0.0032, in a range from −0.0659 \
to +0.0229.
""",
        ),
        md(
            pt="""
## 6. Primeira conclusão: o C0 foi escolhido pelo desempenho nas dobras, e as inclinações aleatórias para pares \
curtos continuam em aberto

O C0 foi selecionado pelo desempenho nas dobras internas. O C1 melhorou o estrato de histórico curto (menos de 26 \
semanas) nas duas dobras, mas falhou na regra global: piorou na dobra A, e o ganho médio de −0,49% ficou abaixo do \
limiar de 0,5%. A tabela repete os MAEs por par que sustentam a decisão.
""",
            en="""
## 6. First conclusion: C0 was selected on fold performance, and random slopes for short pairs remain open

C0 was selected on the performance in the internal folds. C1 improved the short-history stratum (fewer than 26 \
weeks) in both folds, but failed the global rule: it was worse in fold A, and the mean gain of −0.49% fell below \
the 0.5% threshold. The table repeats the per-pair MAEs that support the decision.
""",
        ),
        code("""
display(interpret.metric_table(folds_m, "a_seen", LANG, models=["C0", "C1"]))
short = {
    f: interpret.relative_difference(folds_m, "a_seen", "<26", "C1", "C0", fold=f) for f in ("A", "B")
}
label = f"{t('model_c1', LANG)} vs. {t('model_c0', LANG)}, {t('stratum_lt26', LANG)} (%)"
display(pd.Series(short, name=label).to_frame())
"""),
        md(
            pt="""
No estrato com menos de 26 semanas, o C1 teve MAE por par 3,70% menor que o do C0 na dobra A e 0,85% menor na \
dobra B. Esse estrato reúne 246 pares na dobra A e 221 na dobra B, e as diferenças são pontuais, sem análise de \
incerteza. Na dobra A, o C1 piorou o estrato de 26 a 77 semanas, que concentra a maior parte dos pares, e o ganho \
global ficou negativo.

A janela final avaliou o C0, que compartilha informação pelos interceptos e tem uma inclinação de desconto **comum** \
a todos os pares. Por isso, a janela final deixa em aberto a questão das inclinações aleatórias para pares curtos: o \
C1 não foi pontuado nela, por determinação do protocolo, e o resultado da janela final não informa se o ganho do C1 \
nos pares curtos das dobras se manteria.
""",
            en="""
In the stratum with fewer than 26 weeks, C1 had a per-pair MAE 3.70% lower than that of C0 in fold A and 0.85% \
lower in fold B. This stratum holds 246 pairs in fold A and 221 in fold B, and the differences are point values, \
without an uncertainty analysis. In fold A, C1 worsened the 26 to 77 weeks stratum, which holds most of the pairs, \
and the overall gain was negative.

The final window evaluated C0, which shares information through the intercepts and has a discount slope that is \
**common** to all pairs. The final window therefore leaves the question of random slopes for short pairs open: C1 \
was not scored there, by the protocol, and the result of the final window does not say whether the gain of C1 on \
the short pairs of the folds would hold.
""",
        ),
        md(
            pt="""
## 7. Segunda conclusão: C0 e GPBoost tiveram resultados próximos, que diferem conforme a métrica

Nos pares vistos, o GPBoost teve MAE por par 0,47% menor que o do C0 (0,4520 contra 0,4541). O C0, por sua vez, \
teve o menor MAE por linha (0,4446 contra 0,4478) e o menor erro nos pares inéditos (0,6260 contra 0,6876). A \
tabela reúne essas medidas.
""",
            en="""
## 7. Second conclusion: C0 and GPBoost performed closely, with results that differ by metric

On the seen pairs, GPBoost had a per-pair MAE 0.47% lower than that of C0 (0.4520 against 0.4541). C0, in turn, \
had the lowest MAE per row (0.4446 against 0.4478) and the lowest error on the unseen pairs (0.6260 against \
0.6876). The table brings these measures together.
""",
        ),
        code("""
specs = [
    ("a_seen", "all", "mae_pair_mean"),
    ("a_seen", "all", "mae_row"),
    ("c_unseen", "all", "mae_pair_mean"),
    ("c_unseen", "all", "mae_row"),
]
display(interpret.model_comparison(final_m, LANG, specs, ["C0", "GPBoost"], rel=("GPBoost", "C0")))
display(interpret.metric_table(final_m, "a_seen", LANG, models=["C0", "GPBoost"]))
"""),
        md(
            pt="""
A diferença é apresentada como **observada**. A coluna de diferença mostra o GPBoost 0,47% abaixo do C0 no MAE por \
par dos pares vistos, e 0,72% acima no MAE por linha. Nos pares inéditos, o GPBoost ficou 9,84% acima no MAE por par. \
A vantagem do GPBoost no MAE por par concentra-se nos estratos de histórico curto e médio, e a do C0 no estrato de 78 \
semanas ou mais. Na série semanal, o C0 foi melhor em 30 das 52 semanas, com diferença média de −0,0032.

Não foi feita análise de incerteza para essa comparação: o bootstrap de pares está adiado, e a série semanal é \
descritiva. Por isso, as diferenças não são declaradas equivalentes nem significativas. Além disso, o GPBoost foi \
calibrado nas dobras internas com subamostra de 50% e no máximo 300 rodadas, e com parâmetros nas bordas do espaço de \
busca (seção 4), e a vantagem dele nas dobras (cerca de −1,9%) foi maior que a observada na janela final (−0,47%).
""",
            en="""
The difference is presented as **observed**. The difference column shows GPBoost 0.47% below C0 in the per-pair \
MAE of the seen pairs, and 0.72% above in the MAE per row. On the unseen pairs, GPBoost was 9.84% above in the \
per-pair MAE. The advantage of GPBoost in the per-pair MAE is concentrated in the short and medium history strata, \
and that of C0 in the stratum of 78 weeks or more. In the weekly series, C0 was better in 30 of the 52 weeks, with \
a mean difference of −0.0032.

No uncertainty analysis was run for this comparison: the pair bootstrap is deferred, and the weekly series is \
descriptive. The differences are therefore not declared equivalent or significant. In addition, GPBoost was tuned \
on the internal folds with a 50% subsample and at most 300 rounds, and with parameters at the edges of the search \
space (section 4), and its advantage on the folds (about −1.9%) was larger than that observed on the final window \
(−0.47%).
""",
        ),
        md(
            pt="""
## 8. Terceira conclusão: o resultado nos pares curtos depende da população e da ponderação

Nos 59 pares curtos (menos de 26 semanas) cobertos pelo OLS local, o MAE por par com peso igual foi de 0,4002 para o \
GPBoost, 0,4063 para o OLS local e 0,4773 para o C0. No MAE por linha, nesse mesmo subconjunto, o C0 teve erro menor \
que o do OLS local (0,4804 contra 0,5255), e o GPBoost teve 0,4888.
""",
            en="""
## 8. Third conclusion: the short-pair result depends on the population and on the weighting

In the 59 short pairs (fewer than 26 weeks) covered by the local OLS, the equal-weight per-pair MAE was 0.4002 for \
GPBoost, 0.4063 for the local OLS and 0.4773 for C0. In the MAE per row, in that same subset, C0 had a lower error \
than the local OLS (0.4804 against 0.5255), and GPBoost had 0.4888.
""",
        ),
        code("""
specs = [("b_ols_covered", "<26", "mae_pair_mean"), ("b_ols_covered", "<26", "mae_row")]
display(interpret.model_comparison(final_m, LANG, specs, ["C0", "GPBoost", "OLS_pair"]))
c0 = final_m[(final_m["model"] == "C0") & (final_m["stratum"] == "<26")].set_index("population")
display(
    pd.DataFrame(
        {
            t("pop_a_seen", LANG): c0.loc["a_seen", ["n_pairs", "n_rows"]],
            t("pop_b_ols_covered", LANG): c0.loc["b_ols_covered", ["n_pairs", "n_rows"]],
        }
    ).rename(index={"n_pairs": t("col_pairs", LANG), "n_rows": t("col_rows", LANG)})
)
"""),
        md(
            pt="""
**Ressalva de cobertura (E2).** A cobertura do OLS por par segue a regra operacional aplicada na execução: um termo \
sem variação na janela de ajuste do par, ou descartado por colinearidade, conta como não estimado, e uma linha fica \
sem cobertura quando tem valor diferente de zero nesse termo. Só 59 dos 98 pares curtos vistos são cobertos, de \
modo que o subconjunto é selecionado pela própria regra do OLS local e não é uma amostra aleatória dos pares curtos. \
A regra operacional não equivale a um critério de estimabilidade. Se uma promoção esteve sempre ativa no treino de \
um par, ou sempre acompanhou o mesmo padrão de desconto, o intercepto ou os outros termos absorvem o seu nível \
médio, mas o ajuste fica sem base para estimar o que acontece quando ela é desativada ou quando essa relação muda. \
Um critério de estimabilidade, que só cobre a linha quando o seu vetor de covariáveis está no espaço gerado pelo \
desenho de treino do par, mudaria a classificação de 175 das 176.529 linhas de pares vistos na janela final, em \
63 pares: 130 hoje cobertas deixariam de sê-lo e 45 hoje sem cobertura passariam a ser cobertas. Todas as mudanças \
vêm de termos descartados por colinearidade; o caso de termo constante não altera nenhuma linha nesta janela. Os \
resultados gravados foram preservados com a regra operacional.

O ordenamento depende, portanto, da população e da ponderação. Com peso igual por par, o OLS local (0,4063) foi \
melhor que o C0 (0,4773) nos 59 pares curtos cobertos, e o GPBoost (0,4002) foi o melhor dos três. Com peso igual por \
linha, no mesmo subconjunto, o C0 (0,4804) foi melhor que o OLS local (0,5255). As duas ponderações respondem a \
perguntas diferentes, e os artefatos gravados não permitem dizer quais pares sustentam a diferença. Nas dobras \
internas, em contraste, o C0 teve MAE por par menor que o do OLS por par também no estrato curto das linhas cobertas \
(seção 3 mostra a população b); a janela final não reproduziu essa ordem nos 59 pares. Nesse subconjunto, portanto, a \
ideia de que o encolhimento do C0 favorece sobretudo os pares de pouco histórico não se confirmou para o C0 no MAE \
por par, e a conclusão não se estende aos pares curtos sem cobertura do OLS local.
""",
            en="""
**Coverage caveat (E2).** Per-pair OLS coverage follows the operational rule applied in the run: a term without \
variation in the pair's fit window, or dropped by collinearity, counts as not estimated, and a row is uncovered when \
it has a non-zero value on that term. Only 59 of the 98 short seen pairs are covered, so the subset is selected by \
the local OLS's own rule and is not a random sample of the short pairs. The operational rule is not equivalent to an \
estimability criterion. If a promotion was always active in a pair's training window, or always came with the same \
discount pattern, the intercept or the other terms absorb its average level, but the fit has no basis to estimate \
what happens when it is switched off or when that relation changes. An estimability criterion, which covers a row \
only when its covariate vector lies in the space spanned by the pair's training design, would change the \
classification of 175 of the 176,529 seen-pair rows in the final window, in 63 pairs: 130 currently covered rows \
would become uncovered and 45 currently uncovered rows would become covered. All changes come from terms dropped by \
collinearity; the constant-term case changes no row in this window. The stored results were preserved with the \
operational rule.

The ranking therefore depends on the population and on the weighting. With equal weight per pair, the local OLS \
(0.4063) was better than C0 (0.4773) in the 59 covered short pairs, and GPBoost (0.4002) was the best of the three. \
With equal weight per row, in the same subset, C0 (0.4804) was better than the local OLS (0.5255). The two \
weightings answer different questions, and the stored artefacts do not say which pairs sustain the difference. On \
the internal folds, in contrast, C0 had a lower per-pair MAE than the per-pair OLS also in the short stratum of the \
covered rows (section 3 shows population b); the final window did not reproduce that order in the 59 pairs. In \
that subset, therefore, the idea that the shrinkage of C0 helps mainly the pairs with little history was not \
confirmed for C0 in the per-pair MAE, and the conclusion does not extend to the short pairs without local OLS \
coverage.
""",
        ),
        md(
            pt="""
## 9. Teste complementar das inclinações aleatórias no treino

O notebook 02 usou a razão de verossimilhanças para comparar as estruturas de interceptos aleatórios. Aqui foi \
feita outra comparação, entre C0 e C1, para a hipótese conjunta de que a variância da inclinação do desconto \
por loja e a variância por UPC sejam iguais a zero. Os dois modelos foram ajustados por máxima verossimilhança, com \
a mesma parte fixa, os mesmos interceptos aleatórios e as mesmas 347.092 linhas das semanas 0 a 103.

Sob a hipótese nula, as duas variâncias estão na fronteira (Self & Liang, 1987; Stram & Lee, 1994). O teste foi \
calibrado por bootstrap paramétrico sob o C0 ajustado, seguindo o procedimento de simular sob o modelo restrito \
para calibrar a razão de verossimilhanças em modelos mistos (Halekoh & Højsgaard, 2014; Ott, 2014). Em cada uma \
das 39 réplicas planejadas, \
foi simulada uma resposta no mesmo desenho e os dois modelos foram reajustados a essa resposta. A estatística \
é duas vezes a diferença de log-verossimilhança entre C1 e C0. O valor-p de Monte Carlo é o número de \
réplicas com estatística igual ou maior que a observada, acrescido de um, dividido pelo total de réplicas \
acrescido de um. Os resultados abaixo foram lidos da execução gravada, sem reexecutar os ajustes.
""",
            en="""
## 9. Supplementary test of random slopes in training

Notebook 02 used likelihood-ratio tests to compare random-intercept structures. Here, C0 and C1 were compared \
for the joint hypothesis that the variance of the discount slope across stores and the variance across UPCs \
are both zero. Both models were fitted by maximum likelihood with the same fixed effects, the same random \
intercepts and the same 347,092 rows from weeks 0 to 103.

Under the null hypothesis, both variances are on the boundary (Self & Liang, 1987; Stram & Lee, 1994). The test \
was calibrated by parametric bootstrap under fitted C0, following the procedure of simulating under the restricted \
model to calibrate the likelihood ratio in mixed models (Halekoh & Højsgaard, 2014; Ott, 2014). In each of the \
39 planned replicates, \
a response was simulated on the same design and both models were refitted to that response. The statistic \
is twice the difference in log-likelihood between C1 and C0. The Monte Carlo p-value is the number of \
replicates with a statistic at least as large as the observed one, plus one, divided by the total number of \
replicates plus one. The results below were read from the stored run without refitting the models.
""",
        ),
        code("""
slope_test = load_run("discount_slope_lr", "20261125-112223-aa33d3e6")
display(interpret.discount_slope_lr_table(slope_test, LANG))
"""),
        md(
            pt="""
A LR observada foi **12.974,40**, e a maior LR simulada foi **5,18**. Nenhuma das 39 réplicas válidas atingiu \
a estatística observada. O valor-p de Monte Carlo foi **0,025**, a menor resolução do desenho. Sob as hipóteses \
do modelo, o teste conjunto rejeita a 5% que as duas variâncias sejam zero. Isso sustenta que pelo menos uma \
das duas inclinações varia entre grupos. Os desvios-padrão estimados separadamente por loja e UPC descrevem o \
ajuste do C1, e a decisão inferencial é conjunta.

A simulação reproduz a estrutura residual assumida pelo C0. A autocorrelação entre semanas do mesmo par, \
diagnosticada no notebook 03, limita a interpretação do valor-p. O teste documenta o ganho de ajuste no \
treino. Para previsão, o P1 manteve o C0 pelo erro nas dobras temporais. As semanas 104 a 155 ficaram fora \
deste teste complementar.
""",
            en="""
The observed LR was **12,974.40**, and the largest simulated LR was **5.18**. None of the 39 valid \
replicates reached the observed statistic. The Monte Carlo p-value was **0.025**, the finest resolution \
of the design. Under the model assumptions, the joint test rejects at 5% that both variances are zero. \
This supports variation in at least one of the two slopes across groups. The separate standard deviation \
estimates for stores and UPCs describe the C1 fit, and the inferential decision is joint.

The simulation reproduces the residual structure assumed by C0. The residual autocorrelation across weeks \
within pairs, diagnosed in notebook 03, limits the interpretation of the p-value. The test documents the \
improvement in fit on training data. For prediction, P1 kept C0 based on error in the temporal folds. \
Weeks 104 to 155 were outside this supplementary test.
""",
        ),
        md(
            pt="""
## 10. O que permanece adiado

Permanecem adiados a pontuação do C1 na janela final, que seria uma exploração posterior e exige aprovação, e o \
bootstrap por pares para a incerteza da diferença de erro preditivo entre os modelos. As análises que este notebook \
deixava em aberto já foram concluídas, todas nas semanas de treino e sem alterar o C0 nem o ranking registrado \
aqui. O notebook 05 executou os contrastes secundários do P1, as interações do desconto com segmento, marca \
própria e categoria, com correção de Holm (Holm, 1979), e a sensibilidade com o efeito aleatório de categoria × \
semana. O notebook 06 executou a camada de decisão por cenários.

Em resumo, a seleção pré-registrada manteve o C0, e a janela final, que é retrospectiva e teve exposição \
exploratória prévia, mostrou o C0 melhor que os dois OLS no agregado e próximo do GPBoost, com \
resultados que diferem conforme a métrica e a população. A pergunta sobre as inclinações aleatórias para pares \
curtos continua em aberto.
""",
            en="""
## 10. What remains deferred

Still deferred are the scoring of C1 on the final window, which would be a later exploration and needs approval, \
and the pair bootstrap for uncertainty in the difference in predictive error between models. The analyses \
that this notebook left open have since been completed, all on the training weeks and without changing C0 or the \
ranking recorded here. Notebook 05 ran the secondary contrasts of P1, the interactions of the discount with \
segment, private label and category, with the Holm correction (Holm, 1979), and the sensitivity with the category \
× week random effect. Notebook 06 ran the decision layer by scenarios.

In summary, the pre-registered selection kept C0, and the final window, which is retrospective and had prior \
exploratory exposure, showed C0 better than both OLS references in the aggregate and \
close to GPBoost, with results that \
differ by metric and by population. The question about random slopes for short pairs remains open.
""",
        ),
    ]


SETUP_05 = """
import warnings

import pandas as pd
from IPython.display import display

from retail_multilevel import viz
from retail_multilevel.i18n import contrast_label, format_p, level_label, localize_columns, t
from retail_multilevel.runs import load_run

warnings.filterwarnings("ignore", category=FutureWarning)
viz.use_retina()
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:,.4f}".format)
contrasts = load_run("p2_contrasts", "20261124-220904-eb35f45e")
catweek = load_run("p2_catweek", "20261124-220500-f7c26a70")
print(t("data_credit", LANG))
stored = {"contrasts": contrasts, "catweek": catweek}
display(pd.DataFrame({k: {"run_id": r.run_id, "finished_at": r.metadata["finished_at"]} for k, r in stored.items()}).T)

MODELS = {"F3_ref": t("model_f3_ref", LANG), "F3_catweek": t("model_f3_catweek", LANG)}
GROUPS = {"CATWEEK": t("group_catweek", LANG)}


def localized(df):
    out = df.copy()
    if "test" in out:
        for col in ("level", "reference"):
            if col in out:
                out[col] = [level_label(a, b, LANG) for a, b in zip(out["test"], out[col], strict=True)]
        out["test"] = [contrast_label(x, LANG) for x in out["test"]]
    if "model" in out:
        out["model"] = out["model"].map(lambda x: MODELS.get(x, x))
    if "group" in out:
        out["group"] = out["group"].map(lambda x: GROUPS.get(x, viz.group_label(x, LANG)))
    if "term" in out:
        out["term"] = [viz.term_label(x, LANG) for x in out["term"]]
    return localize_columns(out, LANG)
"""


def contrasts_cells() -> list[Cell]:
    """Cells of notebook 05, the planned contrasts and the category x week sensitivity read from stored runs."""
    return [
        md(
            pt="""
# 05 · Contrastes planejados e sensibilidade a choques de categoria × semana

Este notebook reúne duas análises secundárias do projeto, as duas motivadas pelos resultados dos notebooks 03 e 04. A primeira examina se a associação entre desconto e vendas difere entre segmentos de loja, entre marca de fabricante e marca própria e entre categorias, por meio de três contrastes planejados, com correção de Holm sobre essa família fixa de três testes. A segunda compara o F3 do notebook 03 com uma versão que acrescenta um intercepto aleatório de categoria × semana, como sensibilidade motivada pelo diagnóstico residual do notebook 03.

Este notebook **não ajusta modelos**: os ajustes foram feitos por scripts (`scripts/fit_contrasts.py` e `scripts/fit_catweek_sensitivity.py`) e gravados em `outputs/runs`. O notebook apenas lê essas duas execuções, o que torna as versões em português e em inglês idênticas em resultados. Somente as 104 semanas de treino (semanas 0 a 103) entram nos cálculos, e as estimativas são associações condicionais em semanas com transações registradas, e não efeitos causais. A especificação oficial do C0 e o ranking preditivo gravado no protocolo P1 permanecem como foram registrados.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 05 · Planned contrasts and sensitivity to category × week shocks

This notebook gathers two secondary analyses of the project, both motivated by the results of notebooks 03 and 04. The first examines whether the association between discount and sales differs across store segments, between manufacturer brand and private label, and across categories, through three planned contrasts, with a Holm correction over this fixed family of three tests. The second compares the F3 of notebook 03 with a version that adds a random category × week intercept, as a sensitivity analysis motivated by the residual diagnostic of notebook 03.

This notebook **fits no models**: the fits were made by scripts (`scripts/fit_contrasts.py` and `scripts/fit_catweek_sensitivity.py`) and stored in `outputs/runs`. The notebook only reads these two runs, so the Portuguese and English versions show identical results. Only the 104 training weeks (weeks 0 to 103) enter the computations, and the estimates are conditional associations in weeks with recorded transactions, not causal effects. The official C0 specification and the predictive ranking recorded in protocol P1 stay as they were registered.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_05),
        md(
            pt="""
As duas execuções lidas estão identificadas acima (identificador e horário de término). Os identificadores são fixados no código, de modo que reexecutar o notebook lê sempre os mesmos artefatos.

## 1. A pergunta e o desenho dos contrastes

A pergunta é se a inclinação do log das unidades em relação ao log do desconto (`LOG_DISC`) é a mesma em todos os níveis de um moderador. Os moderadores são o segmento da loja (referência `MAINSTREAM`), o tipo de marca (referência: marca de fabricante) e a categoria (referência `BAG SNACKS`, a primeira em ordem alfabética). Para cada moderador foi ajustado um modelo próprio, por mínimos quadrados com efeitos fixos de par loja × produto e de semana, obtidos pela transformação *within* (Wooldridge, 2010). A resposta é `LOG_UNITS`, e os controles são `LOG_DISC`, `LOG_BASE_PRICE`, `FEATURE`, `DISPLAY` e `TPR_ONLY`, os mesmos da sensibilidade intrapar do notebook 03. O modelo com interação acrescenta `LOG_DISC` multiplicado pelo indicador de cada nível diferente da referência, e a versão restrita, com a mesma amostra e os mesmos controles, omite essas interações.

Um ponto de identificação merece registro. O segmento, o tipo de marca e a categoria são constantes dentro de cada par loja × produto, de modo que o efeito principal do moderador é absorvido pelo efeito fixo do par e não é estimado. A interação com `LOG_DISC` permanece identificada, porque o desconto varia ao longo das semanas dentro do par: a diferença de inclinação é estimada comparando, entre níveis, como as vendas de um mesmo par respondem às próprias variações de desconto.

Uma diferença de inclinação descreve a heterogeneidade condicional de uma associação entre grupos. Ela responde a uma pergunta distinta da que motivou o candidato C1 do protocolo P1 (notebook 04), no qual a elasticidade ao desconto varia aleatoriamente entre lojas e produtos (Gelman & Hill, 2007) e é avaliada pelo erro de previsão fora da amostra. Nenhum resultado deste notebook altera a seleção preditiva do P1 nem o ranking gravado nele.

## 2. As populações e o suporte por nível

A primeira questão operacional é quais linhas entram em cada teste. A tabela a seguir mostra, para cada moderador, as linhas do treino, as linhas usadas, os pares e as exclusões.
""",
            en="""
The two runs that were read are identified above (identifier and finish time). The identifiers are fixed in the code, so re-running the notebook always reads the same artefacts.

## 1. The question and the design of the contrasts

The question is whether the slope of log(units) on the log of the discount (`LOG_DISC`) is the same at every level of a moderator. The moderators are the store segment (reference `MAINSTREAM`), the brand type (reference: manufacturer brand) and the category (reference `BAG SNACKS`, the first in alphabetical order). One model was fitted for each moderator, by least squares with store × product pair and week fixed effects, obtained through the within transformation (Wooldridge, 2010). The response is `LOG_UNITS`, and the controls are `LOG_DISC`, `LOG_BASE_PRICE`, `FEATURE`, `DISPLAY` and `TPR_ONLY`, the same as in the within-pair sensitivity of notebook 03. The model with interaction adds `LOG_DISC` multiplied by the indicator of each level other than the reference, and the restricted version, with the same sample and the same controls, omits those interactions.

One identification point deserves a record. Segment, brand type and category are constant within each store × product pair, so the main effect of the moderator is absorbed by the pair fixed effect and is not estimated. The interaction with `LOG_DISC` remains identified, because the discount varies over the weeks within the pair: the slope difference is estimated by comparing, across levels, how the sales of a given pair respond to its own discount variations.

A slope difference describes the conditional heterogeneity of an association between groups. It answers a question distinct from the one that motivated candidate C1 of protocol P1 (notebook 04), in which the discount elasticity varies randomly across stores and products (Gelman & Hill, 2007) and is judged by out-of-sample prediction error. No result of this notebook changes the predictive selection of P1 or the ranking recorded in it.

## 2. Populations and support by level

The first operational question is which rows enter each test. The table below shows, for each moderator, the training rows, the rows used, the pairs and the exclusions.
""",
        ),
        code("""
display(localized(contrasts.load_table("populations")))
"""),
        md(
            pt="""
O teste por segmento usa 338.033 linhas e 3.763 pares, depois da exclusão de 9.059 linhas das duas lojas de segmento ambíguo (4503 e 17627); não há segmento ausente. Os testes por tipo de marca e por categoria usam as 347.092 linhas e os 3.859 pares do treino, sem casos sem classificação. As referências de cada fator foram fixadas antes da inspeção das estimativas.

A segunda questão é se há variação de desconto dentro do par em cada nível, porque é ela que identifica a interação. A tabela seguinte mostra, por nível, as linhas, os pares, o desvio-padrão intrapar de `LOG_DISC` e o número de pares com variação de desconto.
""",
            en="""
The segment test uses 338,033 rows and 3,763 pairs, after the exclusion of 9,059 rows from the two stores with an ambiguous segment (4503 and 17627); no segment is missing. The brand-type and category tests use the 347,092 rows and the 3,859 pairs of the training sample, with no unclassified cases. The reference of each factor was fixed before the estimates were inspected.

The second question is whether there is discount variation within the pair at each level, because that variation is what identifies the interaction. The next table shows, by level, the rows, the pairs, the within-pair standard deviation of `LOG_DISC` and the number of pairs with discount variation.
""",
        ),
        code("""
display(localized(contrasts.load_table("support")))
"""),
        md(
            pt="""
Todos os níveis têm variação de desconto dentro do par. A proporção de pares com variação de `LOG_DISC` vai de 96,3% (VALUE, com 887 de 921 pares) a 100% (COLD CEREAL, com 1.155 de 1.155 pares), e o desvio-padrão intrapar do desconto vai de 0,101 (BAG SNACKS) a 0,175 (ORAL HYGIENE PRODUCTS). Há, portanto, variação de onde extrair cada inclinação, embora a distribuição dos descontos difira entre os níveis.

## 3. Os testes globais e a correção de Holm

Para cada moderador, a hipótese nula conjunta é que todas as diferenças de inclinação em relação à referência são zero. A estatística é um Wald conjunto das q interações, calculado com a matriz completa de covariância agrupada em duas vias, por par e por semana (Cameron et al., 2011). Como há 104 semanas, número menor que o de pares, a referência é uma F(q, 103), isto é, com `min(G_par, G_semana) − 1` graus de liberdade no denominador, uma convenção aproximada para o número de clusters da dimensão menor (Cameron & Miller, 2015). A estatística F é a de Wald dividida por q. A tabela mostra a estatística, o número de restrições, os graus de liberdade, a contagem de clusters e o valor-p bruto.
""",
            en="""
Every level has discount variation within the pair. The share of pairs with `LOG_DISC` variation ranges from 96.3% (VALUE, with 887 of 921 pairs) to 100% (COLD CEREAL, with 1,155 of 1,155 pairs), and the within-pair standard deviation of the discount ranges from 0.101 (BAG SNACKS) to 0.175 (ORAL HYGIENE PRODUCTS). There is, therefore, variation from which to extract each slope, although the distribution of discounts differs across levels.

## 3. The global tests and the Holm correction

For each moderator, the joint null hypothesis is that all slope differences relative to the reference are zero. The statistic is a joint Wald test of the q interactions, computed with the full covariance matrix clustered two ways, by pair and by week (Cameron et al., 2011). Because there are 104 weeks, fewer than the number of pairs, the reference is an F(q, 103), that is, with `min(G_pair, G_week) − 1` denominator degrees of freedom, an approximate convention for the number of clusters in the smaller dimension (Cameron & Miller, 2015). The F statistic is the Wald statistic divided by q. The table shows the statistic, the number of restrictions, the degrees of freedom, the cluster counts and the raw p-value.
""",
        ),
        code("""
wald = contrasts.load_table("wald")
tests = wald[["test", "reference", "q", "W", "F", "df1", "df2", "G_pair", "G_week"]].assign(
    p_raw=[format_p(p) for p in wald["p_raw"]]
)
display(localized(tests))
"""),
        md(
            pt="""
O teste por segmento (q = 2) tem F = 6,12 e valor-p bruto de 0,0031; o de tipo de marca (q = 1), F = 30,27 e valor-p de 2,76e-07; o de categoria (q = 3), F = 37,00 e valor-p de 2,61e-16. A confiabilidade desses valores depende de diagnósticos do desenho e da covariância, apresentados a seguir: o posto da matriz de desenho, o número de condição do bloco q × q da covariância das interações e seu menor autovalor, e a indicação de que uma correção para semidefinição positiva foi necessária.
""",
            en="""
The segment test (q = 2) has F = 6.12 and a raw p-value of 0.0031; the brand-type test (q = 1), F = 30.27 and a p-value of 2.76e-07; the category test (q = 3), F = 37.00 and a p-value of 2.61e-16. The reliability of these values depends on diagnostics of the design and of the covariance, presented next: the rank of the design matrix, the condition number of the q × q block of the covariance of the interactions and its smallest eigenvalue, and the indication of whether a correction for positive semidefiniteness was needed.
""",
        ),
        code("""
diagnostics = wald[
    ["test", "design_rank", "n_terms", "rank_deficient", "cond_number", "min_eig", "twoway_psd_fixed", "design_cond", "available"]
]
display(localized(diagnostics))
"""),
        md(
            pt="""
Os três desenhos têm posto completo (7, 6 e 8 colunas), nenhum precisou de correção para semidefinição positiva, e o bloco q × q da covariância tem número de condição de no máximo 6,0, com menor autovalor entre 0,0017 e 0,0081. Os três testes estão, portanto, disponíveis.

A correção de Holm é aplicada a uma família fixa de três testes globais, ao nível de 5% (Holm, 1979). Os valores-p brutos, em ordem crescente, são multiplicados por 3, 2 e 1, com imposição de monotonicidade. Se algum teste fosse indisponível, por perda de posto ou por covariância instável, a família continuaria com três posições, e o valor-p da posição indisponível entraria como 1 no ajuste, sendo exibido como indisponível. Aqui, nenhuma posição é indisponível.
""",
            en="""
The three designs have full rank (7, 6 and 8 columns), none needed a correction for positive semidefiniteness, and the q × q block of the covariance has a condition number of at most 6.0, with a smallest eigenvalue between 0.0017 and 0.0081. The three tests are therefore available.

The Holm correction is applied to a fixed family of three global tests, at the 5% level (Holm, 1979). The raw p-values, in ascending order, are multiplied by 3, 2 and 1, with monotonicity enforced. If a test were unavailable, because of lost rank or an unstable covariance, the family would keep three positions, and the p-value of the unavailable position would enter as 1 in the adjustment and be displayed as unavailable. Here, no position is unavailable.
""",
        ),
        code("""
holm = contrasts.load_table("holm")
holm_view = holm.assign(
    p_raw=[format_p(p) for p in holm["p_raw"]],
    p_holm=[format_p(p) if ok else t("p_unavailable", LANG) for p, ok in zip(holm["p_holm"], holm["available"], strict=True)],
)
display(localized(holm_view))
"""),
        md(
            pt="""
Depois do ajuste, os valores-p são 0,0031 para segmento, 5,51e-07 para tipo de marca e 7,83e-16 para categoria, e os três testes rejeitam a hipótese nula global a 5%. O ajuste não altera a decisão de nenhum dos três testes, e o de segmento, o de menor evidência da família, mantém o valor-p bruto por ocupar a última posição da ordenação.

Dois cuidados acompanham essa leitura. O primeiro é que o agrupamento em duas vias acomoda a dependência dentro do par, como a correlação serial entre semanas do mesmo par, e os choques contemporâneos compartilhados entre pares na mesma semana. A dependência fora dessas dimensões, em especial choques comuns persistentes entre semanas, não é coberta, e por isso o teste é tratado como aproximado. O segundo é que a covariância em duas vias é bem maior que a agrupada só por par, como mostra a tabela abaixo para o coeficiente de `LOG_DISC` no modelo restrito: o erro-padrão em duas vias, cerca de 0,079, é cerca de 1,8 vez o agrupado por par, cerca de 0,044, de modo que uma inferência baseada apenas no agrupamento por par seria otimista.
""",
            en="""
After the adjustment, the p-values are 0.0031 for segment, 5.51e-07 for brand type and 7.83e-16 for category, and the three tests reject the global null hypothesis at 5%. The adjustment changes the decision of none of the three tests, and the segment test, the one with the weakest evidence in the family, keeps its raw p-value because it occupies the last position of the ordering.

Two cautions accompany this reading. The first is that two-way clustering accommodates dependence within the pair, such as serial correlation between weeks of the same pair, and contemporaneous shocks shared among pairs in the same week. Dependence outside those dimensions, in particular persistent common shocks across weeks, is not covered, so the test is treated as approximate. The second is that the two-way covariance is much larger than the one clustered by pair only, as the table below shows for the `LOG_DISC` coefficient in the restricted model: the two-way standard error, about 0.079, is about 1.8 times the pair-clustered one, about 0.044, so an inference based only on pair clustering would be optimistic.
""",
        ),
        code("""
coef = contrasts.load_table("coefficients")
restricted = coef[(coef["model"] == "restricted") & (coef["term"] == "LOG_DISC")]
display(localized(restricted.drop(columns=["model", "term"]).rename(columns={"estimate": "slope"})))
"""),
        md(
            pt="""
## 4. As inclinações por nível

Dado que os testes globais rejeitam, a questão seguinte é descritiva: como são as inclinações do desconto em cada nível. Uma inclinação mais negativa indica uma resposta maior das unidades a uma mesma variação do desconto. A tabela mostra, por nível, a inclinação estimada, seus erros-padrão, o intervalo de 95%, a diferença em relação à referência e o intervalo dessa diferença. Todos os intervalos são pontuais, calculados com o erro-padrão em duas vias, sem ajuste para multiplicidade. Para esta entrega, a inferência se encerra na família planejada de três testes, e as diferenças por nível são apresentadas de forma descritiva, sem uma bateria de comparações par a par.
""",
            en="""
## 4. Slopes by level

Given that the global tests reject, the next question is descriptive: what the discount slopes look like at each level. A more negative slope indicates a larger response of units to the same discount change. The table shows, by level, the estimated slope, its standard errors, the 95% interval, the difference from the reference and the interval of that difference. All intervals are pointwise, computed with the two-way standard error, without adjustment for multiplicity. For this deliverable, inference ends at the planned family of three tests, and the level differences are presented descriptively, without a battery of pairwise comparisons.
""",
        ),
        code("""
slopes = contrasts.load_table("slopes")
display(localized(slopes.drop(columns=["interval"])))
fig = viz.plot_moderator_slopes(slopes, LANG)
"""),
        md(
            pt="""
Por segmento, as inclinações são −1,746 (MAINSTREAM, a referência), −1,580 (UPSCALE) e −1,686 (VALUE). A diferença de UPSCALE em relação à referência é de 0,166, com intervalo pontual de 0,072 a 0,261, e a de VALUE é de 0,060, com intervalo de −0,052 a 0,172, que contém zero. Entre as diferenças por nível do segmento, apenas a de UPSCALE tem intervalo pontual que exclui zero. Essa leitura é descritiva, porque o intervalo é pontual e não tem ajuste para multiplicidade. O teste global de segmento indica heterogeneidade no conjunto dos segmentos e, por si só, não identifica qual segmento difere.

Por tipo de marca, a inclinação é de −1,906 para marca de fabricante e de −1,411 para marca própria, uma diferença de 0,494 (intervalo de 0,316 a 0,673): a associação do desconto com as unidades é menos intensa para a marca própria. Por categoria, as inclinações são −1,083 (BAG SNACKS, a referência), −2,044 (COLD CEREAL), −2,416 (FROZEN PIZZA) e −1,387 (ORAL HYGIENE PRODUCTS), com diferenças em relação à referência de −0,961 (intervalo de −1,215 a −0,708), −1,333 (de −1,603 a −1,064) e −0,305 (de −0,458 a −0,152). O desconto está associado a uma resposta maior em pizza congelada e em cereal do que em pretzels.

Esses números descrevem diferenças condicionais de inclinação, e não explicam por que elas ocorrem. Parte delas pode refletir características dos níveis que o modelo não representa, como a distribuição dos descontos praticados (a seção 2 mostrou que o desvio-padrão intrapar do desconto varia entre os níveis) e o calendário promocional. A inclinação comum de `LOG_DISC` no OLS intrapar restrito, de cerca de −1,70, resume essas inclinações nesse estimador, com efeitos fixos de par e de semana e o log do preço-base como controle. Ela não é a inclinação do F3, de −1,685, estimada pelo modelo multinível com interceptos aleatórios, com o preço-base decomposto em desvio e média do par e com tendência e sazonalidade. As duas são próximas, mas vêm de estimadores e controles diferentes, e os cenários do notebook 06 usam a do F3.

## 5. Sensibilidade a choques comuns de categoria × semana

O diagnóstico residual do notebook 03 mostrou que as médias semanais dos resíduos condicionais do F3 variam muito mais do que o esperado sob independência entre linhas, e que a parcela da variância residual associada à combinação de categoria e semana é de 6,4%, contra 0,1% esperado. A questão é se as associações estimadas pelo F3 são sensíveis à representação desses choques comuns. A comparação é, portanto, uma sensibilidade motivada pelo diagnóstico do notebook 03, e não uma troca da especificação oficial.

A referência é o F3 do notebook 03, o C0 do P1, com a parte fixa das decisões D12 a D14 e interceptos aleatórios de loja, produto e par, ajustado por REML com o `lme4` (Bates et al., 2015). A sensibilidade acrescenta `(1 | CATEGORY:WEEK)`, nas mesmas linhas, com a mesma parte fixa e as mesmas médias aprendidas no treino. Um intercepto de categoria × semana representa desvios contemporâneos, comuns a todas as lojas e produtos de uma categoria numa mesma semana. Os efeitos específicos de categoria × semana são condicionais às células já observadas no treino, e não se projetam para semanas futuras. As tabelas a seguir mostram as contagens e a convergência dos dois ajustes.
""",
            en="""
By segment, the slopes are −1.746 (MAINSTREAM, the reference), −1.580 (UPSCALE) and −1.686 (VALUE). The UPSCALE difference from the reference is 0.166, with a pointwise interval from 0.072 to 0.261, and the VALUE difference is 0.060, with an interval from −0.052 to 0.172, which contains zero. Among the level differences of the segment, only that of UPSCALE has a pointwise interval that excludes zero. This reading is descriptive, because the interval is pointwise and has no multiplicity adjustment. The global segment test indicates heterogeneity across the set of segments and does not, by itself, identify which segment differs.

By brand type, the slope is −1.906 for manufacturer brand and −1.411 for private label, a difference of 0.494 (interval from 0.316 to 0.673): the association of the discount with units is less intense for the private label. By category, the slopes are −1.083 (BAG SNACKS, the reference), −2.044 (COLD CEREAL), −2.416 (FROZEN PIZZA) and −1.387 (ORAL HYGIENE PRODUCTS), with differences from the reference of −0.961 (interval from −1.215 to −0.708), −1.333 (from −1.603 to −1.064) and −0.305 (from −0.458 to −0.152). The discount is associated with a larger response in frozen pizza and in cereal than in pretzels.

These numbers describe conditional slope differences, and do not explain why they occur. Part of them may reflect characteristics of the levels that the model does not represent, such as the distribution of the discounts applied (section 2 showed that the within-pair standard deviation of the discount varies across levels) and the promotional calendar. The common `LOG_DISC` slope of the restricted within-pair OLS, of about −1.70, summarises these slopes in that estimator, with pair and week fixed effects and the log base price as a control. It is not the F3 slope of −1.685, which is estimated by the multilevel model with random intercepts, with the base price split into the pair deviation and pair mean and with trend and seasonality. The two are close, but they come from different estimators and controls, and the scenarios of notebook 06 use the F3 one.

## 5. Sensitivity to common category × week shocks

The residual diagnostic of notebook 03 showed that the weekly means of the conditional residuals of the F3 vary much more than expected under independence between rows, and that the share of the residual variance associated with the combination of category and week is 6.4%, against 0.1% expected. The question is whether the associations estimated by the F3 are sensitive to the representation of these common shocks. The comparison is therefore a sensitivity analysis motivated by the diagnostic of notebook 03, not a replacement of the official specification.

The reference is the F3 of notebook 03, the C0 of P1, with the fixed part of decisions D12 to D14 and random intercepts for store, product and pair, fitted by REML with `lme4` (Bates et al., 2015). The sensitivity adds `(1 | CATEGORY:WEEK)`, on the same rows, with the same fixed part and the same means learned on the training sample. A category × week intercept represents contemporaneous deviations, common to all stores and products of a category in the same week. The category × week effects are conditional on the cells already observed in training, and do not project onto future weeks. The tables below show the counts and the convergence of the two fits.
""",
        ),
        code("""
display(localized(catweek.load_table("counts")).T)
display(localized(catweek.load_table("convergence")))
"""),
        md(
            pt="""
Os dois ajustes usam exatamente as mesmas 347.092 linhas, com 3.859 pares, 77 lojas, 55 produtos, 4 categorias e 416 níveis de categoria × semana (4 categorias em 104 semanas), sem categoria ausente. Ambos convergiram com o otimizador `bobyqa`, sem mensagens de convergência, e nenhum é singular, de modo que não há limitação a registrar e a comparação pode ser lida diretamente. A tabela seguinte compara os efeitos fixos.
""",
            en="""
The two fits use exactly the same 347,092 rows, with 3,859 pairs, 77 stores, 55 products, 4 categories and 416 category × week levels (4 categories over 104 weeks), with no missing category. Both converged with the `bobyqa` optimizer, with no convergence messages, and neither is singular, so there is no limitation to record and the comparison can be read directly. The next table compares the fixed effects.
""",
        ),
        code("""
fixef_compare = catweek.load_table("fixef_compare")
display(localized(fixef_compare))
fig = viz.plot_catweek_compare(fixef_compare, LANG)
"""),
        md(
            pt="""
O coeficiente do desconto passa de −1,685 para −1,727, e o do preço regular dentro do par (`LOG_BASE_DEV`), de −1,592 para −1,660. Os coeficientes de encarte, display e TPR mudam menos de 0,007. O sinal e a ordem de grandeza de todas as associações se mantêm, e as mudanças em desconto e preço regular são de 2,5% e 4,3% do valor de referência. Há uma exceção na precisão: o erro-padrão da tendência (`TREND`) é 7,3 vezes maior, indo de 0,0018 para 0,0129, enquanto os demais ficam entre 0,98 e 1,03 vez os da referência. Como a tendência é uma função apenas da semana, ela é em grande parte representada pelos níveis de categoria × semana, e deixa de ser bem identificada quando eles entram no modelo. Os erros-padrão desta tabela são os do modelo, que o notebook 03 mostrou serem estreitos demais para a inferência completa.

A tabela seguinte mostra as variâncias estimadas dos dois ajustes.
""",
            en="""
The discount coefficient moves from −1.685 to −1.727, and that of the regular price within the pair (`LOG_BASE_DEV`), from −1.592 to −1.660. The feature, display and TPR coefficients change by less than 0.007. The sign and the order of magnitude of all associations are kept, and the changes in discount and regular price are 2.5% and 4.3% of the reference value. There is one exception in precision: the standard error of the trend (`TREND`) is 7.3 times larger, going from 0.0018 to 0.0129, while the others stay between 0.98 and 1.03 times those of the reference. Because the trend is a function of the week only, it is largely represented by the category × week levels, and stops being well identified when they enter the model. The standard errors in this table are the model-based ones, which notebook 03 showed to be too narrow for full inference.

The next table shows the estimated variances of the two fits.
""",
        ),
        code("""
display(localized(catweek.load_table("varcomp")))
"""),
        md(
            pt="""
A variância de categoria × semana é de 0,0179 (desvio-padrão de 0,134), o que corresponde a 1,4% da variância total, e a variância residual cai de 0,272 para 0,255. As variâncias de loja, de produto e de par quase não se alteram. A tabela seguinte resume os choques comuns nos resíduos condicionais: a variância das médias semanais, a esperada sob independência, a razão entre elas e a parcela da variância de categoria × semana.
""",
            en="""
The category × week variance is 0.0179 (standard deviation 0.134), which corresponds to 1.4% of the total variance, and the residual variance falls from 0.272 to 0.255. The store, product and pair variances barely change. The next table summarises the common shocks in the conditional residuals: the variance of the weekly means, the variance expected under independence, the ratio between them and the share of the category × week variance.
""",
        ),
        code("""
week_summary = catweek.load_table("week_summary")
columns = [
    "model", "var_week_means", "var_expected_independent", "variance_ratio",
    "autocorr_week_means", "share_category_week", "share_category_week_expected",
]  # fmt: skip
with pd.option_context("display.float_format", "{:,.4g}".format):
    display(localized(week_summary[columns]))
"""),
        md(
            pt="""
A razão da variância das médias semanais frente à esperada sob independência cai de 97,2 na referência para 0,03 com o intercepto de categoria × semana, e a parcela de categoria × semana nos resíduos cai de 6,4% para 0,002%. Esse resultado é em parte mecânico, porque os resíduos condicionais são calculados depois de subtrair os efeitos previstos de categoria × semana: ele mostra que o termo absorve os choques comuns contemporâneos, e não que a dependência desapareceu. A autocorrelação de lag 1 das médias semanais dos resíduos já era próxima de zero na referência (−0,005), de modo que não há indício de persistência forte dos choques semanais agregados.

A persistência dentro do par é outra coisa. A tabela e o gráfico seguintes comparam as correlações dos resíduos condicionais do mesmo par em defasagens de calendário de 1 a 4 semanas.
""",
            en="""
The ratio of the variance of the weekly means to the one expected under independence falls from 97.2 in the reference to 0.03 with the category × week intercept, and the category × week share of the residuals falls from 6.4% to 0.002%. This result is partly mechanical, because the conditional residuals are computed after subtracting the predicted category × week effects: it shows that the term absorbs the contemporaneous common shocks, not that the dependence has vanished. The lag-1 autocorrelation of the weekly means of the residuals was already close to zero in the reference (−0.005), so there is no sign of strong persistence in the aggregate weekly shocks.

Persistence within the pair is a different matter. The following table and chart compare the correlations of the conditional residuals of the same pair at calendar lags of 1 to 4 weeks.
""",
        ),
        code("""
lag_corr = catweek.load_table("lag_corr")
wide = lag_corr.pivot(index="lag", columns="model", values="corr")[list(MODELS)].rename(columns=MODELS)
display(wide.rename_axis(t("f_lag", LANG)))
fig = viz.plot_lag_corr_compare(lag_corr, LANG)
"""),
        md(
            pt="""
As correlações mudam pouco: na defasagem 1, de 0,196 para 0,194, e nas defasagens de 2 a 4 em no máximo 0,008. A persistência residual dentro do par não é um choque contemporâneo comum a uma categoria, e por isso o termo de categoria × semana não a representa. Nos testes da seção 3, essa persistência é acomodada pelo agrupamento por par, que admite correlação arbitrária entre as semanas de um mesmo par.

Resta registrar a estatística de razão de verossimilhanças entre os dois ajustes REML, que têm a mesma parte fixa. Como se acrescenta uma única variância, o valor-p vem da mistura 50:50 de χ² com 0 e com 1 grau de liberdade (Self & Liang, 1987). O valor-p não é representável em ponto flutuante e é exibido como menor que 1e-300, e não como zero. Dada a dependência remanescente, essa estatística é apenas descritiva, e nenhuma conclusão deste notebook se apoia nela.
""",
            en="""
The correlations change little: at lag 1, from 0.196 to 0.194, and at lags 2 to 4 by at most 0.008. The residual persistence within the pair is not a contemporaneous shock common to a category, so the category × week term does not represent it. In the tests of section 3, this persistence is accommodated by the pair clustering, which allows arbitrary correlation between the weeks of the same pair.

It remains to record the likelihood-ratio statistic between the two REML fits, which have the same fixed part. Because a single variance is added, the p-value comes from the 50:50 mixture of χ² with 0 and with 1 degree of freedom (Self & Liang, 1987). The p-value is not representable in floating point and is displayed as less than 1e-300, not as zero. Given the remaining dependence, this statistic is only descriptive, and no conclusion of this notebook rests on it.
""",
        ),
        code("""
lr = catweek.load_json("key_numbers")["lr_descriptive"]
display(
    localize_columns(
        pd.DataFrame([{"stat": lr["stat"], "df": lr["df"], "p_value": format_p(lr["p_value"])}]),
        LANG,
    )
)
"""),
        md(
            pt="""
## 6. O que a evidência permite concluir

Nos contrastes, os três testes globais rejeitam a igualdade das inclinações do desconto a 5%, depois da correção de Holm sobre a família fixa de três: por segmento (valor-p ajustado de 0,0031), por tipo de marca (5,51e-07) e por categoria (7,83e-16). Sob a aproximação descrita, isso sustenta que as inclinações não são todas iguais dentro de cada moderador. Não sustenta, por si só, qual nível difere de qual, nem uma explicação para a diferença: as inclinações por nível são descritivas, com intervalos pontuais, e a heterogeneidade descrita é condicional e associativa. A conclusão do segmento é a mais fraca das três e indica heterogeneidade no conjunto dos segmentos. A diferença de UPSCALE, a única com intervalo pontual que exclui zero, é descritiva e não identifica, por si só, o segmento que difere. A inferência é aproximada, porque a dependência entre pares diferentes em semanas diferentes, como choques comuns que persistem de uma semana para outra, não é coberta pelo agrupamento em duas vias.

Na sensibilidade, a introdução de um intercepto aleatório de categoria × semana, que absorve os choques comuns contemporâneos, altera pouco as associações do F3: o desconto e o preço regular dentro do par ficam um pouco mais negativos (de −1,685 para −1,727 e de −1,592 para −1,660), as promoções mudam menos de 0,007, e só a tendência perde precisão. As associações do F3 não se mostram sensíveis, em sinal e em ordem de grandeza, à representação desses choques. Esse resultado não é um teste da especificação oficial, que o C0 e o ranking preditivo do P1 preservam, e a persistência residual dentro do par continua sem tratamento por esse termo.

Os contrastes por nível mostram, enfim, que a inclinação comum do F3 (−1,685) resume associações que, no OLS intrapar, variam de −1,083 a −2,416 entre categorias. Essa informação acompanha a leitura dos cenários do notebook 06, que usa a inclinação comum.
""",
            en="""
## 6. What the evidence allows one to conclude

In the contrasts, the three global tests reject the equality of the discount slopes at 5%, after the Holm correction over the fixed family of three: by segment (adjusted p-value of 0.0031), by brand type (5.51e-07) and by category (7.83e-16). Under the approximation described, this supports that the slopes are not all equal within each moderator. It does not support, by itself, which level differs from which, nor an explanation for the difference: the slopes by level are descriptive, with pointwise intervals, and the heterogeneity described is conditional and associative. The segment conclusion is the weakest of the three and indicates heterogeneity across the set of segments. The UPSCALE difference, the only one with a pointwise interval that excludes zero, is descriptive and does not, by itself, identify the segment that differs. The inference is approximate, because dependence between different pairs in different weeks, such as common shocks that persist from one week to the next, is not covered by the two-way clustering.

In the sensitivity analysis, the introduction of a random category × week intercept, which absorbs the contemporaneous common shocks, changes the associations of the F3 little: the discount and the regular price within the pair become slightly more negative (from −1.685 to −1.727 and from −1.592 to −1.660), the promotions change by less than 0.007, and only the trend loses precision. The associations of the F3 do not appear sensitive, in sign or in order of magnitude, to the representation of these shocks. This result is not a test of the official specification, which C0 and the predictive ranking of P1 preserve, and the residual persistence within the pair remains untreated by this term.

The contrasts by level finally show that the common slope of the F3 (−1.685) summarises associations that, in the within-pair OLS, range from −1.083 to −2.416 across categories. This information accompanies the reading of the scenarios of notebook 06, which uses the common slope.
""",
        ),
    ]


SETUP_06 = """
import warnings

import pandas as pd
from IPython.display import display

from retail_multilevel import interpret, viz
from retail_multilevel.i18n import localize_columns, scenario_label, t
from retail_multilevel.runs import load_run

warnings.filterwarnings("ignore", category=FutureWarning)
viz.use_retina()
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:,.3f}".format)
scen = load_run("p2_scenarios", "20261124-220614-acbfe8b9")
fixed = load_run("fixed_effects", "20261124-114202-1fda5a5d")
print(t("data_credit", LANG))
stored = {"p2_scenarios": scen, "fixed_effects": fixed}
display(pd.DataFrame({k: {"run_id": r.run_id, "finished_at": r.metadata["finished_at"]} for k, r in stored.items()}).T)


def localized(df):
    out = df.copy()
    if "scenario" in out:
        out["scenario"] = [scenario_label(x, LANG) for x in out["scenario"]]
    if "term" in out:
        out["term"] = [viz.term_label(x, LANG) for x in out["term"]]
    return localize_columns(out, LANG)
"""


def scenario_cells() -> list[Cell]:
    """Cells of notebook 06, the price and promotion scenarios read from the stored runs."""
    return [
        md(
            pt="""
# 06 · Cenários de preço e promoção

Este notebook traduz as estimativas do F3 do notebook 03 em cenários comparáveis de preço e promoção, cada um acompanhado do apoio empírico que encontra nos dados de treino. A leitura executiva compara uma TPR com 20% de desconto com um display com 20% de desconto, e os demais cenários são apresentados com a mesma profundidade.

Este notebook **não ajusta modelos**: o F3 foi reajustado por REML pelo script `scripts/fit_scenarios.py`, que gravou em `outputs/runs` os coeficientes, a covariância conjunta, a tabela de cenários e a conciliação com o notebook 03. O notebook apenas lê essa execução e a do F3 original, o que torna as versões em português e em inglês idênticas em resultados. Somente as semanas de treino (0 a 103) entram nos cálculos. As estimativas são associações condicionais em semanas com vendas registradas, e não efeitos causais: o calendário promocional, as rupturas de estoque e outros fatores podem influenciar as estimativas. Margem, custo de promoção, retorno sobre o investimento e recomendação causal de preço estão fora do escopo.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 06 · Pricing and promotion scenarios

This notebook translates the estimates of the F3 of notebook 03 into comparable price and promotion scenarios, each accompanied by the empirical support it finds in the training data. The executive reading compares a TPR with a 20% discount with a display with a 20% discount, and the other scenarios are presented with the same depth.

This notebook **fits no models**: the F3 was refitted by REML by the script `scripts/fit_scenarios.py`, which stored in `outputs/runs` the coefficients, the joint covariance, the scenario table and the reconciliation with notebook 03. The notebook only reads that run and the one of the original F3, so the Portuguese and English versions show identical results. Only the training weeks (0 to 103) enter the computations. The estimates are conditional associations in weeks with recorded sales, not causal effects: the promotional calendar, stockouts and other factors may influence the estimates. Margin, promotion cost, return on investment and causal price recommendations are out of scope.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_06),
        md(
            pt="""
As duas execuções lidas estão identificadas acima (identificador e horário de término), e os identificadores são fixados no código, de modo que reexecutar o notebook lê sempre os mesmos artefatos.

## 1. A pergunta e a definição dos cenários

A pergunta é quanto as vendas modeladas diferem, em cada combinação de desconto e promoção, de uma semana regular, definida como uma semana sem desconto e sem promoção, e quanto apoio cada combinação tem nos dados. Para um cenário com desconto d e indicadores de encarte f, de display s e de TPR r, a diferença no log das unidades é Δlog = β_desconto · log(1 − d) + β_encarte · f + β_display · s + β_TPR · r, com o preço-base e os demais controles mantidos constantes. O efeito é `100 × expm1(Δlog)`, uma razão modelada de vendas na escala retransformada, expressa como variação percentual. A modalidade TPR corresponde ao indicador `TPR_ONLY`, isto é, uma redução temporária de preço sem encarte nem display, e os indicadores de cada cenário respeitam essa exclusividade.

Os oito cenários são os registrados no notebook 03. Eles foram fixados antes dos resultados do notebook 05 e mantidos sem alteração depois deles: nenhum foi acrescentado, removido ou modificado em função dos contrastes. A tabela mostra o desenho de cada cenário.
""",
            en="""
The two runs that were read are identified above (identifier and finish time), and the identifiers are fixed in the code, so re-running the notebook always reads the same artefacts.

## 1. The question and the definition of the scenarios

The question is how much the modelled sales differ, under each combination of discount and promotion, from a regular week, defined as a week with no discount and no promotion, and how much support each combination has in the data. For a scenario with discount d and indicators of feature f, display s and TPR r, the difference in the log of units is Δlog = β_discount · log(1 − d) + β_feature · f + β_display · s + β_TPR · r, with the base price and the other controls held constant. The effect is `100 × expm1(Δlog)`, a modelled sales ratio on the back-transformed scale, expressed as a percentage change. The TPR modality corresponds to the `TPR_ONLY` indicator, that is, a temporary price reduction without feature or display, and the indicators of each scenario respect this exclusivity.

The eight scenarios are those registered in notebook 03. They were fixed before the results of notebook 05 and kept unchanged after them: none was added, removed or modified as a function of the contrasts. The table shows the design of each scenario.
""",
        ),
        code("""
design = pd.DataFrame(interpret.DEFAULT_SCENARIOS, columns=["scenario", "discount", "feature", "display", "tpr"])
display(localized(design.assign(discount=design["discount"] * 100)))
"""),
        md(
            pt="""
Os intervalos de cada cenário usam a covariância conjunta dos coeficientes do F3 ajustado por REML com o `lme4` (Bates et al., 2015): o erro-padrão de Δlog é obtido da combinação linear dos coeficientes com essa covariância, e os limites, retransformados pela mesma fórmula do efeito. Eles não são construídos a partir dos limites marginais de cada coeficiente, que ignoram as covariâncias entre eles. São intervalos pontuais de 95%, baseados no modelo.

## 2. Conciliação com o notebook 03

Antes de ler os cenários, convém verificar se o reajuste que gerou a covariância conjunta reproduz o F3 do notebook 03. A tabela compara os 14 coeficientes gravados naquele notebook com os do reajuste.
""",
            en="""
The intervals of each scenario use the joint covariance of the coefficients of the F3 fitted by REML with `lme4` (Bates et al., 2015): the standard error of Δlog is obtained from the linear combination of the coefficients with that covariance, and the limits are back-transformed by the same formula as the effect. They are not built from the marginal limits of each coefficient, which ignore the covariances among them. They are pointwise 95% intervals, based on the model.

## 2. Reconciliation with notebook 03

Before reading the scenarios, it is worth checking that the refit that produced the joint covariance reproduces the F3 of notebook 03. The table compares the 14 coefficients stored in that notebook with those of the refit.
""",
        ),
        code("""
recon = scen.load_table("reconciliation")
with pd.option_context("display.float_format", "{:,.6f}".format):
    display(localized(recon))
info = scen.load_json("reconciliation")
display(
    localize_columns(
        pd.DataFrame(
            [
                {
                    "nobs": info["nobs"],
                    "max_week": info["max_week"],
                    "messages": info["n_messages"],
                    "coef_max_abs_diff": info["coef_max_abs_diff"],
                    "effect_diff_pp": info["scenario_effect_max_abs_diff_pp"],
                }
            ]
        ),
        LANG,
    )
)
"""),
        md(
            pt="""
O reajuste usa as mesmas 347.092 linhas das semanas 0 a 103, converge sem mensagens e reproduz os 14 coeficientes gravados, com diferença absoluta máxima de 0,0 (a tolerância do script era de 1e-6). Os efeitos pontuais dos cenários também coincidem com os calculados, pela função de cenários do notebook 03, a partir dos coeficientes gravados, com diferença máxima de 0,0 ponto percentual. Os valores e o significado das estimativas são, portanto, os do notebook 03: a mesma combinação linear de coeficientes, na mesma escala. O que se acrescenta é a covariância conjunta, que permite os intervalos que o notebook 03 não podia apresentar.

## 3. Os cenários

A tabela a seguir traz, para cada cenário, o desconto, a variação modelada nas vendas em relação à semana regular e o intervalo pontual de 95%, e, em uma segunda tabela, a diferença em log e o seu erro-padrão. O gráfico mostra os mesmos efeitos ao lado do suporte observado de cada cenário nas linhas de treino.
""",
            en="""
The refit uses the same 347,092 rows of weeks 0 to 103, converges with no messages and reproduces the 14 stored coefficients, with a maximum absolute difference of 0.0 (the tolerance of the script was 1e-6). The point effects of the scenarios also coincide with those computed, by the scenario function of notebook 03, from the stored coefficients, with a maximum difference of 0.0 percentage points. The values and the meaning of the estimates are therefore those of notebook 03: the same linear combination of coefficients, on the same scale. What is added is the joint covariance, which allows the intervals that notebook 03 could not present.

## 3. The scenarios

The table below gives, for each scenario, the discount, the modelled change in sales relative to the regular week and the pointwise 95% interval, and, in a second table, the log difference and its standard error. The chart shows the same effects beside the observed support of each scenario in the training rows.
""",
        ),
        code("""
scenarios = scen.load_table("scenarios")
effects = scenarios.assign(discount=scenarios["discount"] * 100)[
    ["scenario", "discount", "effect_pct", "ci_low_pct", "ci_high_pct"]
]
display(localized(effects))
with pd.option_context("display.float_format", "{:,.4f}".format):
    display(localized(scenarios[["scenario", "dlog", "se"]]))
fig = viz.plot_scenario_effects(scenarios, LANG)
"""),
        md(
            pt="""
A semana regular tem contraste zero, como deve ser. Uma TPR com 10% de desconto está associada a uma variação modelada de +14,6% nas vendas (intervalo de 13,9% a 15,3%), e com 20% de desconto, de +39,8% (de 39,0% a 40,5%). O encarte sem desconto tem +64,4% (de 63,0% a 65,8%), e com 20% de desconto, +139,5% (de 137,8% a 141,2%). O display sem desconto tem +50,5% (de 49,4% a 51,5%), e com 20% de desconto, +119,2% (de 117,7% a 120,7%). O encarte com display e 25% de desconto tem +301,8% (de 298,7% a 304,9%).

Esses valores dependem da forma funcional: a elasticidade é constante em log(1 − d), o desconto e as modalidades são aditivos na escala logarítmica, e não há interação estimada entre encarte e display. O valor do cenário combinado, por exemplo, é o produto dos efeitos estimados em separado. Os intervalos são estreitos porque refletem apenas a variabilidade condicional ao modelo. O notebook 03 mostrou que os erros-padrão agrupados por par e por semana da sensibilidade intrapar ficam de 4,1 a 7,4 vezes os do F3 (Cameron et al., 2011), e portanto os intervalos acima não representam a incerteza completa. Eles também não incorporam o viés possível da seleção de quando e onde cada promoção é aplicada (Villas-Boas & Winer, 1999), que é uma questão de identificação, e não de variabilidade amostral.

## 4. A leitura executiva: TPR com 20% contra display com 20%

A comparação executiva coloca lado a lado os dois cenários de 20% de desconto, que diferem apenas pela modalidade. A razão modelada de vendas do display em relação à TPR é calculada pela diferença dos log-efeitos, com o erro-padrão da combinação linear obtido da covariância conjunta, e retransformada pela exponencial.
""",
            en="""
The regular week has zero contrast, as it should. A TPR with a 10% discount is associated with a modelled change of +14.6% in sales (interval from 13.9% to 15.3%), and with a 20% discount, of +39.8% (from 39.0% to 40.5%). Feature without discount has +64.4% (from 63.0% to 65.8%), and with a 20% discount, +139.5% (from 137.8% to 141.2%). Display without discount has +50.5% (from 49.4% to 51.5%), and with a 20% discount, +119.2% (from 117.7% to 120.7%). Feature with display and a 25% discount has +301.8% (from 298.7% to 304.9%).

These values depend on the functional form: the elasticity is constant in log(1 − d), the discount and the modalities are additive on the logarithmic scale, and no interaction between feature and display is estimated. The value of the combined scenario, for instance, is the product of the separately estimated effects. The intervals are narrow because they reflect only the variability conditional on the model. Notebook 03 showed that the standard errors clustered by pair and by week of the within-pair sensitivity analysis are 4.1 to 7.4 times those of the F3 (Cameron et al., 2011), so the intervals above do not represent the full uncertainty. They also do not incorporate the possible bias from the selection of when and where each promotion is applied (Villas-Boas & Winer, 1999), which is a matter of identification rather than of sampling variability.

## 4. The executive reading: TPR at 20% against display at 20%

The executive comparison places side by side the two scenarios with a 20% discount, which differ only in the modality. The modelled sales ratio of the display relative to the TPR is computed from the difference of the log effects, with the standard error of the linear combination obtained from the joint covariance, and back-transformed by the exponential.
""",
        ),
        code("""
executive = scen.load_json("executive")
keys = ["dlog", "se", "effect_pct", "ci_low_pct", "ci_high_pct"]
chosen = pd.DataFrame(
    [{"scenario": executive[s], **{k: executive[e][k] for k in keys}} for s, e in (("a", "effect_a"), ("b", "effect_b"))]
)
ratio = executive["difference_b_minus_a"]
with pd.option_context("display.float_format", "{:,.4f}".format):
    display(localized(chosen))
    display(
        localize_columns(
            pd.DataFrame(
                [
                    {
                        "dlog": ratio["dlog"],
                        "se": ratio["se"],
                        "ratio": ratio["ratio"],
                        "ratio_low": ratio["ratio_low"],
                        "ratio_high": ratio["ratio_high"],
                    }
                ]
            ),
            LANG,
        )
    )
"""),
        md(
            pt="""
A TPR com 20% de desconto está associada a +39,8% nas vendas modeladas (de 39,0% a 40,5%) e o display com 20%, a +119,2% (de 117,7% a 120,7%). A razão modelada de vendas do display frente à TPR é de 1,568, com intervalo pontual de 1,555 a 1,581, ou seja, vendas modeladas cerca de 57% maiores no cenário de display. Os dois cenários têm suporte nos dados, como mostra a seção seguinte, mas a diferença continua sendo uma associação condicional, sujeita à seleção de quando e onde cada modalidade é aplicada, e não um efeito causal da escolha entre elas.

## 5. O suporte observado

A leitura dos cenários só faz sentido se houver linhas observadas que os realizem. O suporte é contado nas semanas 0 a 103, para a combinação exata de modalidades e com desconto dentro de ± 3 pontos percentuais do valor do cenário. A tabela mostra as linhas, os pares distintos e a participação nas linhas de treino, o mesmo suporte restrito às linhas com preço-base observado (sem `BASE_IMPUTED`), o número de linhas de suporte que vieram do preço-base de fallback (`BASE_PRICE = PRICE`, que zera o desconto da linha) e a indicação de cenário fora do suporte observado.
""",
            en="""
The TPR with a 20% discount is associated with +39.8% in modelled sales (from 39.0% to 40.5%) and the display with 20%, with +119.2% (from 117.7% to 120.7%). The modelled sales ratio of the display relative to the TPR is 1.568, with a pointwise interval from 1.555 to 1.581, that is, modelled sales about 57% higher in the display scenario. Both scenarios have support in the data, as the next section shows, but the difference remains a conditional association, subject to the selection of when and where each modality is applied, not a causal effect of choosing between them.

## 5. The observed support

The reading of the scenarios only makes sense if observed rows realise them. The support is counted in weeks 0 to 103, for the exact combination of modalities and with a discount within ± 3 percentage points of the scenario value. The table shows the rows, the distinct pairs and the share of the training rows, the same support restricted to rows with an observed base price (without `BASE_IMPUTED`), the number of support rows that came from the fallback base price (`BASE_PRICE = PRICE`, which zeroes the discount of the row) and the flag for a scenario outside the observed support.
""",
        ),
        code("""
support = scenarios.assign(
    share=scenarios["share"] * 100, share_observed_base=scenarios["share_observed_base"] * 100
).rename(columns={"share": "share_pct"})[
    [
        "scenario", "n_rows", "n_pairs", "share_pct", "n_rows_observed_base", "n_pairs_observed_base",
        "share_observed_base", "n_rows_fallback", "outside_support",
    ]
]  # fmt: skip
with pd.option_context("display.float_format", "{:,.2f}".format):
    display(localized(support))
"""),
        md(
            pt="""
Nenhum cenário está fora do suporte observado. A semana regular tem 247.331 linhas (71,26% do treino, 3.858 pares), e os cenários promocionais têm de 0,30% a 2,93% das linhas: TPR com 10% de desconto tem 9.393 linhas (2,71%, 2.005 pares), TPR com 20% tem 10.185 (2,93%, 2.261 pares), encarte sem desconto tem 1.057 (0,30%, 641 pares), encarte com 20% tem 2.574 (0,74%, 1.212 pares), display sem desconto tem 8.624 (2,48%, 1.648 pares), display com 20% tem 2.273 (0,65%, 883 pares) e encarte com display e 25% tem 2.770 (0,80%, 968 pares). O suporte dos cenários de interesse executivo, TPR e display com 20%, é de 10.185 e 2.273 linhas, respectivamente. Restringir o suporte às linhas com preço-base observado muda pouco as contagens.

As linhas com preço-base de fallback, que têm desconto zero por construção, aparecem apenas nos cenários sem desconto: 18 linhas no encarte sem desconto (de 1.057) e 69 no display sem desconto (de 8.624). Em relação à tabela do notebook 03, quatro contagens diferem em 1 a 3 linhas (encarte sem desconto, 1.057 contra 1.055; display sem desconto, 8.624 contra 8.622; display com 20%, 2.273 contra 2.272; encarte com display e 25%, 2.770 contra 2.767), porque a borda da janela de ± 3 pontos percentuais passou a ser contada como incluída, com tolerância numérica.

O tratamento de um cenário sem suporte está implementado no código e coberto por testes sintéticos: a linha permanece na tabela, é marcada como fora do suporte observado e não é apresentada como aplicação de negócio. Esse caso não ocorre aqui, e a coluna correspondente continua visível. Não se mostra um recorte do suporte por grupo, porque o F3 usa inclinações comuns a todos os grupos: um recorte por segmento ou por categoria descreveria apenas onde estão os dados, e não efeitos específicos de grupo. O notebook 05 mostrou que as inclinações do desconto variam por nível, de −1,083 a −2,416 entre categorias, de modo que os valores acima descrevem uma associação média, e não a de uma categoria ou de um segmento em particular.

## 6. Limites e escopo

Os cenários são associações condicionais, calculadas em semanas com vendas registradas, e a retransformação para a escala original dá uma razão modelada de vendas, e não uma previsão de vendas em um futuro controlado. Cada estimativa depende da forma funcional aditiva em log, do calendário promocional observado, das rupturas e de outros fatores que o modelo não representa, e a seleção de quando e onde cada promoção foi aplicada pode contaminar a associação, em especial a de preço (Villas-Boas & Winer, 1999). Os intervalos são pontuais, baseados no modelo e com cobertura nominal de 95%. Como os erros-padrão agrupados da sensibilidade intrapar ficam muito acima dos do modelo, a incerteza total é maior que a mostrada. O efeito promocional estimado é o de um incremento nas vendas (Blattberg & Neslin, 1989), sem custo, margem ou canibalização, de modo que nenhum ranking de rentabilidade decorre destes números. Margem, custo de promoção, retorno sobre o investimento e recomendação causal de preço permanecem fora do escopo, e a camada de decisão do projeto é uma leitura comparável de cenários com o apoio empírico de cada um.
""",
            en="""
No scenario is outside the observed support. The regular week has 247,331 rows (71.26% of training, 3,858 pairs), and the promotional scenarios have from 0.30% to 2.93% of the rows: TPR with a 10% discount has 9,393 rows (2.71%, 2,005 pairs), TPR with 20% has 10,185 (2.93%, 2,261 pairs), feature without discount has 1,057 (0.30%, 641 pairs), feature with 20% has 2,574 (0.74%, 1,212 pairs), display without discount has 8,624 (2.48%, 1,648 pairs), display with 20% has 2,273 (0.65%, 883 pairs) and feature with display at 25% has 2,770 (0.80%, 968 pairs). The support of the scenarios of executive interest, TPR and display with 20%, is 10,185 and 2,273 rows, respectively. Restricting the support to rows with an observed base price changes the counts little.

The rows with a fallback base price, which have zero discount by construction, appear only in the scenarios without a discount: 18 rows in feature without discount (of 1,057) and 69 in display without discount (of 8,624). Relative to the table of notebook 03, four counts differ by 1 to 3 rows (feature without discount, 1,057 against 1,055; display without discount, 8,624 against 8,622; display with 20%, 2,273 against 2,272; feature with display at 25%, 2,770 against 2,767), because the edge of the ± 3 percentage-point window is now counted as included, with numerical tolerance.

The handling of a scenario without support is implemented in the code and covered by synthetic tests: the row stays in the table, is flagged as outside the observed support and is not presented as a business application. That case does not occur here, and the corresponding column stays visible. No cut of the support by group is shown, because the F3 uses slopes common to all groups: a cut by segment or by category would describe only where the data are, not group-specific effects. Notebook 05 showed that the discount slopes vary by level, from −1.083 to −2.416 across categories, so the values above describe an average association, not that of a particular category or segment.

## 6. Limits and scope

The scenarios are conditional associations, computed in weeks with recorded sales, and the back-transformation to the original scale gives a modelled sales ratio, not a forecast of sales in a controlled future. Each estimate depends on the additive log functional form, on the observed promotional calendar, on stockouts and on other factors that the model does not represent, and the selection of when and where each promotion was applied may contaminate the association, in particular the price one (Villas-Boas & Winer, 1999). The intervals are pointwise, model-based and with nominal 95% coverage. Because the clustered standard errors of the within-pair sensitivity analysis are far above the model ones, the total uncertainty is larger than the one shown. The estimated promotional effect is that of an increment in sales (Blattberg & Neslin, 1989), with no cost, margin or cannibalisation, so no profitability ranking follows from these numbers. Margin, promotion cost, return on investment and causal price recommendations remain out of scope, and the decision layer of the project is a comparable reading of scenarios with the empirical support of each.
""",
        ),
        code("""
slope_test = load_run("discount_slope_lr", "20261125-112223-aa33d3e6")
varcomp = slope_test.load_table("varcomp")
dispersion = varcomp.loc[(varcomp["model"] == "C1") & (varcomp["term"] == "LOG_DISC"), ["group", "sd"]].copy()
dispersion["group"] = dispersion["group"].map({"UPC": t("group_product", LANG), "STORE_NUM": t("group_store", LANG)})
dispersion["common_slope"] = fixed.load_table("fixef").set_index("term").loc["LOG_DISC", "estimate"]
dispersion["sd_over_common"] = dispersion["sd"] / dispersion["common_slope"].abs()
display(localize_columns(dispersion.reset_index(drop=True), LANG))
"""),
        md(
            pt="""
### Dispersão da inclinação do desconto entre produtos

Os cenários usam a inclinação comum do desconto do F3 (−1,685), e os intervalos estreitos refletem a precisão dessa inclinação comum. O teste complementar do notebook 04 (execução `discount_slope_lr`) ajustou o candidato C1, que acrescenta ao C0 inclinações aleatórias do desconto por produto e por loja (Gelman & Hill, 2007), e estimou um desvio-padrão da inclinação de 0,884 entre produtos e de 0,274 entre lojas. Em relação à inclinação comum, esses desvios equivalem a cerca de 52% e 16% do seu valor absoluto. Sob o C1, portanto, a inclinação do desconto de um produto individual se afasta da inclinação comum com desvio-padrão em torno de 0,88 (lojas, 0,27), o que é grande diante de −1,69. Os efeitos dos cenários descrevem, assim, uma associação média, e a resposta de um produto específico pode diferir substancialmente dela. Os intervalos estreitos referem-se à inclinação comum, não à de um produto em particular. A execução não grava a estimativa fixa de `LOG_DISC` do C1, de modo que nenhum efeito de cenário por produto é calculado sob esse modelo. O C0 permanece o modelo selecionado pelo protocolo P1.
""",
            en="""
### Dispersion of the discount slope across products

The scenarios use the common discount slope of the F3 (−1.685), and the narrow intervals reflect the precision of that common slope. The complementary test of notebook 04 (run `discount_slope_lr`) fitted the candidate C1, which adds to the C0 random discount slopes by product and by store (Gelman & Hill, 2007), and estimated a standard deviation of the slope of 0.884 across products and 0.274 across stores. Relative to the common slope, these deviations amount to about 52% and 16% of its absolute value. Under the C1, therefore, the discount slope of an individual product is estimated to deviate from the common slope with a standard deviation of about 0.88 (stores, 0.27), which is large against −1.69. The scenario effects thus describe an average association, and the response of a specific product can differ substantially from it. The narrow intervals refer to the common slope, not to that of a particular product. The run does not store the fixed `LOG_DISC` estimate of the C1, so no per-product scenario effect is computed under that model. The C0 remains the model selected by the P1 protocol.
""",
        ),
    ]


NOTEBOOKS: list[NotebookSpec] = [
    NotebookSpec("01", "01_eda_painel.ipynb", "01_panel_eda.ipynb", with_references(eda_cells())),
    NotebookSpec("02", "02_modelo_nulo.ipynb", "02_null_model.ipynb", with_references(null_model_cells())),
    NotebookSpec("03", "03_efeitos_fixos.ipynb", "03_fixed_effects.ipynb", with_references(fixed_effects_cells())),
    NotebookSpec("04", "04_protocolo_p1.ipynb", "04_protocol_p1.ipynb", with_references(protocol_p1_cells())),
    NotebookSpec(
        "05",
        "05_contrastes_e_sensibilidade.ipynb",
        "05_contrasts_and_sensitivity.ipynb",
        with_references(contrasts_cells()),
    ),
    NotebookSpec(
        "06",
        "06_cenarios_preco_promocao.ipynb",
        "06_pricing_promotion_scenarios.ipynb",
        with_references(scenario_cells()),
    ),
]


def render(spec: NotebookSpec, lang: str) -> nbformat.NotebookNode:
    """Render ``spec`` in ``lang`` ('pt' or 'en') as an nbformat notebook."""
    cells = []
    for cell in spec.cells:
        if cell.kind == "md":
            cells.append(new_markdown_cell(cell.pt if lang == "pt" else cell.en))
        elif cell.kind == "lang":
            cells.append(new_code_cell(f'LANG = "{lang}"'))
        else:
            cells.append(new_code_cell(cell.source))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    return nb


def execute(path: Path, cwd: Path = ROOT, timeout: int = CELL_TIMEOUT) -> None:
    """Execute the notebook at ``path`` in place, with ``cwd`` as working directory."""
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute()
    nbformat.write(nb, path)


def build(
    out_dir: Path = NOTEBOOKS_DIR, only: str | None = None, run: bool = False, specs: list[NotebookSpec] | None = None
) -> list[Path]:
    """Write the PT and EN notebooks under ``out_dir`` (optionally executing them); return the paths."""
    written = []
    for spec in NOTEBOOKS if specs is None else specs:
        if only is not None and spec.number != only:
            continue
        for lang, name in (("pt", spec.pt_name), ("en", spec.en_name)):
            path = out_dir / LANG_DIRS[lang] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(render(spec, lang), path)
            if run:
                print(f"executing {path}", flush=True)
                execute(path)
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="notebook number to build, e.g. 01")
    parser.add_argument("--execute", action="store_true", help="execute the notebooks in place after writing")
    args = parser.parse_args(argv)
    paths = build(only=args.only, run=args.execute)
    if not paths:
        print(f"no notebook matches --only {args.only!r}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
