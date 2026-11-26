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

*Dados: dunnhumby Source Files – Breakfast at the Frat.* A análise descreve o painel sem ajuste de modelos. As funções de `src/retail_multilevel` geram as tabelas e os gráficos agregados. Os dados brutos são licenciados e ficam fora do repositório.
""",
            en="""
# 01 · Exploratory analysis of the panel

*Data: dunnhumby Source Files – Breakfast at the Frat.* The analysis describes the panel without fitting models. The functions in `src/retail_multilevel` generate aggregate tables and charts. The raw data are licensed and kept outside the repository.
""",
        ),
        lang_cell(),
        code(SETUP),
        md(
            pt="""
## Briefing: como funciona o negócio

As definições vêm do guia do usuário e da aba *Glossary* do arquivo da dunnhumby. As contagens foram calculadas a partir do painel.

O painel reúne vendas de 77 lojas de uma rede varejista em quatro estados dos EUA. O guia omite o nome da rede e registra 79 linhas na tabela de lojas. A rede classificou cada loja como *Value*, *Mainstream* ou *Upscale*. O guia descreve esses segmentos como patamares de preço e deixa em aberto como foram definidos. Nas lojas 4503 e 17627, o cadastro traz dois segmentos distintos, tratados como ambíguos.
""",
            en="""
## Briefing: how the business works

Definitions come from dunnhumby's user guide and the *Glossary* sheet of its file. Counts were calculated from the panel.

The panel records sales from 77 stores of a retail chain in four US states. The guide leaves the chain unnamed and lists 79 rows in the store table. The chain assigned each store to *Value*, *Mainstream* or *Upscale*. The guide describes these segments as pricing levels and leaves their definition open. The store table assigns two different segments to stores 4503 and 17627, so both assignments are treated as ambiguous.
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
São acompanhadas quatro categorias, que incluem enxaguantes bucais (`ORAL HYGIENE PRODUCTS`), pretzels (`BAG SNACKS`), pizza congelada (`FROZEN PIZZA`) e cereal em caixa (`COLD CEREAL`). Dos 55 produtos (UPCs), 43 são de marcas nacionais e 12 de marca própria da rede (`PRIVATE LABEL` no campo de fabricante), com três de marca própria em cada categoria. Há 15 outros fabricantes.
""",
            en="""
Four categories are tracked, covering mouthwash (`ORAL HYGIENE PRODUCTS`), pretzels (`BAG SNACKS`), frozen pizza (`FROZEN PIZZA`) and boxed cereal (`COLD CEREAL`). Of the 55 products (UPCs), 43 are national-brand products and 12 are the retailer's private-label products (`PRIVATE LABEL` in the manufacturer field), with three private-label products in each category. There are 15 other manufacturers.
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
Cada linha do painel corresponde a um produto (UPC) em uma loja em uma semana. O painel cobre 156 semanas, de 2009-01-14 a 2012-01-04 pelas datas de encerramento.

`UNITS` conta as unidades vendidas e `SPEND` registra o faturamento em dólares. `VISITS` conta as cestas que incluíram ao menos uma unidade do produto, e `HHS` conta os domicílios que o compraram. O desconto é calculado comparando `PRICE`, o preço pago na loja, com `BASE_PRICE`, o preço de referência do item.
""",
            en="""
Each row of the panel represents one product (UPC) in one store in one week. The panel covers 156 weeks, from 2009-01-14 to 2012-01-04 by week-ending date.

`UNITS` counts units sold and `SPEND` records revenue in dollars. `VISITS` counts baskets containing at least one unit of the product, and `HHS` counts the households that bought it. The discount is calculated by comparing `PRICE`, the price paid in the store, with `BASE_PRICE`, the item's reference price.
""",
        ),
        md(
            pt="""
### Promoções semanais

A **TPR** (*temporary price reduction*, `TPR_ONLY`) é um corte temporário de preço, comunicado pelo valor na prateleira. O **encarte** (*feature*, `FEATURE`) indica que o produto apareceu no folheto de ofertas. O **display** (`DISPLAY`) indica que o produto teve um ponto de exposição especial na loja.

A seção 4 confirma que `TPR_ONLY` só aparece em linhas sem encarte nem display. Encarte e display podem ocorrer juntos.
""",
            en="""
### Weekly promotions

A **TPR** (temporary price reduction, `TPR_ONLY`) is a temporary price cut communicated through the price on the shelf. A **feature** (`FEATURE`) records that the product appeared in the store's weekly flyer. A **display** (`DISPLAY`) records that the product was given a special display location in the store.

Section 4 confirms that `TPR_ONLY` occurs only in rows with neither feature nor display. Feature and display can occur together.
""",
        ),
        md(
            pt="""
### Preço, promoção e margem

Um preço menor ou uma promoção pode aumentar as unidades vendidas, com redução da margem unitária e, em alguns casos, custos de promoção. A **elasticidade-preço** mede a variação percentual das unidades associada a uma variação de 1% no preço (Tellis, 1988; Bijmolt et al., 2005). Com elasticidade de −2, um corte de 10% no preço corresponde a cerca de 20% mais unidades pela aproximação −2 × −10%. No modelo logarítmico de elasticidade constante, o cálculo exato é 0,9<sup>−2</sup> − 1 ≈ 23,5%. O **lift de promoção** mede o aumento das unidades em semanas com promoção em relação a semanas sem ela (Blattberg & Neslin, 1989).

A resposta ao preço e às promoções pode variar entre lojas e produtos. Muitos pares loja × produto têm poucas semanas ou pouca variação de preço. Quando cada par é ajustado separadamente, as elasticidades podem ficar muito dispersas e até trocar de sinal, como mostra a seção 8. Uma elasticidade única, por sua vez, deixa de representar diferenças entre pares. Com *partial pooling* (agrupamento parcial), os **modelos multinível** combinam a informação de vários grupos. As estimativas dos grupos com menos informação ficam mais próximas da média (Gelman & Hill, 2007; Raudenbush & Bryk, 2002).
""",
            en="""
### Price, promotions and margins

A lower price or a promotion can increase units sold, with a lower unit margin and, in some cases, promotion costs. **Price elasticity** measures the percentage change in units associated with a 1% change in price (Tellis, 1988; Bijmolt et al., 2005). With an elasticity of −2, a 10% price cut corresponds to about 20% more units using the approximation −2 × −10%. In a log model with constant elasticity, the exact calculation is 0.9<sup>−2</sup> − 1 ≈ 23.5%. **Promotion lift** measures the increase in units in weeks with a promotion relative to weeks without it (Blattberg & Neslin, 1989).

The response to price and promotions can vary across stores and products. Many store × product pairs have few weeks or little price variation. When each pair is fitted separately, the elasticities can vary widely and even change sign, as section 8 shows. A single elasticity, in turn, misses differences across pairs. With partial pooling, **multilevel models** combine information from several groups. Estimates for groups with less information are pulled closer to the mean (Gelman & Hill, 2007; Raudenbush & Bryk, 2002).
""",
        ),
        md(
            pt="""
## A pergunta-chave

**Como as vendas se associam ao preço e às promoções em cada loja e produto?** O guia pergunta para quais produtos vale reduzir o preço para vender mais ou aumentá-lo para melhorar o lucro. Essas perguntas orientam a análise de cenários do projeto.

A análise começa pela variação das vendas entre lojas, produtos e pares loja × produto, e entre semanas do mesmo par. Em seguida são estimadas a elasticidade-preço média e as associações com encarte, display e TPR. Segmento da loja, categoria e marca própria entram na análise das diferenças de elasticidade.

A previsão fora da amostra compara o multinível com uma regressão de nível único (OLS) e com o GPBoost (Sigrist, 2022). Nos cenários, os coeficientes representam associações condicionais em semanas com transações registradas. Identificação causal e dados de margem seriam necessários para recomendar preços ou calcular o ROI incremental.

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

**How are sales associated with price and promotions in each store and product?** The guide asks which products should have their prices lowered to sell more or raised to improve profit. These questions guide the project's scenario analysis.

The analysis starts with variation in sales across stores, products and store × product pairs, and across weeks within the same pair. It then estimates the average price elasticity and the associations with feature, display and TPR. Store segment, category and private label enter the analysis of differences in elasticity.

Out-of-sample prediction compares the multilevel model with a single-level regression (OLS) and with GPBoost (Sigrist, 2022). In the scenarios, the coefficients represent conditional associations in weeks with recorded transactions. Pricing recommendations or an estimate of incremental ROI would require causal identification and margin data.

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

As primeiras 104 semanas foram usadas para ajustar os modelos e selecionar as especificações. As últimas 52 semanas, de 2011-01-12 a 2012-01-04, cobrem um ciclo sazonal completo e formam a **janela de avaliação temporal retrospectiva com exposição exploratória prévia**, chamada de "teste" nas tabelas e nos gráficos. O painel inteiro foi usado nos resumos exploratórios de unidades semanais, tamanho do painel e taxas de promoção, e no OLS exploratório e no perfil de variação de preço de `PROJECT_BRIEF.md`. A exploração motivou decisões de desenvolvimento como a decomposição de preço de D12. A avaliação fora da amostra (Tashman, 2000) é apresentada no notebook 04 e segue um protocolo registrado antes do ajuste dos candidatos (Nosek et al., 2018; Weston et al., 2019).
""",
            en="""
## 2. Temporal split (D11, revised)

The first 104 weeks were used to fit the models and select specifications. The final 52 weeks, from 2011-01-12 to 2012-01-04, cover a full seasonal cycle and form the **retrospective temporal evaluation window with prior exploratory exposure**, labelled "test" in tables and charts. The full panel was used for exploratory summaries of weekly units, panel size and promotion rates, as well as the exploratory OLS and price-variation profile in `PROJECT_BRIEF.md`. That exploration informed development decisions such as the D12 price decomposition. Out-of-sample evaluation (Tashman, 2000) is presented in notebook 04 and follows a protocol registered before the candidates were fitted (Nosek et al., 2018; Weston et al., 2019).
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
A amostra de treino tem 347.092 linhas em 3.859 pares, e a janela de avaliação tem 177.767 linhas em 3.870 pares. Cinquenta pares só aparecem na avaliação. Para esses pares, a parte aleatória da previsão usa os efeitos de loja e produto disponíveis, pois o par não tem histórico no treino.
""",
            en="""
The training sample has 347,092 rows in 3,859 pairs, and the evaluation window has 177,767 rows in 3,870 pairs. Fifty pairs appear only in the evaluation window. For these pairs, the random part of the prediction uses the available store and product effects because the pair has no training history.
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
A mediana é de 102 semanas de treino por par, e 1.537 pares têm as 104 semanas. Um quarto dos pares tem até 86 semanas, e alguns têm uma só. Com *partial pooling*, os pares de histórico curto são estimados também com a informação dos demais.
""",
            en="""
The median is 102 training weeks per pair, and 1,537 pairs have all 104 weeks. A quarter of the pairs have at most 86 weeks, and some have just one. With partial pooling, estimates for pairs with short histories also use information from other pairs.
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
Há alguma promoção em 28,5% das linhas, com encarte em 8,4%, display em 11,0% e TPR em 13,5%. A primeira tabela confirma que `TPR_ONLY` nunca coincide com encarte ou display (0 linhas). A segunda mostra 129.830 linhas com preço abaixo de 95% do base, todas com promoção. As outras 19.555 linhas com promoção ficaram no limiar de 95% ou acima dele. A frequência de encarte varia por categoria, de 2,5% em snacks a 16,7% em pizza congelada.
""",
            en="""
A promotion appears in 28.5% of rows, with feature in 8.4%, display in 11.0% and TPR in 13.5%. The first table confirms that `TPR_ONLY` never coincides with feature or display (0 rows). The second shows 129,830 rows with prices below 95% of base, all with a promotion. The other 19,555 promotion rows were at or above the 95% threshold. The frequency of feature varies by category, from 2.5% in bag snacks to 16.7% in frozen pizza.
""",
        ),
        md(
            pt="""
## 5. Variação de preço

O desvio-padrão de `LOG_PRICE` mede quanta variação de preço dentro de cada par está disponível no treino para estimar a elasticidade.
""",
            en="""
## 5. Price variation

The standard deviation of `LOG_PRICE` measures how much within-pair price variation is available in the training sample to estimate elasticity.
""",
        ),
        code("""
variation = eda.price_variation(train)
variation[["sd_log_price", "share_below_base"]].describe()
"""),
        md(
            pt="""
A mediana do desvio-padrão de log(preço) dentro do par é de 0,111 (cerca de 11%). Em média, 24% das semanas de cada par têm preço abaixo de 95% do base. Essa variação concentra-se em semanas de promoção, o que dificulta distinguir a associação com preço da associação com encarte e display.
""",
            en="""
The median within-pair standard deviation of log(price) is 0.111 (about 11%). On average, 24% of a pair's weeks have prices below 95% of base. This variation is concentrated in promotion weeks, which makes it harder to distinguish the association with price from the associations with feature and display.
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
As diferenças entre as médias de cada produto respondem por 53,3% da variância de log(unidades), as diferenças entre lojas por 6,7% e as diferenças entre pares por 72,7%. As 3.859 médias de par incorporam diferenças de loja e produto e a interação entre os dois. O modelo nulo com efeitos aleatórios cruzados (Baayen et al., 2008), no notebook 02, estima componentes separados de loja, produto, interação e resíduo.
""",
            en="""
Differences between product means account for 53.3% of log(units) variance, differences between stores for 6.7% and differences between pairs for 72.7%. The 3,859 pair means incorporate differences between stores and products as well as their interaction. The crossed random-effects null model (Baayen et al., 2008) in notebook 02 estimates separate components for store, product, interaction and residual.
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
No treino, o lift bruto foi de 195% para encarte, 129% para display e 33% para TPR. A comparação descritiva inclui as reduções de preço, a ocorrência conjunta de encarte e display e a possibilidade de concentrar essas promoções em produtos ou semanas de maior demanda.
""",
            en="""
In the training sample, raw lift was 195% for feature, 129% for display and 33% for TPR. This descriptive comparison includes price cuts, weeks with both feature and display, and the possibility that these promotions were concentrated in products or weeks of higher demand.
""",
        ),
        md(
            pt="""
## 8. Inclinações do desconto estimadas par a par

Foi estimado um OLS para cada par loja × produto com ao menos 8 semanas de treino (semanas 0 a 103). A especificação é `LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY`, com `LOG_DISC = log(preço / preço base)`. São omitidos termos constantes no par ou colineares com os anteriores. Se `LOG_DISC` for constante, o par fica sem inclinação estimada. Os resultados foram gravados por `scripts/fit_pair_ols_slopes.py`. Como `LOG_DISC` diminui quando o desconto aumenta, uma inclinação negativa associa descontos maiores a mais unidades. Uma inclinação positiva associa descontos maiores a menos unidades.
""",
            en="""
## 8. Discount slopes estimated pair by pair

One OLS was fitted for each store × product pair with at least 8 training weeks (weeks 0 to 103). The specification is `LOG_UNITS ~ LOG_DISC + FEATURE + DISPLAY + TPR_ONLY`, with `LOG_DISC = log(price / base price)`. Terms that are constant within the pair or collinear with preceding terms are omitted. If `LOG_DISC` is constant, no slope is estimated for the pair. The results were saved by `scripts/fit_pair_ols_slopes.py`. Since `LOG_DISC` decreases as the discount increases, a negative slope associates deeper discounts with more units. A positive slope associates deeper discounts with fewer units.
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

Uma inclinação positiva vai na direção oposta à relação de demanda esperada. Em um par isolado, ela pode refletir poucas semanas com desconto ou fatores como a época das promoções. Essa dispersão motiva o uso de *partial pooling* nos notebooks 02 a 04 (Gelman & Hill, 2007).
""",
            en="""
Of the 3,859 training pairs, 3,788 had the discount slope estimated: 42 pairs have fewer than 8 weeks and 29 were fitted without estimable discount variation. The median slope is −1.65, but the dispersion is large, with the 5th percentile at −5.40 and the 95th percentile at +3.18. A total of 751 pairs (19.8%) have a positive slope. The share is 36.2% among pairs with fewer than 26 weeks (25 of 69), 27.3% among those with 26 to 77 weeks (148 of 543) and 18.2% among those with 78 weeks or more (578 of 3,176). It decreases with history, but it still reaches almost one pair in five among the longest ones. Slopes that are exactly zero, which occur in pairs with constant units, and coefficients that are zero up to rounding are not counted as positive.

A positive slope runs counter to the expected demand relationship. For an individual pair, it may reflect few discount weeks or factors such as promotion timing. This spread motivates partial pooling in notebooks 02 to 04 (Gelman & Hill, 2007).
""",
        ),
        md(
            pt="""
## 9. Resumo

O painel tem 524.926 linhas, 77 lojas, 55 produtos, 3.909 pares e 156 semanas, e a amostra de modelagem exclui 67 \
linhas. As últimas 52 semanas formam a janela de avaliação retrospectiva, com exposição exploratória prévia, e 50 \
pares só aparecem nesse período. O painel de treino é desbalanceado (mediana de 102 semanas por par), o que \
favorece o *partial pooling*. Estimadas par a par, 19,8% das inclinações do desconto são positivas, o que reforça essa motivação.

Em 28,5% das linhas há alguma promoção. Todo preço abaixo de 95% do base foi observado em semanas com promoção, e `TPR_ONLY` ocorre sem encarte nem display. O lift bruto combina a associação promocional com a de preço. Na análise descritiva de log(unidades), produtos e pares respondem pelas maiores frações de variância entre grupos. Essas frações não são aditivas.

O notebook 02 lê os modelos nulos ajustados por script (`runs.load_run("null_models", "20261124-114105-b15be79c")`) e decompõe a variância \
entre loja, produto, interação e resíduo.
""",
            en="""
## 9. Summary

The panel has 524,926 rows, 77 stores, 55 products, 3,909 pairs and 156 weeks, and the modelling sample excludes \
67 rows. The last 52 weeks form the retrospective evaluation window, with prior exploratory exposure, and 50 pairs \
appear only in that period. The training panel is unbalanced (median of 102 weeks per pair), which favours partial \
pooling. Estimated pair by pair, 19.8% of the discount slopes are positive, which reinforces this motivation.

A promotion appears in 28.5% of rows. All prices below 95% of base were observed in promotion weeks, and `TPR_ONLY` occurs without feature or display. Raw lift combines the promotion and price associations. In the descriptive analysis of log(units), products and pairs account for the largest between-group variance shares. These shares are not additive.

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

Na seleção por etapas (*step-up*) de modelos multinível, o modelo nulo contém apenas o intercepto na parte fixa e reparte a variância da resposta entre os componentes do modelo (Raudenbush & Bryk, 2002). As observações semanais estão aninhadas nos pares loja × produto, e os pares pertencem a dois agrupamentos cruzados, loja e produto (Baayen et al., 2008). O ajuste foi feito com as 104 semanas iniciais de treino (D11 revisada).

As duas versões do notebook leem a mesma execução, gravada por `scripts/fit_null_models.py` em `outputs/runs/null_models`.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 02 · Null model: variance decomposition

In step-up selection for multilevel models, the null model has only an intercept in the fixed part and partitions the response variance into model components (Raudenbush & Bryk, 2002). Weekly observations are nested within store × product pairs, and the pairs belong to two crossed groupings, store and product (Baayen et al., 2008). The models were fitted on the first 104 weeks of training data (D11, revised).

Both language versions of the notebook read the same run, saved by `scripts/fit_null_models.py` in `outputs/runs/null_models`.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_02),
        md(
            pt="""
O identificador e o horário de término da execução são mostrados acima. O treino reúne 347.092 linhas, 77 lojas, 55 produtos e 3.859 pares, de 2009-01-14 a 2011-01-05.
""",
            en="""
The run identifier and finish time are shown above. The training sample has 347,092 rows, 77 stores, 55 products and 3,859 pairs, from 2009-01-14 to 2011-01-05.
""",
        ),
        md(
            pt="""
## 1. Os modelos M0 a M4

Os cinco modelos foram ajustados por REML (Bates et al., 2015). Como todos têm apenas o intercepto na parte fixa, as verossimilhanças REML são comparáveis (Pinheiro & Bates, 2000). M0 é a regressão de nível único (OLS). M1 tem efeito aleatório de loja e M2, de produto. M3 inclui os dois, e M4 acrescenta a interação loja × produto (o par).
""",
            en="""
## 1. Models M0 to M4

The five models were fitted by REML (Bates et al., 2015). Their REML likelihoods are comparable because each has only an intercept in the fixed part (Pinheiro & Bates, 2000). M0 is a single-level regression (OLS). M1 has a store random effect and M2 a product random effect. M3 includes both, and M4 adds the store × product interaction (the pair).
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
A log-verossimilhança sobe de −563.135 (M0) para −346.786 (M4). O AIC cai de 1.126.273 para 693.583, e AIC e BIC diminuem a cada modelo adicionado. Nenhum ajuste gerou mensagens de convergência. O M4 levou cerca de 29 segundos.
""",
            en="""
The log-likelihood rises from −563,135 (M0) to −346,786 (M4). AIC drops from 1,126,273 to 693,583, and both AIC and BIC decrease with each added model. No fit produced convergence messages. M4 took about 29 seconds.
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
As estatísticas vão de 23.545 (M0 contra M1) a 322.868 (M1 contra M3), com valores p abaixo de 1e-300 em todas as comparações. Os testes dão evidência de componentes de variância de loja, produto e interação loja × produto. Com 347.092 linhas de treino, mesmo componentes pequenos podem ser detectados. A decomposição de M4 quantifica a participação de cada componente.
""",
            en="""
The statistics range from 23,545 (M0 versus M1) to 322,868 (M1 versus M3), with p-values below 1e-300 in every comparison. The tests provide evidence for store, product and store × product variance components. With 347,092 training rows, even small components can be detected. The M4 decomposition quantifies each component's share.
""",
        ),
        md(
            pt="""
## 4. Correlação intraclasse (ICC)

O gráfico mostra os componentes da variância de log(unidades) em cada modelo de M1 a M4 (Raudenbush & Bryk, 2002). Em cada modelo, as participações somam 1.
""",
            en="""
## 4. Intraclass correlation (ICC)

The chart shows the log(units) variance components for each model from M1 to M4 (Raudenbush & Bryk, 2002). The shares sum to 1 within each model.
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

No modelo nulo, as diferenças entre produtos respondem pela maior parcela da variância de log(unidades). O resíduo, de cerca de 25% no M4, capta a variação semanal dentro do par que o modelo deixa sem explicar. Essa parcela será reavaliada com preço e promoção no notebook 03. A participação do resíduo cai de 93,3% em M1 para 24,6% em M4 à medida que são acrescentados outros componentes.
""",
            en="""
In M4, the product accounts for 57.2% of the variance, the residual for 24.6%, the store for 9.6% and the store × \
product interaction for 8.5%. The correlation between two weeks of the same pair is (store + product + pair) / \
total = 75.4%, and the correlation between two different products in the same store is the store share, 9.6%.

In the null model, differences between products account for the largest share of log(units) variance. The residual, about 25% in M4, captures the week-to-week variation within a pair that the model leaves unexplained. This share is re-examined with price and promotion in notebook 03. The residual share falls from 93.3% in M1 to 24.6% in M4 as other components are added.
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
Os BLUPs são desvios previstos para os grupos, encolhidos em direção à média geral (Henderson, 1975; Robinson, 1991). No caso ilustrativo de um único agrupamento com intercepto aleatório, o desvio observado é multiplicado por σ²ᵤ / (σ²ᵤ + σ²ₑ/n). Quanto menor o número de observações do grupo (n), maior é o encolhimento, chamado de *partial pooling*. No M4, os desvios-padrão condicionais são de cerca de 0,07 para lojas e 0,06 para produtos. Os intervalos são estreitos em relação à dispersão entre grupos, sobretudo entre produtos.
""",
            en="""
BLUPs are predicted group deviations shrunk towards the overall mean (Henderson, 1975; Robinson, 1991). As an illustration, in a random-intercept model with a single grouping factor, the observed deviation is multiplied by σ²ᵤ / (σ²ᵤ + σ²ₑ/n). The fewer observations the group has (n), the greater the shrinkage from partial pooling. In M4, the conditional standard deviations are about 0.07 for stores and 0.06 for products. The intervals are narrow relative to the spread across groups, especially across products.
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
A diferença relativa entre as estimativas da variância residual é de 1,3e-6. Para a variância de loja, a diferença é de 0,6% (0,1001 contra 0,1007). O `MixedLM` foi usado nessa verificação com um único agrupamento. Efeitos cruzados no `MixedLM` exigem um grupo único com `vc_formula`. A sintaxe direta de efeitos cruzados motivou a escolha do lme4 para os demais modelos (decisão D1). O desempenho de `MixedLM` com efeitos cruzados neste volume ficou fora da verificação.
""",
            en="""
The relative difference between the residual variance estimates is 1.3e-6. For the store variance, the difference is 0.6% (0.1001 versus 0.1007). `MixedLM` was used for this single-grouping check. Crossed effects in `MixedLM` require a single group with `vc_formula`. The direct syntax for crossed effects motivated the choice of lme4 for the other models (decision D1). The performance of `MixedLM` with crossed effects at this size fell outside the check.
""",
        ),
        md(
            pt="""
## 7. Conclusões e próximo passo

No M4, produtos respondem por 57,2% da variância de log(unidades), lojas por 9,6%, a interação loja × produto por 8,5% e o resíduo por 24,6%. Os testes de razão de verossimilhanças com correção de fronteira dão evidência para cada componente de agrupamento.

O M4 estima uma correlação de 75,4% entre duas semanas do mesmo par e de 9,6% entre produtos diferentes na mesma loja. Na verificação com um único agrupamento, lme4 e statsmodels produziram estimativas próximas.

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

In M4, products account for 57.2% of log(units) variance, stores for 9.6%, the store × product interaction for 8.5% and the residual for 24.6%. The boundary-corrected likelihood-ratio tests provide evidence for each grouping component.

M4 estimates a correlation of 75.4% between two weeks of the same pair and 9.6% between different products in the same store. In the single-grouping check, lme4 and statsmodels produced similar estimates.

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

No segundo passo da seleção por etapas (*step-up*), os preditores da observação semanal entram na parte fixa (Raudenbush & Bryk, 2002). São mantidos os interceptos aleatórios de loja, produto e par loja × produto do modelo nulo M4. Os modelos foram ajustados nas 104 semanas iniciais de treino (D11 revisada). As 52 semanas finais formam a janela de avaliação temporal retrospectiva com exposição exploratória prévia.

As duas versões do notebook leem as execuções gravadas em `outputs/runs` por `scripts/fit_fixed_effects.py` e pelos scripts de sensibilidade. Os coeficientes representam associações condicionais em semanas com transações registradas.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 03 · Fixed effects of price and promotion

In the second step of model selection, predictors at the weekly observation level enter the fixed part (Raudenbush & Bryk, 2002). The random intercepts for store, product and store × product pair are carried over from the M4 null model. The models were fitted on the first 104 weeks of training data (D11, revised). The final 52 weeks form the retrospective temporal evaluation window with prior exploratory exposure.

Both language versions of the notebook read the runs saved in `outputs/runs` by `scripts/fit_fixed_effects.py` and the sensitivity scripts. The coefficients represent conditional associations in weeks with recorded transactions.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_03),
        md(
            pt="""
O identificador e o horário de término da execução são mostrados acima. O treino reúne 347.092 linhas em 3.859 pares loja × produto. `LOG_DISC`, o logaritmo da razão entre preço e preço base, tem média de −0,069 e desvio-padrão de 0,147. Em 25,0% das linhas de treino, `LOG_DISC` fica abaixo de −0,05, o equivalente a um desconto de cerca de 5% ou mais. Em 111 linhas, o preço base foi preenchido com o preço observado (`BASE_PRICE = PRICE`). A seção 8 avalia a sensibilidade a essa regra.
""",
            en="""
The run identifier and finish time are shown above. The training sample has 347,092 rows in 3,859 store × product pairs. `LOG_DISC`, the log ratio of price to base price, has a mean of −0.069 and a standard deviation of 0.147. In 25.0% of training rows, `LOG_DISC` is below −0.05, equivalent to a discount of about 5% or more. In 111 rows, the base price was filled with the observed price (`BASE_PRICE = PRICE`). Section 8 checks sensitivity to this rule.
""",
        ),
        md(
            pt="""
## 1. A especificação

A variável-resposta é log(unidades). As decisões D12 a D14 definem a parte fixa com base na exploração inicial do painel (D11 revisada). Como o log do preço varia sobretudo entre produtos, uma diferença de preço também pode refletir características dos produtos. Por isso o preço foi decomposto em três termos. `LOG_DISC = log(preço / preço base)` mede o desconto temporário e seu coeficiente estima a elasticidade ao desconto (Tellis, 1988; Bijmolt et al., 2005). `LOG_BASE_DEV` mede o desvio do log do preço regular em relação à média do próprio par e seu coeficiente estima a associação com mudanças do preço regular dentro do par ao longo do tempo. `LOG_BASE_MEAN` é a média do log do preço regular do par. Esse termo de Mundlak separa a associação com o nível médio de preço das mudanças dentro do par (Mundlak, 1978). As médias foram calculadas com as linhas de treino. Os valores ausentes de preço base foram preenchidos pela regra D8 revisada, que usa o histórico anterior do par quando disponível.

O desconto e os indicadores de promoção (`FEATURE`, `DISPLAY` e `TPR_ONLY`, todos 0 ou 1) entram sem decomposição entre e dentro de pares. A seção 9 reestima essas associações com efeitos fixos de par e de semana. Cada indicador de promoção é interpretado com o desconto mantido constante. Como o preço fica abaixo do base só em semanas com promoção, são poucas as faixas de desconto em que certas promoções podem ser comparadas (seção 4). Três pares de termos de Fourier (`SIN_k` e `COS_k`, com k de 1 a 3) representam a sazonalidade da semana do ano, e `TREND` representa a tendência linear em anos. O perfil de Fourier é comum a todo o painel e usa seis coeficientes para os dois anos de treino (Hyndman & Athanasopoulos, 2021).

Os testes de razão de verossimilhanças (LR) que comparam parte fixa diferente usam ajustes por **máxima \
verossimilhança (ML)**, porque a verossimilhança restrita (REML) depende da parte fixa e, por isso, não é \
comparável entre esses modelos (Pinheiro & Bates, 2000). Os coeficientes e as variâncias do modelo final, por sua \
vez, vêm de um novo \
ajuste por REML (Bates et al., 2015), que estima as variâncias com menos viés. Os quatro modelos, de F0 a F3, são \
mostrados a seguir.
""",
            en="""
## 1. The specification

The response is log(units). Decisions D12 to D14 specify the fixed part based on the initial exploration of the panel (D11, revised). Since log price varies mainly between products, a difference in price can also reflect product characteristics. Price was therefore decomposed into three terms. `LOG_DISC = log(price / base price)` measures the temporary discount, and its coefficient estimates discount elasticity (Tellis, 1988; Bijmolt et al., 2005). `LOG_BASE_DEV` measures the deviation of log regular price from its pair mean, and its coefficient estimates the association with changes in regular price over time within a pair. `LOG_BASE_MEAN` is the pair mean of log regular price. This Mundlak term separates the association with the average price level from changes within a pair (Mundlak, 1978). The means were calculated from training rows. Missing base prices were filled under revised rule D8, which uses the pair's prior history when available.

The discount and the promotion indicators (`FEATURE`, `DISPLAY` and `TPR_ONLY`, all 0 or 1) enter without a within- and between-pair decomposition. Section 9 re-estimates these associations using pair and week fixed effects. Each promotion indicator is interpreted with the discount held constant. As price falls below base only in promotion weeks, some promotions can be compared over only a few discount bands (section 4). Three pairs of Fourier terms (`SIN_k` and `COS_k`, with k from 1 to 3) represent week-of-year seasonality, and `TREND` represents a linear trend in years. The Fourier profile is shared across the panel and uses six coefficients for the two training years (Hyndman & Athanasopoulos, 2021).

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

Os 347.092 registros de treino dão aos testes sensibilidade para detectar contribuições pequenas. Pela estatística de razão de verossimilhanças, o bloco de preço acrescenta mais ao ajuste (101.800), seguido pelo de promoções (41.481). A estatística do bloco de sazonalidade e tendência (1.225) é cerca de 1/83 da do preço.
""",
            en="""
Here, unlike in notebook 02, the tests do not involve variances on the boundary of the parameter space: they \
concern fixed effects, and the reference is an ordinary χ², which is why `p_value` and `p_naive` coincide. The \
statistics are 101,800 (F0 against F1, 3 degrees of freedom), 41,481 (F1 against F2, 3 degrees) and 1,225 (F2 \
against F3, 7 degrees). The first two p-values are below 1e-300 and the third is 2.69e-260.

The 347,092 training rows give the tests sensitivity to detect small contributions. By the likelihood-ratio statistic, the price block adds the most to fit (101,800), followed by the promotion block (41,481). The statistic for the seasonality and trend block (1,225) is about 1/83 that of price.
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
A tabela converte coeficientes e limites de confiança para variações percentuais nas unidades. A comparação usa uma redução de 10% na razão entre preço e preço base, um aumento de 10% no preço regular, a presença de cada promoção ou a passagem de um ano. Em cada linha, as demais covariáveis ficam constantes.
""",
            en="""
The table converts coefficients and confidence limits to percentage changes in units. The comparisons use a 10% reduction in the price-to-base-price ratio, a 10% increase in regular price, the presence of each promotion or the passage of one year. The other covariates are held constant in each row.
""",
        ),
        code("""
with pd.option_context("display.float_format", "{:,.1f}".format):
    display(interpret.effects_table(fixef, LANG))
"""),
        md(
            pt="""
O coeficiente do desconto é −1,685 (IC 95% de Wald: −1,706 a −1,664), de modo que uma redução de 10% na razão entre preço e preço base \
está associado a cerca de 19,4% mais unidades (IC 95%: 19,2% a 19,7%). O preço regular, medido dentro do par, tem \
coeficiente −1,592, e um preço regular 10% maior está associado a cerca de 14,1% menos unidades. O encarte tem \
coeficiente 0,497, ou seja, cerca de 64,4% mais unidades, e o display, 0,409, ou cerca de 50,5% mais, ambos com o \
desconto e as demais covariáveis mantidos constantes. A tendência é de −0,021 por ano, cerca de 2,1% menos \
unidades a cada ano.

O coeficiente de `TPR_ONLY` é −0,041, ou seja, −4,0% nas unidades (IC 95%: −4,7% a −3,3%). Esse valor é a \
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
The discount coefficient is −1.685 (95% Wald CI: −1.706 to −1.664), so a 10% reduction in the price-to-base-price ratio is associated with \
about 19.4% more units (95% CI: 19.2% to 19.7%). The regular price, measured within the pair, has a coefficient of \
−1.592, and a 10% higher regular price is associated with about 14.1% fewer units. Feature has a coefficient of \
0.497, that is, about 64.4% more units, and display 0.409, or about 50.5% more, both with the discount and the \
other covariates held constant. The trend is −0.021 per year, about 2.1% fewer units each year.

The coefficient of `TPR_ONLY` is −0.041, that is, −4.0% in units (95% CI: −4.7% to −3.3%). This value is \
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

As combinações de desconto e promoção são usadas para comparar cenários de negócio. A tabela cruzada conta as linhas de treino por tipo de promoção e faixa de desconto (`1 − PRICE / BASE_PRICE`).
""",
            en="""
## 4. Joint discount-and-promotion scenarios

Business scenarios compare combinations of discount and promotion. The cross-tabulation counts training rows by promotion type and discount band (`1 − PRICE / BASE_PRICE`).
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
Em relação à semana regular, uma TPR com 10% de desconto está associada a 14,6% mais unidades (9.393 linhas de treino, 2,7%). Com 20%, a diferença estimada é de 39,8% (10.185 linhas, 2,9%). Encarte com 20% (+139,5%, 2.574 linhas), display com 20% (+119,2%, 2.272 linhas) e encarte com display e 25% (+301,8%, 2.767 linhas) têm entre 0,7% e 0,8% das linhas cada. Sem desconto, o display está associado a +50,5% (8.622 linhas, 2,5%) e o encarte a +64,4% (1.055 linhas, 0,3%). O encarte sem desconto tem o menor suporte observado.

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
Relative to a regular week, TPR at a 10% discount is associated with 14.6% more units (9,393 training rows, 2.7%). At 20%, the estimated difference is 39.8% (10,185 rows, 2.9%). Feature at 20% (+139.5%, 2,574 rows), display at 20% (+119.2%, 2,272 rows) and feature with display at 25% (+301.8%, 2,767 rows) each account for 0.7% to 0.8% of rows. Without a discount, display is associated with +50.5% (8,622 rows, 2.5%) and feature with +64.4% (1,055 rows, 0.3%). Feature without a discount has the lowest observed support.

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

Os seis termos de Fourier são combinados em um perfil sazonal, expresso como a diferença percentual nas unidades em relação à semana média do ano.
""",
            en="""
## 5. Seasonality and trend

The six Fourier terms are combined into a seasonal profile, expressed as the percentage difference in units from the average week of the year.
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
diferença entre os extremos é de cerca de 11 pontos percentuais, frente aos 19,4% associados a uma redução de 10% na razão entre preço e preço base. Alguns termos individuais têm coeficientes pequenos (por exemplo, `SIN_1`, −0,002, cujo \
intervalo inclui o zero), mas o bloco como um todo é claramente diferente de zero pelo teste de razão de \
verossimilhanças (1.225). A tendência, por fim, é de uma queda de cerca de 2,1% por ano (IC 95%: −2,5% a −1,8%) \
nas duas temporadas de treino, após o controle de preço, promoção e sazonalidade.
""",
            en="""
The amplitude of seasonality is small compared with the price and promotion associations: the weakest week of the \
year (week 29) has about 4.2% fewer units than the average week, and the strongest (week 2) about 6.6% more. The \
difference between the extremes is about 11 percentage points, to be compared with the 19.4% associated with a 10% reduction in the price-to-base-price ratio. Some individual terms have small coefficients (for example, `SIN_1`, −0.002, whose interval \
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

Os pseudo-R² negativos correspondem a aumentos pequenos nos componentes estimados de loja e par. Ao acrescentar preditores, a variância pode se redistribuir entre os níveis. Neste ajuste, as maiores reduções ocorreram no resíduo e no componente de produto.
""",
            en="""
The residual, which represents the week-to-week movement within the pair, falls by 34.4% (from 0.415 to 0.272), so \
price and promotion explain about a third of the weekly variance within the pair. The product variance falls by \
29.3% (from 0.965 to 0.682): because the regular price level distinguishes products, this is consistent with the \
average price of the pair explaining part of the difference between products. The store (−2.6%) and pair (−4.2%) \
variances, in contrast, increase slightly.

The negative pseudo-R² values correspond to small increases in the estimated store and pair components. Adding predictors can redistribute variance across levels. In this fit, the largest reductions were in the residual and product components.
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

O OLS agrupado combina variação entre pares e dentro dos pares. Se descontos e promoções se concentram em pares com níveis de vendas diferentes, a composição dos pares pode mudar a associação estimada. Essa é uma explicação possível para a diferença observada entre a elasticidade de −0,36 no OLS agrupado e a de −1,69 no multinível.
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

Pooled OLS combines between-pair and within-pair variation. If discounts and promotions are concentrated in pairs with different sales levels, the mix of pairs can change the estimated association. This is one possible explanation for the observed difference between the elasticity of −0.36 in pooled OLS and −1.69 in the multilevel model.
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
O painel tem 141 linhas com preço base de fallback, das quais 111 estão no treino. As linhas de treino têm 18 ocorrências só de encarte, 69 só de display e 24 dos dois. Todas registram desconto zero, mesmo nas semanas em que pode ter havido desconto. Para medir a sensibilidade ao fallback, o F3 foi reajustado por REML após excluir essas 111 linhas (`fixed_effects_no_fallback`).
""",
            en="""
The panel has 141 fallback base-price rows, of which 111 are in training. The training rows include 18 with feature only, 69 with display only and 24 with both. All record a zero discount, even in weeks when a discount may have occurred. To assess sensitivity to the fallback, F3 was refitted by REML after excluding those 111 rows (`fixed_effects_no_fallback`).
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

Os resíduos condicionais de F3 têm correlação serial positiva dentro do par e choques comuns de semana e categoria. Os diagnósticos indicam que a dependência residual amplia a incerteza em relação aos intervalos de Wald, embora as estimativas pontuais fiquem próximas das do modelo com efeitos fixos de par e de semana (seção 9). No estimador de efeitos fixos, os erros-padrão agrupados por par e por semana (Cameron et al., 2011) ficam de 4,1 a 7,4 vezes os de Wald do F3, com maior contribuição do agrupamento por semana. Os sinais das estimativas se mantêm (`fe_sensitivity`, emenda E4 do registro de decisões).
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

F3's conditional residuals show positive within-pair serial correlation and shocks shared by week and category. The diagnostics indicate that residual dependence increases uncertainty relative to the Wald intervals, while point estimates are close to those from the model with pair and week fixed effects (section 9). In the fixed-effects estimator, standard errors clustered by pair and week (Cameron et al., 2011) are 4.1 to 7.4 times the F3 Wald standard errors, with week clustering contributing more. The signs of the estimates are unchanged (`fe_sensitivity`, amendment E4 of the decision log).
""",
        ),
        md(
            pt="""
## 11. Limitações

A rede decide preços e promoções. Ela pode planejar promoções para semanas de demanda alta ou aplicar descontos em resposta a estoques e à concorrência. Nesse caso, parte da associação estimada reflete os critérios de seleção das promoções e dos preços (endogeneidade, Villas-Boas & Winer, 1999). Uma recomendação de preço ou um ROI incremental exige identificação causal e dados de margem. A especificação log-log aproxima a relação com uma elasticidade constante para descontos como 10% e 30%. As comparações entre promoções ficam restritas pelas faixas de desconto em que elas aparecem (seção 4).

A dependência residual foi medida (seção 10): há correlação serial de 0,196 entre semanas consecutivas do mesmo \
par e choques comuns de semana. Ela não foi modelada, e os intervalos de Wald do F3 são condicionais à sua \
especificação e estreitos demais, como indicam os erros-padrão agrupados da seção 9, de 2,2 a 4,1 vezes maiores. \
Além disso, a formulação (a decomposição de preço da D12 e as demais) foi motivada pela exploração inicial do \
painel, que incluiu as 52 semanas finais, e é uma decisão de desenvolvimento. Nenhum modelo foi ajustado nessas \
semanas, e nenhuma comparação preditiva foi calculada nelas.

Foram modeladas as semanas com vendas registradas no par (D3). Semanas sem registro podem corresponder a vendas zero ou a ruptura. O F3 estima uma elasticidade comum a lojas e produtos. A variação da elasticidade entre eles é examinada no notebook 04.
""",
            en="""
The retailer sets prices and promotions. It may plan promotions for high-demand weeks or offer discounts in response to inventory and competition. In that case, part of the estimated association reflects how prices and promotions are selected (endogeneity, Villas-Boas & Winer, 1999). Pricing recommendations or incremental ROI require causal identification and margin data. The log-log specification approximates the relationship using a constant elasticity at discounts such as 10% and 30%. Comparisons between promotions are limited by the discount bands in which they appear (section 4).

Residual dependence was measured (section 10): there is a serial correlation of 0.196 between consecutive weeks of \
the same pair and shocks common to the week. It was not modelled, and the Wald intervals of F3 are conditional on \
its specification and too narrow, as the clustered standard errors of section 9, 2.2 to 4.1 times larger, \
indicate. In addition, the formulation (the price decomposition of D12 and the others) was motivated by the \
initial exploration of the panel, which included the final 52 weeks, and is a development decision. No model was \
fitted on those weeks, and no predictive comparison was computed on them.

The model was fitted to weeks with recorded sales within a pair (D3). Weeks without records may reflect zero sales or stock-outs. F3 estimates a common elasticity across stores and products. Variation in elasticity across stores and products is examined in notebook 04.
""",
        ),
        md(
            pt="""
## 12. Conclusões e próximo passo

Com a hierarquia mantida (interceptos aleatórios de loja, de produto e de par), uma redução de 10% na razão entre preço e preço base está \
associado a cerca de 19,4% mais unidades (elasticidade de −1,685), e um preço regular 10% maior a cerca de 14,1% \
menos. O encarte (+64,4%) e o display (+50,5%) estão associados a mais unidades além do desconto, e a flag de TPR \
tem coeficiente de −0,041 (−4,0%), valor que depende de comparar semanas sem promoção com o mesmo desconto, \
comparação que os dados quase não contêm. Os cenários de TPR da seção 4 estimam +14,6% com 10% de desconto e +39,8% com 20%. Ambos têm suporte nos dados. O encarte sem desconto tem o menor suporte observado (0,3% do treino). A tendência é de −2,1% por ano. A sazonalidade varia de −4,2% a +6,6% em relação à semana média, uma faixa menor que a variação de 19,4% associada ao preço. Todos esses valores são associações \
condicionais em semanas com transações registradas.

Preço e promoção explicam 34,4% da variância semanal dentro do par e 29,3% da variância entre produtos, e os \
pequenos valores negativos de pseudo-R² nos níveis de loja e de par são um artefato conhecido. O OLS agrupado \
estima uma elasticidade ao desconto de −0,36, contra −1,69 do multinível. A exclusão das 111 linhas de fallback \
quase não altera as estimativas, e a regressão com efeitos fixos de par e de semana reproduz as estimativas \
pontuais do F3 em até 0,02, com erros-padrão agrupados de 2,2 a 4,1 vezes maiores. Os resíduos condicionais têm correlação de 0,196 entre semanas consecutivas e apresentam choques comuns de semana. A dependência residual indica que os intervalos de Wald do F3 subestimam a incerteza. Os erros-padrão agrupados por par e por semana, registrados na emenda E4, são de 4,1 a 7,4 vezes os de Wald do F3.

O notebook 04 apresenta os resultados do protocolo pré-registrado P1, aprovado com emendas, sobre se deixar a \
elasticidade ao desconto variar entre lojas e produtos melhora a previsão fora da amostra.
""",
            en="""
## 12. Conclusions and next step

With the hierarchy kept (random intercepts for store, product and pair), a 10% reduction in the price-to-base-price ratio is associated with \
about 19.4% more units (elasticity of −1.685), and a 10% higher regular price with about 14.1% fewer. Feature \
(+64.4%) and display (+50.5%) are associated with more units beyond the discount, and the TPR flag has a \
coefficient of −0.041 (−4.0%), a value that depends on comparing weeks with no promotion at the same discount, a \
comparison that the data hardly contain. The TPR scenarios in section 4 estimate +14.6% at a 10% discount and +39.8% at 20%. Both have support in the data. Feature without a discount has the lowest observed support (0.3% of training rows). The trend is −2.1% per year. Seasonality ranges from −4.2% to +6.6% relative to the average week, a narrower range than the 19.4% change associated with price. All these \
values are conditional associations in weeks with recorded transactions.

Price and promotion explain 34.4% of the weekly variance within the pair and 29.3% of the variance between \
products, and the small negative pseudo-R² values at the store and pair levels are a known artefact. The pooled \
OLS estimates a discount elasticity of −0.36, against −1.69 for the multilevel model. Excluding the 111 fallback \
rows barely changes the estimates, and the regression with pair and week fixed effects reproduces the F3 point \
estimates within 0.02, with clustered standard errors 2.2 to 4.1 times larger. The conditional residuals have a correlation of 0.196 between consecutive weeks and show shocks shared within weeks. Residual dependence indicates that the F3 Wald intervals understate uncertainty. The standard errors clustered by pair and week, recorded in amendment E4, are 4.1 to 7.4 times the F3 Wald standard errors.

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

O protocolo pré-registrado P1 mede o erro de previsão de log(unidades), condicionada ao preço e à promoção observados na semana. Os scripts ajustaram os modelos e fizeram a avaliação final uma vez cada. As duas versões deste notebook leem as mesmas três execuções em `outputs/runs` (`p1_folds`, `gpboost_tuning` e `p1_final`). A seção 9 consulta uma quarta execução, com o teste inferencial complementar feito somente no treino após a conclusão do P1.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 04 · Protocol P1: random slopes and out-of-sample comparison

The pre-registered P1 protocol measures prediction error for log(units), conditional on the price and promotion observed that week. Scripts fitted the models and ran the final evaluation once each. Both language versions of this notebook read the same three runs in `outputs/runs` (`p1_folds`, `gpboost_tuning` and `p1_final`). Section 9 reads a fourth run for the supplementary inferential test conducted on training data alone after P1 ended.

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

O notebook 03 estimou uma elasticidade ao desconto comum a todos os pares loja × produto. O P1 avalia se permitir que essa elasticidade varie entre lojas e produtos melhora a previsão fora da amostra, sobretudo para pares com pouco histórico. Para isso, testa inclinações aleatórias de `LOG_DISC` (Gelman & Hill, 2007). Quando há poucas semanas no histórico, a elasticidade ajustada só com os dados do par pode ser instável. O encolhimento em direção às médias da loja e do produto pode estabilizar a estimativa (Robinson, 1991).

O protocolo e suas seis emendas (E1 a E6) foram aprovados antes do ajuste de qualquer candidato, conforme a prática de pré-registro (Nosek et al., 2018; Weston et al., 2019). O registro fixou os candidatos, a ordem de comparação, a regra de parada, a métrica e as populações. As emendas definiram o F3 do notebook 03 como candidato inicial C0 (E1), a cobertura do OLS por par (E2), a regra sequencial de parada (E3) e o congelamento antes da avaliação final (E6). Os limiares e as fórmulas já estavam definidos quando saíram os resultados das dobras internas.

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

A sequência compara C1, C2, C3 e C4, nessa ordem, sempre com o último modelo aceito. Para ser aceito, o candidato precisa convergir sem singularidade (Bates et al., 2015), com nova tentativa pelo otimizador `nloptwrap` em caso de aviso. O MAE por par deve ser menor nas duas dobras internas, com redução relativa de pelo menos 0,5% na média das dobras. Se alguma condição falhar, a sequência termina e mantém o último modelo aceito.

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

Notebook 03 estimated a common discount elasticity across all store × product pairs. P1 asks whether allowing that elasticity to vary across stores and products improves out-of-sample prediction, especially for pairs with little history. It tests random slopes for `LOG_DISC` (Gelman & Hill, 2007). With few weeks of history, an elasticity fitted using only the pair's own data can be unstable. Shrinkage towards the store and product means can stabilise the estimate (Robinson, 1991).

The protocol and its six amendments (E1 to E6) were approved before any candidate was fitted, in line with pre-registration practice (Nosek et al., 2018; Weston et al., 2019). The record fixed the candidates, comparison order, stopping rule, metric and populations. The amendments defined F3 from notebook 03 as the initial candidate C0 (E1), specified coverage for per-pair OLS (E2), and set the sequential stopping rule (E3) and the freeze before final evaluation (E6). The thresholds and formulae were already in place when the internal fold results came in.

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

The sequence compares C1, C2, C3 and C4 in that order, each against the last accepted model. To qualify, a candidate must converge without singularity (Bates et al., 2015), with a retry using the `nloptwrap` optimiser if a warning occurs. The per-pair MAE must be lower in both internal folds, with a relative reduction of at least 0.5% on average across folds. If any condition fails, the sequence ends and retains the last accepted model.

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

Antes da avaliação final, o modelo selecionado (C0), os parâmetros do \
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

Before the final evaluation, the selected model (C0), the \
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
## 6. Primeira conclusão sobre a seleção de C0 e os pares curtos

A regra do P1 selecionou o C0 pelo desempenho nas dobras internas. O C1 reduziu o MAE por par no estrato com menos de 26 semanas nas duas dobras. Considerando todos os pares vistos, o erro aumentou na dobra A e o ganho médio de −0,49% ficou abaixo do limiar de 0,5%.
""",
            en="""
## 6. First conclusion on C0 selection and short-history pairs

The P1 rule selected C0 based on performance in the internal folds. C1 reduced per-pair MAE in the stratum with fewer than 26 weeks in both folds. Across all seen pairs, error rose in fold A and the average gain of −0.49% fell below the 0.5% threshold.
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
No estrato com menos de 26 semanas, o MAE por par do C1 foi 3,70% menor na dobra A (246 pares) e 0,85% menor na dobra B (221 pares), em relação ao C0. São diferenças pontuais, sem análise de incerteza. Na dobra A, o estrato de 26 a 77 semanas concentra a maior parte dos pares, e ali o erro do C1 aumentou.

Pela regra do P1, a janela final avaliou apenas o candidato selecionado, o C0. Ele compartilha informação pelos interceptos e estima uma inclinação de desconto **comum** a todos os pares. A questão de saber se o ganho do C1 nos pares curtos das dobras se manteria na janela final continua em aberto.
""",
            en="""
In the stratum with fewer than 26 weeks, C1's per-pair MAE was 3.70% lower in fold A (246 pairs) and 0.85% lower in fold B (221 pairs) than C0's. These are point differences without an uncertainty analysis. In fold A, most pairs are in the 26 to 77 weeks stratum, where C1's error rose.

Under the P1 rule, the final window evaluated only the selected candidate, C0. It shares information through the intercepts and estimates a discount slope **common** to all pairs. Whether C1's gain on short-history pairs in the folds would hold in the final window remains an open question.
""",
        ),
        md(
            pt="""
## 7. Segunda conclusão sobre C0 e GPBoost na janela final

Nos pares vistos, o GPBoost teve MAE por par 0,47% menor que o do C0 (0,4520 contra 0,4541). O C0 teve menor MAE por linha (0,4446 contra 0,4478) e menor MAE por par nos pares inéditos (0,6260 contra 0,6876).
""",
            en="""
## 7. Second conclusion on C0 and GPBoost in the final window

On seen pairs, GPBoost's per-pair MAE was 0.47% lower than C0's (0.4520 versus 0.4541). C0 had lower MAE per row (0.4446 versus 0.4478) and lower per-pair MAE on unseen pairs (0.6260 versus 0.6876).
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
A cobertura do OLS por par segue a regra operacional da emenda E2. A regra trata como não estimado um termo \
sem variação na janela de ajuste do par ou descartado por colinearidade. Uma linha fica \
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
Per-pair OLS coverage follows the operational rule in amendment E2. The rule treats a term as unestimated if it has no \
variation in the pair's fit window or was dropped because of collinearity. A row is uncovered when \
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

O bootstrap por pares para medir a incerteza da diferença de erro preditivo continua adiado. Pontuar o C1 na janela final seria uma exploração posterior e exige aprovação.

Os notebooks 05 e 06 concluíram as demais análises nas semanas de treino, preservando o C0 e o resultado da comparação preditiva. O notebook 05 testou os contrastes secundários do P1 para as interações do desconto com segmento, marca própria e categoria, com correção de Holm (Holm, 1979). Também avaliou a sensibilidade ao efeito aleatório de categoria × semana. O notebook 06 comparou cenários de preço e promoção.
""",
            en="""
## 10. What remains deferred

The pair bootstrap to measure uncertainty in the difference in prediction error remains deferred. Scoring C1 in the final window would be a later exploratory analysis and requires approval.

Notebooks 05 and 06 completed the remaining analyses in the training weeks, retaining C0 and the results of the predictive comparison. Notebook 05 tested P1's secondary contrasts for discount interactions with segment, private label and category, using the Holm correction (Holm, 1979). It also assessed sensitivity to a category × week random effect. Notebook 06 compared price and promotion scenarios.
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

Os contrastes planejados avaliam se a associação entre desconto e vendas varia por segmento da loja, tipo de marca (de fabricante ou própria) ou categoria. A correção de Holm abrange essa família fixa de três testes. O diagnóstico residual do notebook 03 motiva a segunda análise, que compara o F3 com uma versão que acrescenta um intercepto aleatório de categoria × semana.

Os scripts `scripts/fit_contrasts.py` e `scripts/fit_catweek_sensitivity.py` ajustaram os modelos nas 104 semanas de treino (0 a 103) e gravaram os resultados em `outputs/runs`. As duas versões do notebook leem as mesmas execuções. As estimativas descrevem associações condicionais nas semanas com transações registradas. A seleção do C0 e a classificação dos modelos no P1 se baseiam no erro de previsão fora da amostra, documentado no notebook 04.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 05 · Planned contrasts and sensitivity to category × week shocks

The planned contrasts test whether the association between discount and sales varies by store segment, brand type (manufacturer brand or private label) or category. The Holm correction covers this fixed family of three tests. The residual diagnostic in notebook 03 motivates the second analysis, which compares F3 with a version that adds a category × week random intercept.

The scripts `scripts/fit_contrasts.py` and `scripts/fit_catweek_sensitivity.py` fitted the models on the 104 training weeks (0 to 103) and stored the results in `outputs/runs`. Both language versions read the same runs. The estimates describe conditional associations in weeks with recorded transactions. The selection of C0 and the P1 model ranking are based on out-of-sample prediction error, documented in notebook 04.

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

Os efeitos fixos de par absorvem os efeitos principais de segmento, tipo de marca e categoria, pois cada um desses atributos é constante dentro do par. O desconto varia ao longo das semanas do par. Essa variação permite estimar a interação com `LOG_DISC` comparando, entre níveis, a relação entre vendas e descontos dentro de cada par.

Os contrastes comparam inclinações entre grupos definidos pelos moderadores. No protocolo P1 (notebook 04), o candidato C1 permite que a elasticidade ao desconto varie aleatoriamente entre lojas e produtos (Gelman & Hill, 2007). O P1 avalia aquele candidato pelo erro de previsão fora da amostra.

## 2. As populações e o suporte por nível

A primeira questão operacional é quais linhas entram em cada teste. A tabela a seguir mostra, para cada moderador, as linhas do treino, as linhas usadas, os pares e as exclusões.
""",
            en="""
The two runs that were read are identified above (identifier and finish time). The identifiers are fixed in the code, so re-running the notebook always reads the same artefacts.

## 1. The question and the design of the contrasts

The question is whether the slope of log(units) on the log of the discount (`LOG_DISC`) is the same at every level of a moderator. The moderators are the store segment (reference `MAINSTREAM`), the brand type (reference: manufacturer brand) and the category (reference `BAG SNACKS`, the first in alphabetical order). One model was fitted for each moderator, by least squares with store × product pair and week fixed effects, obtained through the within transformation (Wooldridge, 2010). The response is `LOG_UNITS`, and the controls are `LOG_DISC`, `LOG_BASE_PRICE`, `FEATURE`, `DISPLAY` and `TPR_ONLY`, the same as in the within-pair sensitivity of notebook 03. The model with interaction adds `LOG_DISC` multiplied by the indicator of each level other than the reference, and the restricted version, with the same sample and the same controls, omits those interactions.

Pair fixed effects absorb the main effects of segment, brand type and category because each attribute is constant within a pair. Discounts vary across weeks within a pair. That variation identifies the interaction with `LOG_DISC` by comparing the within-pair relationship between sales and discounts across levels.

The contrasts compare slopes across groups defined by the moderators. In protocol P1 (notebook 04), candidate C1 allows discount elasticity to vary randomly across stores and products (Gelman & Hill, 2007). P1 evaluates that candidate using out-of-sample prediction error.

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

Uma inclinação mais negativa indica que uma mesma variação do desconto está associada a uma variação maior nas unidades vendidas. A tabela mostra, por nível, a inclinação estimada, seus erros-padrão, a diferença em relação à referência e os respectivos intervalos de 95%. Os intervalos são pontuais, calculados com o erro-padrão em duas vias e sem ajuste para multiplicidade. As diferenças por nível são descritivas. A inferência se limita à família planejada de três testes globais.
""",
            en="""
## 4. Slopes by level

A more negative slope indicates that the same discount change is associated with a larger change in units sold. For each level, the table reports the estimated slope, its standard errors, the difference from the reference and the corresponding 95% intervals. The intervals are pointwise, calculated using the two-way standard error, with no multiplicity adjustment. Differences by level are descriptive. Inference is limited to the planned family of three global tests.
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

As diferenças condicionais entre níveis podem refletir fatores que o modelo não representa, como a distribuição dos descontos praticados e o calendário promocional. A seção 2 mostrou que o desvio-padrão intrapar do desconto varia entre níveis. A inclinação comum de `LOG_DISC` no OLS intrapar restrito é de cerca de −1,70, com efeitos fixos de par e semana e controle pelo log do preço-base. Para os cenários do notebook 06, a inclinação vem do F3 (−1,685). O F3 usa interceptos aleatórios, decompõe o preço-base em desvio e média do par e inclui tendência e sazonalidade. Os valores são próximos e vêm de estimadores e controles distintos.

## 5. Sensibilidade a choques comuns de categoria × semana

O diagnóstico residual do notebook 03 mostrou que as médias semanais dos resíduos condicionais do F3 variam muito mais do que o esperado sob independência entre linhas, e que a parcela da variância residual associada à combinação de categoria e semana é de 6,4%, contra 0,1% esperado. A questão é se as associações estimadas pelo F3 são sensíveis à representação desses choques comuns. A comparação é, portanto, uma sensibilidade motivada pelo diagnóstico do notebook 03, e não uma troca da especificação oficial.

A referência é o F3 do notebook 03, o C0 do P1, com a parte fixa das decisões D12 a D14 e interceptos aleatórios de loja, produto e par, ajustado por REML com o `lme4` (Bates et al., 2015). A sensibilidade acrescenta `(1 | CATEGORY:WEEK)`, nas mesmas linhas, com a mesma parte fixa e as mesmas médias aprendidas no treino. Um intercepto de categoria × semana representa desvios contemporâneos, comuns a todas as lojas e produtos de uma categoria numa mesma semana. Os efeitos específicos de categoria × semana são condicionais às células já observadas no treino, e não se projetam para semanas futuras. As tabelas a seguir mostram as contagens e a convergência dos dois ajustes.
""",
            en="""
By segment, the slopes are −1.746 (MAINSTREAM, the reference), −1.580 (UPSCALE) and −1.686 (VALUE). The UPSCALE difference from the reference is 0.166, with a pointwise interval from 0.072 to 0.261, and the VALUE difference is 0.060, with an interval from −0.052 to 0.172, which contains zero. Among the level differences of the segment, only that of UPSCALE has a pointwise interval that excludes zero. This reading is descriptive, because the interval is pointwise and has no multiplicity adjustment. The global segment test indicates heterogeneity across the set of segments and does not, by itself, identify which segment differs.

By brand type, the slope is −1.906 for manufacturer brand and −1.411 for private label, a difference of 0.494 (interval from 0.316 to 0.673): the association of the discount with units is less intense for the private label. By category, the slopes are −1.083 (BAG SNACKS, the reference), −2.044 (COLD CEREAL), −2.416 (FROZEN PIZZA) and −1.387 (ORAL HYGIENE PRODUCTS), with differences from the reference of −0.961 (interval from −1.215 to −0.708), −1.333 (from −1.603 to −1.064) and −0.305 (from −0.458 to −0.152). The discount is associated with a larger response in frozen pizza and in cereal than in pretzels.

Conditional differences across levels may reflect factors outside the model, such as the distribution of discounts and the promotional calendar. Section 2 showed that the within-pair standard deviation of the discount varies across levels. The common `LOG_DISC` slope in restricted within-pair OLS is about −1.70, with pair and week fixed effects and log base price as a control. The scenarios in notebook 06 use the slope from F3 (−1.685). F3 uses random intercepts, splits base price into the pair deviation and mean, and includes trend and seasonality. The values are close and come from different estimators and controls.

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

Após a correção de Holm sobre a família fixa de três testes, cada teste global rejeita a 5% a hipótese de inclinações iguais entre os níveis do seu moderador. Os valores-p ajustados são 0,0031 para segmento, 5,51e-07 para tipo de marca e 7,83e-16 para categoria. O teste de segmento tem a evidência mais fraca dos três. As diferenças por nível descrevem associações condicionais com intervalos pontuais. A inferência global é aproximada, pois o agrupamento em duas vias pode deixar de fora a dependência entre pares diferentes em semanas diferentes, como choques comuns que persistem entre semanas.

Ao acrescentar o intercepto aleatório de categoria × semana, o coeficiente do desconto vai de −1,685 a −1,727 e o do preço regular dentro do par, de −1,592 a −1,660. Os coeficientes de promoção mudam menos de 0,007. O sinal e a ordem de grandeza das associações do F3 se mantêm nessa análise de sensibilidade. Só a tendência perde precisão. O intercepto absorve choques comuns da mesma semana, e a persistência residual dentro do par continua presente.

No OLS intrapar, as inclinações estimadas por categoria variam de −1,083 a −2,416. Os cenários do notebook 06 usam a inclinação comum do F3 (−1,685) para resumir associações que variam entre categorias.
""",
            en="""
## 6. What the evidence allows one to conclude

After the Holm correction for the fixed family of three tests, each global test rejects equal slopes across the levels of its moderator at the 5% level. Adjusted p-values are 0.0031 for segment, 5.51e-07 for brand type and 7.83e-16 for category. The segment test has the weakest evidence of the three. Differences by level describe conditional associations with pointwise intervals. Global inference is approximate because two-way clustering can miss dependence between different pairs in different weeks, such as common shocks that persist across weeks.

Adding the category × week random intercept moves the discount coefficient from −1.685 to −1.727 and the within-pair regular price coefficient from −1.592 to −1.660. The promotion coefficients change by less than 0.007. The sign and order of magnitude of the F3 associations hold in this sensitivity analysis. Only the trend loses precision. The intercept absorbs common shocks in the same week, and within-pair residual persistence remains.

In within-pair OLS, the estimated category slopes range from −1.083 to −2.416. The scenarios in notebook 06 use the common F3 slope (−1.685) to summarise associations that vary across categories.
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

A partir das estimativas do F3 no notebook 03, este notebook compara cenários de preço e promoção e mostra quantas observações do treino sustentam cada combinação. A seção 4 compara TPR e display com o mesmo desconto de 20%.

O script `scripts/fit_scenarios.py` reajustou o F3 por REML nas semanas de treino (0 a 103). Em `outputs/runs`, gravou os coeficientes, a covariância conjunta, a tabela de cenários e a conciliação com o notebook 03. As duas versões do notebook leem esse reajuste e a execução do F3 original. As estimativas descrevem associações condicionais nas semanas com vendas registradas.

*Dados: dunnhumby Source Files – Breakfast at the Frat.*
""",
            en="""
# 06 · Pricing and promotion scenarios

Using the F3 estimates from notebook 03, this notebook compares price and promotion scenarios and shows how many training observations support each combination. Section 4 compares TPR and display at the same 20% discount.

The script `scripts/fit_scenarios.py` refitted F3 by REML on the training weeks (0 to 103). It stored the coefficients, joint covariance, scenario table and reconciliation with notebook 03 in `outputs/runs`. Both language versions read this refit and the original F3 run. The estimates describe conditional associations in weeks with recorded sales.

*Data: dunnhumby Source Files – Breakfast at the Frat.*
""",
        ),
        lang_cell(),
        code(SETUP_06),
        md(
            pt="""
As duas execuções lidas estão identificadas acima (identificador e horário de término), e os identificadores são fixados no código, de modo que reexecutar o notebook lê sempre os mesmos artefatos.

## 1. A pergunta e a definição dos cenários

A referência é uma semana regular, sem desconto e sem promoção. Cada cenário estima quanto as vendas modeladas diferem dessa semana e verifica o suporte da combinação no treino. Para um desconto d e indicadores de encarte f, display s e TPR r, a diferença no log das unidades é Δlog = β_desconto · log(1 − d) + β_encarte · f + β_display · s + β_TPR · r. O preço-base e os demais controles ficam constantes. A fórmula `100 × expm1(Δlog)` expressa a variação percentual associada à razão modelada de vendas. O indicador `TPR_ONLY` representa uma redução temporária de preço sem encarte nem display. A tabela mostra como os oito cenários aplicam essa regra.

O notebook 03 registrou os oito cenários antes dos resultados dos contrastes do notebook 05.
""",
            en="""
The two runs that were read are identified above (identifier and finish time), and the identifiers are fixed in the code, so re-running the notebook always reads the same artefacts.

## 1. The question and the definition of the scenarios

The reference is a regular week with no discount or promotion. Each scenario estimates how much modelled sales differ from that week and checks support for the combination in training. For a discount d and indicators for feature f, display s and TPR r, the difference in log(units) is Δlog = β_discount · log(1 − d) + β_feature · f + β_display · s + β_TPR · r. Base price and other controls are held constant. The formula `100 × expm1(Δlog)` expresses the percentage change associated with the modelled sales ratio. The `TPR_ONLY` indicator represents a temporary price reduction without feature or display. The table shows how the eight scenarios apply that rule.

Notebook 03 registered all eight scenarios before the contrast results in notebook 05.
""",
        ),
        code("""
design = pd.DataFrame(interpret.DEFAULT_SCENARIOS, columns=["scenario", "discount", "feature", "display", "tpr"])
display(localized(design.assign(discount=design["discount"] * 100)))
"""),
        md(
            pt="""
Os intervalos pontuais de 95% usam a covariância conjunta dos coeficientes do F3, ajustado por REML com o `lme4` (Bates et al., 2015). Essa matriz fornece o erro-padrão da combinação linear Δlog e leva em conta as covariâncias entre os coeficientes. A mesma fórmula usada para calcular o efeito percentual retransforma os limites.

## 2. Conciliação com o notebook 03

A tabela verifica se os 14 coeficientes do reajuste que gerou a covariância conjunta coincidem com os coeficientes do F3 gravados no notebook 03.
""",
            en="""
The pointwise 95% intervals use the joint covariance of the F3 coefficients fitted by REML with `lme4` (Bates et al., 2015). This matrix gives the standard error of the linear combination Δlog and accounts for covariance between coefficients. The interval limits are back-transformed with the same formula used for the percentage effect.

## 2. Reconciliation with notebook 03

The table checks whether the 14 coefficients from the refit that produced the joint covariance match the stored F3 coefficients from notebook 03.
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
Com 20% de desconto, a TPR está associada a +39,8% nas vendas modeladas em relação à semana regular (de 39,0% a 40,5%). Para display, a variação é de +119,2% (de 117,7% a 120,7%). A razão modelada de vendas de display em relação à TPR é de 1,568 (intervalo pontual de 1,555 a 1,581), o que corresponde a cerca de 57% a mais no cenário de display. Há observações para os dois cenários. A escolha de quando e onde cada modalidade foi aplicada pode influenciar essa associação condicional.

## 5. O suporte observado

A leitura dos cenários só faz sentido se houver linhas observadas que os realizem. O suporte é contado nas semanas 0 a 103, para a combinação exata de modalidades e com desconto dentro de ± 3 pontos percentuais do valor do cenário. A tabela mostra as linhas, os pares distintos e a participação nas linhas de treino, o mesmo suporte restrito às linhas com preço-base observado (sem `BASE_IMPUTED`), o número de linhas de suporte que vieram do preço-base de fallback (`BASE_PRICE = PRICE`, que zera o desconto da linha) e a indicação de cenário fora do suporte observado.
""",
            en="""
At a 20% discount, TPR is associated with +39.8% in modelled sales relative to a regular week (39.0% to 40.5%). For display, the change is +119.2% (117.7% to 120.7%). The modelled sales ratio for display relative to TPR is 1.568 (pointwise interval 1.555 to 1.581), or about 57% higher modelled sales in the display scenario. Both scenarios have observed support. The choice of when and where each promotion ran can influence this conditional association.

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
Todos os cenários têm suporte observado. A semana regular reúne 247.331 linhas (71,26% do treino, 3.858 pares). Cada cenário promocional reúne entre 0,30% e 2,93% das linhas. Com desconto de 10% e 20%, a TPR tem 9.393 linhas (2,71%, 2.005 pares) e 10.185 linhas (2,93%, 2.261 pares), respectivamente. Sem desconto, o encarte tem 1.057 linhas (0,30%, 641 pares) e o display, 8.624 (2,48%, 1.648 pares). Com 20%, o encarte tem 2.574 linhas (0,74%, 1.212 pares) e o display, 2.273 (0,65%, 883 pares). Encarte com display e desconto de 25% tem 2.770 linhas (0,80%, 968 pares). Restringir o suporte às linhas com preço-base observado muda pouco as contagens.

As linhas com preço-base de fallback, que têm desconto zero por construção, aparecem apenas nos cenários sem desconto: 18 linhas no encarte sem desconto (de 1.057) e 69 no display sem desconto (de 8.624). Em relação à tabela do notebook 03, quatro contagens diferem em 1 a 3 linhas (encarte sem desconto, 1.057 contra 1.055; display sem desconto, 8.624 contra 8.622; display com 20%, 2.273 contra 2.272; encarte com display e 25%, 2.770 contra 2.767), porque a borda da janela de ± 3 pontos percentuais passou a ser contada como incluída, com tolerância numérica.

O tratamento de um cenário sem suporte está implementado no código e coberto por testes sintéticos: a linha permanece na tabela, é marcada como fora do suporte observado e não é apresentada como aplicação de negócio. Esse caso não ocorre aqui, e a coluna correspondente continua visível. Não se mostra um recorte do suporte por grupo, porque o F3 usa inclinações comuns a todos os grupos: um recorte por segmento ou por categoria descreveria apenas onde estão os dados, e não efeitos específicos de grupo. O notebook 05 mostrou que as inclinações do desconto variam por nível, de −1,083 a −2,416 entre categorias, de modo que os valores acima descrevem uma associação média, e não a de uma categoria ou de um segmento em particular.

## 6. Limites e escopo

Os cenários estimam razões modeladas de vendas entre condições, usando semanas com vendas registradas e uma forma aditiva em log. O calendário promocional, as rupturas e outros fatores podem influenciar essas associações. A seleção de quando e onde aplicar uma promoção pode enviesar as estimativas, em especial a de preço (Villas-Boas & Winer, 1999). Os intervalos pontuais têm cobertura nominal de 95% sob o modelo. Na sensibilidade intrapar, os erros-padrão agrupados foram muito maiores, o que limita a leitura dos intervalos como medida de incerteza total.

O incremento estimado é de vendas (Blattberg & Neslin, 1989). Uma análise de rentabilidade precisaria considerar custos de promoção, margem, canibalização e retorno sobre o investimento. O suporte observado informa a comparação entre cenários. Prever vendas em condições controladas ou recomendar preços com base em efeitos causais exigiria outro desenho.
""",
            en="""
All scenarios have observed support. The regular week accounts for 247,331 rows (71.26% of training rows, 3,858 pairs). Each promotional scenario accounts for between 0.30% and 2.93% of rows. At discounts of 10% and 20%, TPR has 9,393 rows (2.71%, 2,005 pairs) and 10,185 rows (2.93%, 2,261 pairs), respectively. Without a discount, feature has 1,057 rows (0.30%, 641 pairs) and display has 8,624 (2.48%, 1,648 pairs). At 20%, feature has 2,574 rows (0.74%, 1,212 pairs) and display has 2,273 (0.65%, 883 pairs). Feature with display at a 25% discount has 2,770 rows (0.80%, 968 pairs). Restricting support to rows with an observed base price changes the counts little.

The rows with a fallback base price, which have zero discount by construction, appear only in the scenarios without a discount: 18 rows in feature without discount (of 1,057) and 69 in display without discount (of 8,624). Relative to the table of notebook 03, four counts differ by 1 to 3 rows (feature without discount, 1,057 against 1,055; display without discount, 8,624 against 8,622; display with 20%, 2,273 against 2,272; feature with display at 25%, 2,770 against 2,767), because the edge of the ± 3 percentage-point window is now counted as included, with numerical tolerance.

The handling of a scenario without support is implemented in the code and covered by synthetic tests: the row stays in the table, is flagged as outside the observed support and is not presented as a business application. That case does not occur here, and the corresponding column stays visible. No cut of the support by group is shown, because the F3 uses slopes common to all groups: a cut by segment or by category would describe only where the data are, not group-specific effects. Notebook 05 showed that the discount slopes vary by level, from −1.083 to −2.416 across categories, so the values above describe an average association, not that of a particular category or segment.

## 6. Limits and scope

The scenarios estimate modelled sales ratios across conditions using weeks with recorded sales and an additive log specification. The promotional calendar, stockouts and other factors can influence these associations. Selection of when and where to run a promotion can bias the estimates, particularly the price estimate (Villas-Boas & Winer, 1999). The pointwise intervals have nominal 95% coverage under the model. The clustered standard errors in the within-pair sensitivity analysis were much larger, which limits the interpretation of the intervals as a measure of total uncertainty.

The estimated increment is in sales (Blattberg & Neslin, 1989). A profitability analysis would need to account for promotion costs, margins, cannibalisation and return on investment. Observed support informs the comparison across scenarios. Forecasting sales under controlled conditions or recommending prices based on causal effects would require a different design.
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

Os cenários usam a inclinação comum do desconto do F3 (−1,685). Os intervalos estreitos medem a precisão dos contrastes construídos com os coeficientes do F3. No teste complementar do notebook 04 (execução `discount_slope_lr`), o C1 acrescenta ao C0 inclinações aleatórias do desconto por produto e por loja (Gelman & Hill, 2007). No C1, as inclinações variam em torno da própria inclinação fixa do C1, com desvios-padrão estimados de 0,884 entre produtos e 0,274 entre lojas. Como comparação descritiva de escala, esses desvios equivalem a cerca de 52% e 16% do módulo da inclinação do F3, respectivamente. A associação estimada para um produto específico pode diferir substancialmente da associação média dos cenários. Como a execução não gravou a estimativa fixa de `LOG_DISC` do C1, os cálculos de cenário se limitam ao F3, o C0 selecionado pelo protocolo P1.
""",
            en="""
### Dispersion of the discount slope across products

The scenarios use F3's common discount slope (−1.685). The narrow intervals measure the precision of contrasts built from the F3 coefficients. In the supplementary test in notebook 04 (run `discount_slope_lr`), C1 adds random discount slopes by product and store to C0 (Gelman & Hill, 2007). In C1, the slopes vary around C1's own fixed slope, with estimated standard deviations of 0.884 across products and 0.274 across stores. For a descriptive comparison of scale, those standard deviations are about 52% and 16% of the absolute F3 slope, respectively. The estimated association for an individual product can differ substantially from the average association in the scenarios. Because the run lacks a stored fixed `LOG_DISC` estimate for C1, the scenario calculations use F3, the C0 selected by the P1 protocol.
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
