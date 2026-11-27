# Uplift modeling para targeting de campanha: Hillstrom

Relatório executivo: [Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) · [English](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/en.html)

Idioma: [English](README.md) | Português (Brasil)

Este projeto investiga quem deve receber uma ação de marketing porque a ação muda o resultado do cliente, distinguindo o efeito causal da simples probabilidade de resposta. O experimento randomizado Hillstrom MineThatData serve de base para comparar targeting por propensão, estimadores de CATE, uplift trees, causal forests e políticas de contato sob restrição de orçamento.

A análise forma um fluxo modular, testado e bilíngue, com avaliação pré-registrada em teste selado. A conclusão principal é conservadora: a hipótese confirmatória primária não foi sustentada, e as políticas resultantes ficam restritas a estudo e desenho de piloto, sem liberação para deployment direto.

## Escopo e evidências

Pode sustentar: pesquisa de targeting causal e desenho de um piloto randomizado prospectivo.

Não sustenta: deployment automático de política, ROI validado ou alegações retrospectivas de superioridade.

Superfície de engenharia: componentes modulares em `src/`, workflow de notebooks bilíngue, artefatos persistidos e testes automatizados; nenhuma API produtiva de serving é exposta.

Camada de governança: o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Promotion_Campaign_Uplift_Modelling/) preserva o resultado do teste selado e a fronteira `not_ready_for_direct_deployment`.

## Resultado executivo

A comparação primária pré-registrada foi a diferença de Qini AUC entre a `UpliftTree` e o baseline de propensão no teste selado de 12.800 clientes.

| Evidência | Resultado | Interpretação |
|---|---:|---|
| `UpliftTree - baseline de propensão` | -0,0088; IC 95% [-0,0492; 0,0302] | Sem vantagem confirmatória |
| Qini AUC absoluta da `X+Tree(depth=4)` | 0,0470; IC 95% [0,0180; 0,0749] | Melhor ranking absoluto no teste selado, sem promoção retrospectiva |
| Alinhamento dos rankings visit-spend | Spearman 0,0941 | Targeting para visitas não se transfere de forma confiável para gasto |
| Fronteira decisória final | `not_ready_for_direct_deployment` | Uso restrito a aprendizado e desenho de piloto prospectivo |

As etapas de heterogeneidade, política e ROI são exploratórias e pós-confirmatórias. Elas preservam o resultado primário nulo sem reabrir a seleção de modelos.

## Problema de negócio

Um modelo de propensão ordena clientes pela probabilidade de um outcome observado, o que pode consumir orçamento com clientes que responderiam mesmo sem e-mail. Uplift modeling estima o efeito condicional do tratamento:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]
$$

A questão operacional passa a definir quais clientes contatar, com qual campanha, sob um orçamento de contato e premissas econômicas explícitas.

## Dataset e outcomes

O [Hillstrom MineThatData E-Mail Analytics Challenge](http://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) contém 64.000 clientes randomizados em três braços.

| Braço | Clientes | Descrição |
|---|---:|---|
| No E-Mail | 21.306 | Controle |
| Mens E-Mail | 21.307 | Campanha de produtos masculinos |
| Womens E-Mail | 21.387 | Campanha de produtos femininos |

O dataset tem oito características pré-tratamento, uma coluna de tratamento e três outcomes:

| Outcome | Papel | Prevalência ou escala observada |
|---|---|---|
| `visit` | Primário | 14,7% visitaram o site |
| `conversion` | Secundário | 0,9% compraram; 578 eventos |
| `spend` | Secundário | Proxy de receita com grande massa em zero |

O outcome primário é `visit` porque `conversion` é raro demais para estimação estável de CATE em subgrupos neste tamanho amostral. O projeto testa se um ranking otimizado para engajamento de topo de funil se transfere para conversão ou receita.

## Contrato experimental

O experimento randomizado controlado original com três braços sustenta a identificação causal sob as premissas de randomização verificadas no Notebook 01. O desenvolvimento utiliza divisão determinística de 60%/20%/20% em treino, validação e teste selado, estratificada para preservar tratamento e outcome, com agregação binária dos dois braços de e-mail contra `No E-Mail`.

O Notebook 05 avalia candidatos congelados uma única vez no teste selado, com 2.000 repetições de bootstrap. As seções S7-S9 consomem saídas persistidas sem reajustar candidatos selados, repetir predições seladas ou escolher um modelo vencedor retrospectivo. A interpretação emprega perfis por quantil e um surrogate raso para comunicação, sem uso de SHAP ou alegação de explicação causal individual.

## Fluxo de notebooks

Oito pares de notebooks implementam nove seções analíticas. A edição em português (PT-BR) funciona como referência, e a versão em inglês (EN-US) mantém equivalência editorial com o mesmo código e estado analítico persistido.

| Notebook | PT-BR | EN-US | Papel da evidência |
|---|---|---|---|
| 01. Framing e EDA | [PT-BR](notebooks/pt-BR/01_Framing_EDA_PT.ipynb) | [EN-US](notebooks/en-US/01_Framing_EDA_EN.ipynb) | Descritivo e checagens de identificação |
| 02. Baseline de propensão | [PT-BR](notebooks/pt-BR/02_Baseline_Propensity_PT.ipynb) | [EN-US](notebooks/en-US/02_Baseline_Propensity_EN.ipynb) | Baseline de desenvolvimento |
| 03. Meta-learners | [PT-BR](notebooks/pt-BR/03_Meta_Learners_PT.ipynb) | [EN-US](notebooks/en-US/03_Meta_Learners_EN.ipynb) | Desenvolvimento de candidatos |
| 04. Causal forest e uplift trees | [PT-BR](notebooks/pt-BR/04_Causal_Forest_Uplift_Trees_PT.ipynb) | [EN-US](notebooks/en-US/04_Causal_Forest_Uplift_Trees_EN.ipynb) | Desenvolvimento e pré-registro |
| 05. Avaliação no teste selado | [PT-BR](notebooks/pt-BR/05_Evaluation_Sealed_Test_PT.ipynb) | [EN-US](notebooks/en-US/05_Evaluation_Sealed_Test_EN.ipynb) | Confirmatório |
| 06. Heterogeneidade e uplift funnel | [PT-BR](notebooks/pt-BR/06_Heterogeneity_Uplift_Funnel_PT.ipynb) | [EN-US](notebooks/en-US/06_Heterogeneity_Uplift_Funnel_EN.ipynb) | Exploratório pós-confirmatório |
| 07. Policy learning e ROI | [PT-BR](notebooks/pt-BR/07_Policy_Learning_ROI_PT.ipynb) | [EN-US](notebooks/en-US/07_Policy_Learning_ROI_EN.ipynb) | Exploratório pós-confirmatório |
| 08. Robustez e limitações | [PT-BR](notebooks/pt-BR/08_Robustness_Limitations_PT.ipynb) | [EN-US](notebooks/en-US/08_Robustness_Limitations_EN.ipynb) | Síntese de evidência e fronteira decisória |

## Métodos

O conjunto de candidatos cobre abordagens distintas: um baseline de propensão treinado apenas em observações tratadas, os meta-learners S, T, X e R com pré-processamento compartilhado, `econml.CausalForestDML` para efeitos heterogêneos ortogonalizados e `causalml.UpliftTreeClassifier` para divisões diretas de uplift.

A avaliação reúne Qini AUC, uplift AUC, uplift@30%, holdouts repetidos, envelopes de rankings aleatórios e intervalos por bootstrap pareado. A análise de heterogeneidade utiliza perfis por quantil, resumos no estilo GATES, funil de uplift e correlações de ranking entre outcomes, enquanto a etapa de decisão cobre políticas binárias de contato, atribuição entre três braços, avaliação IPW, restrições de orçamento e cenários substituíveis de margem e custo.

## Achados pós-confirmatórios

As etapas exploratórias adicionam contexto sem alterar o resultado confirmatório. O modelo `X+Tree(depth=4)` produziu a maior Qini AUC absoluta no teste selado, mas sua diferença contra o baseline de propensão incluiu zero. Os rankings de visit e spend tiveram baixo alinhamento (Spearman 0,0941), indicando que o uplift de engajamento falha como proxy de uplift econômico.

Na maioria dos orçamentos, os intervalos das políticas incluíram zero, e os diferentes pontos de orçamento representam visões correlacionadas da mesma validação, sem independência experimental. Os valores de ROI correspondem a cenários pontuais baseados em spend como proxy de receita e em margens e custos ilustrativos, sem constituir um caso de negócio validado nem propagar a incerteza do efeito.

## Fronteira decisória

Os usos permitidos restringem-se a aprendizado, geração de hipóteses, desenho de piloto prospectivo randomizado e simulação ilustrativa com premissas substituíveis. As interpretações proibidas incluem alegar superioridade confirmatória de um candidato de uplift, fazer deployment automático da política aprendida, selecionar retrospectivamente o melhor modelo ou orçamento e tratar múltiplos orçamentos como testes independentes.

O próximo passo defensável consiste em executar um piloto randomizado com potência adequada, política congelada antes do lançamento, outcome primário pré-especificado, custos e margens reais e monitoramento de fadiga, entregabilidade, descadastro, reclamações e capacidade.

## Arquitetura de software

| Caminho | Responsabilidade |
|---|---|
| `src/data.py`, `src/features.py`, `src/splits.py` | Contrato do dataset, features, split determinístico e índices selados |
| `src/learners.py` | Baseline, meta-learners, causal forest e uplift tree |
| `src/evaluation.py` | Métricas de uplift, holdouts repetidos, bootstrap e envelopes aleatórios |
| `src/reports.py` | Heterogeneidade, perfis por quantil, funil e surrogate |
| `src/policy.py`, `src/policy_reports.py` | Políticas, avaliação IPW, budgets, ROI e apresentação |
| `src/robustness_reports.py` | Registro de evidências S9, limitações, fronteira e manifesto de fontes |
| `src/i18n.py`, `src/i18n_catalogs/` | Camada offline de apresentação PT-BR e EN-US |
| `tests/test_suite.py`, `tests/test_notebooks.py` | Testes de dados, leakage, modelos, métricas, artefatos, localização e paridade dos notebooks bilíngues |
| `artifacts/s6` a `artifacts/s9` | Resultado confirmatório congelado e evidências downstream somente de reporte |

## Preparação dos dados

O dataset de terceiros não é redistribuído porque a publicação original omite uma licença explícita de reutilização. As instruções de download, nome do arquivo e checksum constam em [`dataset/README.md`](dataset/README.md) e devem ser seguidas antes da execução dos notebooks ou testes.

## Instalação

O ambiente Conda espera este projeto e `ds_toolkit` como diretórios irmãos, seguindo o layout do monorepo do portfólio:

```text
projects-root/
├── Marketing_Promotion_Campaign_Uplift_Modelling/
└── ds_toolkit/
    └── multilang/
```

Crie o ambiente e registre o kernel:

```bash
mamba env create -f environment.yml
mamba activate uplift
python -m ipykernel install --user --name uplift --display-name "Python (uplift)"
```

O ambiente fixa OpenBLAS no Windows porque a resolução MKL testada para esta stack causou crashes nativos ao importar CausalML, EconML e SHAP.

## Validação

Execute a suíte completa de testes a partir da raiz:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
```

A suíte contém 157 testes de integridade de dados, reprodutibilidade dos splits, fronteiras de fit, hashes de artefatos, métricas, localização, relatórios, estrutura PT-BR/EN-US dos notebooks e igualdade do código executável entre os pares bilíngues.

Para revisão, o Notebook 05 e `artifacts/s6/` são tratados como evidência congelada. A reexecução dos notebooks downstream de reporte é suportada, enquanto um novo experimento no teste selado constituiria outro estudo.

## Referências

- Radcliffe, N. e Surry, P. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Künzel, S. R. et al. (2019), [Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning](https://doi.org/10.1073/pnas.1804597116).
- Nie, X. e Wager, S. (2021), [Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://doi.org/10.1093/biomet/asaa076).
- Athey, S., Tibshirani, J. e Wager, S. (2019), [Generalized Random Forests](https://doi.org/10.1214/18-AOS1709).
- Zhao, Y., Fang, X. e Simchi-Levi, D. (2017), [Uplift Modeling with Multiple Treatments and General Response Types](https://doi.org/10.1137/1.9781611974973.66).

## Autor e responsabilidade

Autor: Enio Rubens<br>
Papel: Data Science e Analytics

Assistentes de programação com IA apoiaram modularização, tradução, scaffolding de testes, documentação e revisão. O framing de negócio, as decisões metodológicas, a interpretação dos resultados e a aprovação final permaneceram sob condução humana, sendo o autor responsável pelas conclusões publicadas.

---

## Licença e agradecimentos

Código: distribuído sob a [Licença MIT](LICENSE).

Dataset: [Hillstrom MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html), publicado por Kevin Hillstrom. Os dados são obtidos diretamente da fonte e não são redistribuídos porque a publicação original omite uma licença explícita de reutilização; consulte [dataset/README.md](dataset/README.md).

Bibliotecas: comunidades de desenvolvimento do scikit-learn, CausalML, EconML, scikit-uplift, XGBoost, LightGBM, Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn, Plotly, Graphviz, Optuna, MLflow e pytest.

Pesquisa: trabalhos citados de Radcliffe e Surry; Künzel et al.; Nie e Wager; Athey, Tibshirani e Wager; e Zhao, Fang e Simchi-Levi.

---

Última atualização: agosto de 2026<br>
Status do projeto: completo e pronto para portfólio<br>
Resultado confirmatório: Δ Qini AUC -0,0088 | IC 95% [-0,0492; 0,0302] | superioridade não estabelecida<br>
Fronteira de decisão: não pronto para implantação direta

---

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![CausalML 0.15.5](https://img.shields.io/badge/CausalML-0.15.5-008B8B.svg)](https://causalml.readthedocs.io/)
[![EconML 0.16.0](https://img.shields.io/badge/EconML-0.16.0-5C2D91.svg)](https://www.pywhy.org/EconML/)
[![scikit-uplift](https://img.shields.io/badge/scikit--uplift-metrics-brightgreen.svg)](https://www.uplift-modeling.com/)
[![pytest](https://img.shields.io/badge/pytest-154%20passed-brightgreen.svg)](https://pytest.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Notebooks bilíngues](https://img.shields.io/badge/notebooks-PT--BR%20%7C%20EN--US-blueviolet.svg)](notebooks/README.md)

---

*Este projeto de portfólio implementa machine learning causal de ponta a ponta, avaliação confirmatória, análise de efeitos heterogêneos de tratamento e aprendizado de políticas com restrição de orçamento.*
