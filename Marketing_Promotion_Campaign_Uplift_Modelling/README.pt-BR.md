# Uplift Modeling para Targeting de Campanha: Hillstrom

**Idioma:** [English](README.md) | Português (Brasil)

Este projeto investiga **quem deve receber uma ação de marketing porque a ação
muda seu resultado**, em vez de apenas identificar quem tem maior probabilidade
de responder. O experimento randomizado Hillstrom MineThatData é usado para
comparar targeting por propensão, estimadores de CATE, uplift trees, causal
forests e políticas de contato sob restrição de orçamento.

A análise forma um fluxo modular, testado e bilíngue, com avaliação
pré-registrada em teste selado. Sua conclusão principal é deliberadamente
conservadora: a hipótese confirmatória primária não foi sustentada, e as
políticas resultantes **não estão prontas para deployment direto**.

## Resultado Executivo

A comparação primária pré-registrada foi a diferença de Qini AUC entre a
`UpliftTree` e o baseline de propensão no teste selado de 12.800 clientes.

| Evidência | Resultado | Interpretação |
|---|---:|---|
| `UpliftTree - baseline de propensão` | -0,0088; IC95% [-0,0492; 0,0302] | Sem vantagem confirmatória |
| Qini AUC absoluta da `X+Tree(depth=4)` | 0,0470; IC95% [0,0180; 0,0749] | Melhor ranking absoluto no teste selado, mas não é vencedor promovível retrospectivamente |
| Alinhamento dos rankings visit-spend | Spearman 0,0941 | Targeting para visitas não se transfere de forma confiável para gasto |
| Fronteira decisória final | `not_ready_for_direct_deployment` | Uso restrito a aprendizado e desenho de piloto prospectivo |

As etapas de heterogeneidade, política e ROI são exploratórias e
pós-confirmatórias. Elas preservam o resultado primário nulo e não reabrem a
seleção de modelos.

## Problema de Negócio

Um modelo de propensão ordena clientes pela probabilidade de um outcome
observado. Esse ranking pode consumir orçamento com clientes que responderiam
mesmo sem e-mail. Uplift modeling estima o efeito condicional do tratamento:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]
$$

A pergunta operacional passa a ser: **quais clientes devem ser contatados, com
qual campanha, sob um orçamento de contato e premissas econômicas explícitas?**

## Dataset e Outcomes

O [Hillstrom MineThatData E-Mail Analytics Challenge](http://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
contém 64.000 clientes randomizados em três braços.

| Braço | Clientes | Descrição |
|---|---:|---|
| No E-Mail | 21.306 | Controle |
| Mens E-Mail | 21.307 | Campanha de produtos masculinos |
| Womens E-Mail | 21.387 | Campanha de produtos femininos |

O dataset tem oito características pré-tratamento, uma coluna de tratamento e
três outcomes:

| Outcome | Papel | Prevalência / escala observada |
|---|---|---|
| `visit` | Primário | 14,7% visitaram o site |
| `conversion` | Secundário | 0,9% compraram; apenas 578 eventos |
| `spend` | Secundário | Proxy de receita com grande massa em zero |

`visit` é o outcome primário porque `conversion` é raro demais para estimação
estável de CATE em subgrupos neste tamanho amostral. O projeto testa
explicitamente se um ranking otimizado para engajamento de topo de funil se
transfere para conversão ou receita.

## Contrato Experimental

- O RCT original com três braços sustenta a identificação causal sob as
  premissas de randomização verificadas no Notebook 01.
- O desenvolvimento usa divisão determinística 60%/20%/20% em treino,
  validação e teste selado, estratificada para preservar tratamento e outcome.
- O desenvolvimento binário agrega os dois braços de e-mail contra
  `No E-Mail`.
- O Notebook 05 avalia candidatos congelados uma única vez no teste selado,
  com 2.000 repetições de bootstrap.
- S7-S9 consomem saídas persistidas. Não reajustam candidatos selados, não
  repetem as predições seladas e não escolhem champion retrospectivo.
- A interpretação usa perfis por quantil e um surrogate raso para comunicação;
  não usa SHAP nem reivindica explicação causal individual.

## Fluxo de Notebooks

Oito pares de notebooks implementam nove seções analíticas. PT-BR é a edição
canônica; EN-US é editorialmente equivalente e usa o mesmo código e estado
analítico persistido.

| Notebook | PT-BR | EN-US | Papel da evidência |
|---|---|---|---|
| 01 - Framing e EDA | [PT-BR](notebooks/pt-BR/01_Framing_EDA_PT.ipynb) | [EN-US](notebooks/en-US/01_Framing_EDA_EN.ipynb) | Descritivo e checagens de identificação |
| 02 - Baseline de propensão | [PT-BR](notebooks/pt-BR/02_Baseline_Propensity_PT.ipynb) | [EN-US](notebooks/en-US/02_Baseline_Propensity_EN.ipynb) | Baseline de desenvolvimento |
| 03 - Meta-learners | [PT-BR](notebooks/pt-BR/03_Meta_Learners_PT.ipynb) | [EN-US](notebooks/en-US/03_Meta_Learners_EN.ipynb) | Desenvolvimento de candidatos |
| 04 - Causal forest e uplift trees | [PT-BR](notebooks/pt-BR/04_Causal_Forest_Uplift_Trees_PT.ipynb) | [EN-US](notebooks/en-US/04_Causal_Forest_Uplift_Trees_EN.ipynb) | Desenvolvimento e pré-registro |
| 05 - Avaliação no teste selado | [PT-BR](notebooks/pt-BR/05_Evaluation_Sealed_Test_PT.ipynb) | [EN-US](notebooks/en-US/05_Evaluation_Sealed_Test_EN.ipynb) | Confirmatório |
| 06 - Heterogeneidade e uplift funnel | [PT-BR](notebooks/pt-BR/06_Heterogeneity_Uplift_Funnel_PT.ipynb) | [EN-US](notebooks/en-US/06_Heterogeneity_Uplift_Funnel_EN.ipynb) | Exploratório pós-confirmatório |
| 07 - Policy learning e ROI | [PT-BR](notebooks/pt-BR/07_Policy_Learning_ROI_PT.ipynb) | [EN-US](notebooks/en-US/07_Policy_Learning_ROI_EN.ipynb) | Exploratório pós-confirmatório |
| 08 - Robustez e limitações | [PT-BR](notebooks/pt-BR/08_Robustness_Limitations_PT.ipynb) | [EN-US](notebooks/en-US/08_Robustness_Limitations_EN.ipynb) | Síntese de evidência e fronteira decisória |

## Métodos

O conjunto de candidatos cobre abordagens deliberadamente distintas:

- Baseline de propensão treinado apenas em observações tratadas.
- Meta-learners S, T, X e R com pré-processamento compartilhado.
- `econml.CausalForestDML` para efeitos heterogêneos ortogonalizados.
- `causalml.UpliftTreeClassifier` para splits diretos de uplift.
- Qini AUC, uplift AUC, uplift@30%, holdouts repetidos, envelopes de rankings
  aleatórios e intervalos por bootstrap pareado.
- Perfis por quantil, resumos no estilo GATES, uplift funnel e correlações de
  ranking entre outcomes.
- Políticas binárias de contato, atribuição entre três braços, avaliação IPW,
  restrições de orçamento e cenários substituíveis de margem/custo.

## Achados Pós-Confirmatórios

As etapas exploratórias adicionam contexto sem mudar o resultado confirmatório:

- `X+Tree(depth=4)` produziu a maior Qini AUC absoluta no teste selado, mas sua
  diferença contra o baseline de propensão ainda incluiu zero.
- Os rankings de visit e spend tiveram baixo alinhamento (Spearman 0,0941),
  portanto uplift de engajamento não é proxy confiável de uplift econômico.
- Na maioria dos budgets, os intervalos das políticas incluíram zero. Pontos de
  orçamento são visões correlacionadas da mesma validação, não experimentos
  independentes.
- Os valores de ROI são cenários pontuais baseados em spend como proxy de
  receita e em margens e custos ilustrativos. Não são business case validado e
  não propagam a incerteza do efeito.

## Fronteira Decisória

Usos permitidos:

- Aprendizado e geração de hipóteses.
- Desenho de piloto prospectivo randomizado.
- Simulação ilustrativa com premissas visivelmente substituíveis.

Interpretações proibidas:

- Alegar superioridade confirmatória de um candidato de uplift.
- Fazer deployment automático da política aprendida.
- Selecionar retrospectivamente o melhor modelo ou budget.
- Tratar múltiplos budgets como testes independentes.

O próximo passo defensável é um piloto randomizado com potência adequada,
política congelada antes do lançamento, outcome primário pré-especificado,
custos e margens reais e monitoramento de fadiga, entregabilidade, descadastro,
reclamações e capacidade.

## Arquitetura de Software

| Caminho | Responsabilidade |
|---|---|
| `src/data.py`, `src/features.py`, `src/splits.py` | Contrato do dataset, features, split determinístico e índices selados |
| `src/learners.py` | Baseline, meta-learners, causal forest e uplift tree |
| `src/evaluation.py` | Métricas de uplift, holdouts repetidos, bootstrap e envelopes aleatórios |
| `src/reports.py` | Heterogeneidade, perfis por quantil, funil e surrogate |
| `src/policy.py`, `src/policy_reports.py` | Políticas, avaliação IPW, budgets, ROI e apresentação |
| `src/robustness_reports.py` | Registro de evidências S9, limitações, fronteira e manifesto de fontes |
| `src/i18n.py`, `src/i18n_catalogs/` | Camada offline de apresentação PT-BR/EN-US |
| `tests/test_suite.py` | Testes de dados, leakage, modelos, métricas, artefatos, localização e notebooks |
| `artifacts/s6` a `artifacts/s9` | Resultado confirmatório congelado e evidências downstream somente de reporte |

## Preparação dos Dados

O dataset de terceiros não é redistribuído porque a publicação original não
declara uma licença explícita de reutilização. Siga as instruções de download,
nome do arquivo e checksum em [`dataset/README.md`](dataset/README.md) antes de
executar os notebooks ou testes.

## Instalação

O ambiente Conda espera este projeto e `ds_toolkit` como diretórios irmãos,
seguindo o layout do monorepo do portfólio:

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

O ambiente fixa OpenBLAS no Windows porque a resolução MKL testada para esta
stack causou crashes nativos ao importar CausalML, EconML e SHAP.

## Validação

Execute o módulo completo de testes a partir da raiz:

```bash
python -m pytest tests/test_suite.py -q -p no:cacheprovider
python -m ruff check src tests
```

A suíte contém 154 testes de integridade de dados, reprodutibilidade dos
splits, fronteiras de fit, hashes de artefatos, métricas, localização,
relatórios e estrutura PT-BR/EN-US dos notebooks.

Para revisão, trate o Notebook 05 e `artifacts/s6/` como evidência congelada.
Reexecutar os notebooks downstream de reporte é suportado; abrir um novo
experimento no teste selado constituiria outro estudo.

## Referências Científicas

- Radcliffe, N. e Surry, P. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Künzel, S. R. et al. (2019), [Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning](https://doi.org/10.1073/pnas.1804597116).
- Nie, X. e Wager, S. (2021), [Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://doi.org/10.1093/biomet/asaa076).
- Athey, S., Tibshirani, J. e Wager, S. (2019), [Generalized Random Forests](https://doi.org/10.1214/18-AOS1709).
- Zhao, Y., Fang, X. e Simchi-Levi, D. (2017), [Uplift Modeling with Multiple Treatments and General Response Types](https://doi.org/10.1137/1.9781611974973.66).

## Licença

O código e a documentação do projeto estão disponíveis sob a
[Licença MIT](LICENSE). A licença não abrange o dataset Hillstrom de terceiros.

## Autor e Responsabilidade

**Autor:** Enio Rubens<br>
**Papel:** Data Science e Analytics

Assistentes de programação com IA apoiaram modularização, tradução, scaffolding
de testes, documentação e revisão. Framing de negócio, decisões metodológicas,
interpretação dos resultados e aprovação final permaneceram sob condução
humana. A responsabilidade pelas conclusões publicadas é do autor.

## Atribuição dos Dados

O dataset foi disponibilizado por Kevin Hillstrom para o MineThatData E-Mail
Analytics Challenge de 2008. Ele é obtido diretamente da fonte e não é
redistribuído aqui; a titularidade e os termos originais de distribuição
permanecem com a fonte correspondente.
