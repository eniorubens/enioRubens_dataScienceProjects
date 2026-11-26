# PROJECT BRIEF — Retail Pricing & Promotion com Modelos Multinível

> Brief de handoff escrito em 2026-11-23 para uma nova sessão do Claude Code aberta nesta pasta.
> Leia inteiro antes de agir. Fale com o usuário em **pt-BR**.

---

## 1. Contexto e origem

- Este é o próximo projeto do portfólio de Data Science do Enio Rubens. O projeto anterior concluído foi o **E-commerce Delivery Network & SLA Risk** (`D:\Projetos\E-commerce-Delivery-Network-&-SLA-Risk`).
- **Tema escolhido pelo usuário:** *Modelos Multinível de Regressão para Dados em Painel*.
- **Técnicas a demonstrar:**
  - Modelos hierárquicos de dois e três níveis: modelo nulo, ICC, intercepto e inclinação aleatórios, interação entre níveis e previsão somando os BLUPs.
  - Medidas repetidas em três níveis com `statsmodels.MixedLM` (`groups` + `re_formula` + `vc_formula`) e com lme4.
  - **Step-up strategy**, testes de razão de verossimilhança (LR) e significância das variâncias.
  - **Especificação correta de inclinações aleatórias em três níveis.** Com `groups='g'` e `re_formula='x'`, `vc_formula={'h': '0 + x'}` não cria inclinação por nível de `h`: duplica a inclinação de `g` e não inclui intercepto por `h`, o que gera componentes não identificáveis. A forma correta é `vc_formula={'h': '0 + C(h)', 'h_x': '0 + C(h):x'}`, ou lme4 `(1 + x | g) + (1 + x | g:h)`. Mostrar essa especificação corretamente é um diferencial. *(Corrigido em 2026-11-24; ver o registro de decisões.)*
- **Decisões do usuário até aqui:**
  - Recusou Ideb, PISA e Rossmann por estarem "fora de negócio" ou serem "rasos".
  - Escolheu **pricing e promoção no varejo**.
  - O **DriftGuard foi descartado** por inviabilidade computacional. Não recomendar nada que exija dataset massivo.
- **Máquina:** i9, RTX 3050, **16 GB de RAM**. Tudo precisa rodar localmente, sem nuvem.

## 2. Pergunta de negócio

> Quanto o **preço** e a **promoção** (encarte, display e desconto temporário, o TPR) movem a venda **em cada loja e produto**? Onde vale a pena baixar preço, subir preço ou promover?

A narrativa central é o **encolhimento (partial pooling)**. A elasticidade estimada loja a loja por OLS é ruidosa, e às vezes positiva, o que não faz sentido econômico. O modelo multinível estabiliza essas estimativas usando a informação do conjunto. O projeto deve demonstrar isso com evidência, inclusive fora da amostra.

## 3. Dados

- **Fonte:** dunnhumby Source Files, *Breakfast at the Frat: A Time Series Analysis*. O usuário baixou com cadastro e aceitou os termos.
- **Arquivos em `dataset/`:**
  - `dunnhumby - Breakfast at the Frat.xlsx` (31,6 MB)
  - `dunnhumby - Breakfast at the Frat User Guide.pdf`
  - o `.zip` original
- **Abas do Excel** (o cabeçalho está na **linha 2**, use `header=1`; a linha 1 é o título):
  - `Glossary`
  - `dh Store Lookup`: 79 linhas. Colunas: STORE_ID, STORE_NAME, ADDRESS_CITY_NAME, ADDRESS_STATE_PROV_CODE, MSA_CODE, SEG_VALUE_NAME, PARKING_SPACE_QTY, SALES_AREA_SIZE_NUM, AVG_WEEKLY_BASKETS.
  - `dh Products Lookup`: 58 linhas. Colunas: UPC, DESCRIPTION, MANUFACTURER, CATEGORY, SUB_CATEGORY, PRODUCT_SIZE.
  - `dh Transaction Data`: 524.950 linhas. Colunas: WEEK_END_DATE, STORE_NUM, UPC, UNITS, VISITS, HHS, SPEND, PRICE, BASE_PRICE, FEATURE, DISPLAY, TPR_ONLY.
- **Licença:** **não versionar os dados brutos.** Coloque `dataset/` e `data/raw/` no `.gitignore`. O README deve ter a seção "Como obter os dados", com link para https://www.dunnhumby.com/source-files/ e crédito "Data: dunnhumby Source Files – Breakfast at the Frat". Publique apenas código, figuras e resultados agregados.

### Perfil já levantado (2026-11-23)

| Item | Valor |
|---|---|
| Lojas / UPCs / semanas na transação | 77 / 55 / 156 (2009-01-14 a 2012-01-04) |
| Combinações loja×UPC | 3.909 de 4.235; mediana de 149 semanas; só 1.225 com as 156 semanas |
| Chaves duplicadas (semana, loja, UPC) | 0 |
| Geografia | TX 41, OH 31, KY 4, IN 1 lojas; 9 MSAs |
| Segmentos | Mainstream, Value, Upscale |
| Produtos | 4 categorias (Bag Snacks/Pretzels, Oral Hygiene/Mouthwash, Cold Cereal, Frozen Pizza); 15 fabricantes, incluindo Private Label |
| Promoções | feature 8,4%, display 11,0%, TPR 13,5%; alguma promoção em 28,5%; preço < preço base em 25% |
| Variação de preço | DP de log(preço) dentro de loja×UPC: mediana 0,11 |
| Variância de log(units) | entre UPC ≈ 52%, entre loja ≈ 7%, entre loja×UPC ≈ 72% (decomposição simples, sem modelo) |
| OLS por loja×UPC (≥ 52 semanas, controlando feature e display) | n = 3.705; mediana −1,72; amplitude de −6,3 a +3,1; **4,5% positivas** |
| OLS agrupado com efeito fixo de UPC | elasticidade −1,08; feature +0,49; display +0,61; TPR ≈ 0 (em log) |

### Problemas de qualidade a tratar e documentar

1. **IDs de loja duplicados no cadastro.** As lojas 4503 (Rockwall) e 17627 (Flower Mound) aparecem como MAINSTREAM e UPSCALE. Defina a regra, por exemplo um segmento dominante ou uma flag, e documente.
2. **Semanas sem venda não existem como linha.** Só há 5 linhas com UNITS = 0. Uma linha ausente pode ser venda zero ou ruptura de estoque, e isso gera um viés de seleção a discutir. Decida entre modelar só as semanas observadas e argumentar, ou fazer uma análise de sensibilidade.
3. **Valores nulos ou zerados:** PRICE nulo em 23 linhas, BASE_PRICE nulo em 185, PRICE = 0 em 1.
4. **Outliers, já avisados no guia da dunnhumby:** units/visit > 5 em 67 linhas; visits/hhs > 3 em 293. Também há vendas anormalmente baixas, que podem indicar ruptura.
5. **PARKING_SPACE_QTY** tem muitos nulos. Use com cuidado ou descarte.
6. **Endogeneidade de preço:** o preço é definido pelo varejista e acompanha as promoções. Controle BASE_PRICE e as flags, e discuta os limites da leitura causal.

## 4. Plano metodológico (proposta, a validar com o usuário)

**Resposta:** `log(UNITS)`. Considere também UNITS como contagem (Poisson ou binomial negativa GLMM), como checagem da transformação log.

**Step-up:**

1. **Modelo nulo:** ICC por loja, por UPC e por loja×UPC. Teste LR contra OLS.
2. **Efeitos fixos:** log(preço) ou desconto relativo, FEATURE, DISPLAY, TPR_ONLY, sazonalidade (semana do ano ou feriados) e tendência.
3. **Intercepto aleatório:** loja e produto **cruzados**, mais loja×UPC.
4. **Inclinações aleatórias:** elasticidade e efeito de promoção por loja e por UPC. Teste LR para cada variância.
5. **Interações entre níveis:** segmento da loja (Value/Mainstream/Upscale), tamanho, AVG_WEEKLY_BASKETS e MSA/estado × preço; categoria e fabricante (incluindo Private Label) × preço.
6. **Estrutura de covariância:** termos aleatórios correlacionados contra independentes, com teste LR.

**Comparações e validação:**

- Compare três abordagens: **OLS separado por loja×UPC**, **OLS agrupado** e **modelo multinível (BLUP)**. Mostre o encolhimento graficamente (funil ou caterpillar).
- **Validação temporal:** treine nas semanas iniciais e teste nas últimas 26 a 52. O multinível precisa prever melhor que o OLS separado, especialmente nas combinações com poucas semanas.
- **Opcional:** GPBoost ou LightGBM como desafiantes de ML (`pip install gpboost`, roda em CPU).

**Camada de decisão:**

- Elasticidade e lift de promoção por loja×produto, com intervalos.
- Recomendações de onde baixar preço, onde subir e quais promoções têm ROI. Uma simulação de calendário de promoções sob orçamento é opcional.
- Use margem hipotética e deixe isso explícito como premissa.

**Motor de estimação:**

- O `statsmodels.MixedLM` lida mal com efeitos cruzados grandes. A forma usual é um único grupo com `vc_formula`, que pode ficar lento ou estourar memória em cerca de 520 mil linhas.
- Avalie, nesta ordem:
  - (a) `lme4` / `glmmTMB` via R, usando pymer4 ou rpy2;
  - (b) `statsmodels` em subamostras para validar;
  - (c) bambi/PyMC, se fizer sentido.
- Confirme com o usuário antes de exigir instalação de R.

## 5. Padrões do portfólio (obrigatórios)

- **Estrutura, como nos outros projetos** (veja `D:\Projetos\E-commerce-Delivery-Network-&-SLA-Risk` como referência mais recente):
  - `src/` (pacote), `tests/` (pytest), `notebooks/`, `scripts/`, `docs/`, `site/` (relatório executivo HTML), `reports/`/`outputs/`;
  - `environment.yml` (conda-forge, `PYTHONNOUSERSITE: "1"`), `pyproject.toml`;
  - `README.md` (EN) + `README.pt-BR.md`, LICENSE MIT.
- **Bilíngue EN/PT:** os notebooks EN e PT executam **código idêntico**. Só a linha `LANG` muda. Textos, títulos e rótulos vêm de um módulo i18n em `src/`. Cálculo e desenho de gráfico também ficam em `src/`. Inclua um teste que compare as células de código dos dois notebooks.
- **Fonte de verdade:** `D:\Projetos\<projeto>`. O clone de publicação `D:\Projetos\_publish_enioRubens_dataScienceProjects` (repo `eniorubens/enioRubens_dataScienceProjects`) é só cópia. Toda mudança é feita aqui primeiro e depois copiada.
- **MLflow:** um servidor por projeto, iniciado da pasta do projeto (`start_mlflow.bat`), se for usado.
- **Git:** commit, push e merge só com OK explícito; merge de PR com `--delete-branch`.
- **Rigor:** cada afirmação deve ter evidência, e as limitações devem ser explícitas (veja os READMEs dos outros projetos).
- **Wiki Obsidian:** `D:\Projetos\Obsidian` (schema em `Obsidian/CLAUDE.md`). Ao final, ingira o projeto na wiki. A pasta `wiki/datasets/` guarda páginas de **projetos**, devido ao mapeamento invertido descrito no schema.

## 6. Ambiente

- Não há `conda` no PATH do PowerShell. O Miniforge fica em `C:\Users\enior\miniforge3`, com os ambientes em `C:\Users\enior\miniforge3\envs\`.
- O perfil inicial foi feito num ambiente conda anterior, com pandas 2.3.3, statsmodels 0.14.2 e openpyxl, sem pymer4.
- **Crie um ambiente próprio**, por exemplo `retail-multilevel`, com `environment.yml` neste projeto.
- Para ler o Excel uma vez e trabalhar em Parquet: `pd.read_excel(..., sheet_name=..., header=1)`. A leitura leva cerca de 1 minuto. Grave em `data/interim/*.parquet`, que também fica no `.gitignore`.

## 7. Primeiros passos sugeridos para a nova sessão

1. Confirmar com o usuário o escopo (seção 4) e o motor de estimação (R/lme4 ou só Python).
2. Criar a estrutura de pastas, `environment.yml`, `pyproject.toml`, `.gitignore` e o esqueleto de README.
3. Fazer a ingestão: Excel → Parquet, limpeza com as regras da seção 3 e testes de qualidade de dados.
4. Fazer a EDA e o modelo nulo (ICCs), e depois seguir o step-up.
