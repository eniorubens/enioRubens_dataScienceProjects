# Previsão horária da demanda de bicicletas compartilhadas em Seul

Neste projeto, a demanda horária do sistema público de bicicletas de Seul é
modelada a partir de aproximadamente nove anos de observações (2015–2024),
enriquecidas com variáveis meteorológicas. O objetivo é estimar a demanda em
condições operacionais normais por meio de uma arquitetura modular, testável e
auditável, desde a análise exploratória até uma demonstração de decisão
operacional com incerteza.

Esta é a edição canônica em Português do Brasil. Os identificadores internos
permanecem em inglês e os textos exibidos são localizados pela camada
`multilang`. Uma edição dos notebooks em inglês será preparada posteriormente.

## Decisão central de modelagem

Nos dados de 2020, foi identificada uma ruptura extraordinária de mobilidade que
não pode ser explicada pelas variáveis meteorológicas disponíveis. Como esse
choque não pertence ao objetivo de previsão em condições operacionais normais,
o ano civil de 2020 foi excluído do ajuste e da métrica primária de seleção.

O dataset de origem não foi truncado. A exclusão foi representada por uma
máscara auditável do regime `normal_operations`, enquanto o ano meteorológico de
2020 foi preservado como diagnóstico de estresse. Desse modo, a anomalia foi
mantida visível sem que dominasse a seleção do modelo.

## Desenho da validação temporal

A validação cruzada utiliza folds expansíveis por ano meteorológico, de dezembro
a novembro, preservando o inverno dentro da mesma janela. A seleção primária foi
calculada nos anos meteorológicos de 2019, 2021, 2022 e 2023, com maior peso para
os folds recentes. O fold de 2020 foi utilizado somente como estresse.

O holdout final cobre dezembro de 2023 a novembro de 2024, com 8.784 observações
horárias. Ele foi materializado por uma única função auditada e aberto uma vez
no Notebook 05. Dezembro de 2024 foi descartado para que não influenciasse o
desenvolvimento nem a confirmação final.

## Fluxo dos notebooks

O projeto é composto por oito notebooks:

| Notebook | Finalidade | Natureza |
|---|---|---|
| [01](notebooks/01_Seoul_Bike_2015-2024_EDA.ipynb) | EDA do período 2015–2024 e análise de outliers por ano e estação | Descritiva |
| [02](notebooks/02_Seoul_Bike_Multivariate_Analysis.ipynb) | Des-tendenciamento, PhiK, VIF, testes de hipótese e seleção de atributos | Descritiva |
| [03](notebooks/03_Feature_Engineering_EDA.ipynb) | Engenharia de atributos e EDA das variáveis criadas | Descritiva |
| [04](notebooks/04_Seoul_Bike_Model_Selection.ipynb) | Seleção de modelos sob CV temporal e pipeline dinâmico | Seleção |
| [05](notebooks/05_Seoul_Bike_Final_Validation.ipynb) | Validação única no holdout selado, resíduos e SHAP | Confirmatória |
| [06](notebooks/06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb) | Persistência dos resíduos e experimentos de incerteza nos folds de desenvolvimento | Experimental |
| [07](notebooks/07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb) | Calibração conformal temporal da camada de incerteza | Experimental |
| [08](notebooks/08_Seoul_Bike_Operational_Forecast_Demo.ipynb) | Replay operacional com previsão, intervalo, capacidade e decisão | Demonstração |

Os Notebooks 06 e 07 não constituem uma segunda validação final. O holdout não
foi reaberto, nenhum candidato pontual foi reajustado e o Champion confirmado
não foi substituído. O Notebook 08 utiliza uma observação OOF histórica para
demonstrar o sistema sem convertê-la em nova evidência confirmatória.

## Arquitetura dinâmica e Champion confirmado

No Notebook 04, famílias distintas de estimadores são otimizadas sob a mesma
geometria temporal. Em cada trial, pré-processamento, encoding, seleção de
atributos, estratégia do alvo e hiperparâmetros do estimator são escolhidos em
conjunto dentro do pipeline. Essa composição dinâmica é a arquitetura central
do projeto.

O `CatBoostRegressor` foi pré-registrado como Champion antes da abertura do
holdout. No Notebook 05, ele foi comparado com dois challengers congelados por
meio de uma regra previamente declarada. O Champion seria confirmado se seu MAE
fosse no máximo 1,05 vez o melhor MAE e seu R² não ficasse mais de 0,02 abaixo do
melhor R².

| Métrica no holdout (n=8.784) | CatBoost Champion | HistGradientBoosting | Random Forest |
|---|---:|---:|---:|
| MAE | 1.118,1 | 1.424,9 | 1.593,4 |
| RMSE | 1.605,7 | 2.069,7 | 2.322,2 |
| R² | 0,839 | 0,733 | 0,664 |
| WAPE | 23,1% | 29,4% | 32,9% |
| Erro absoluto mediano | 768,3 | 893,4 | 988,0 |

O CatBoost foi simultaneamente o candidato pré-registrado e o melhor modelo
empírico no holdout selado.

## Experimentos de resíduos e incerteza

No Notebook 06, foram comparados o baseline reproduzível E0, interações
temporais e meteorológicas, `RMSEWithUncertainty` e uma camada prequential de
escala residual. Não foi identificado sucessor pontual: o E0 manteve MAE
ponderado de 840,165, R² ponderado de 0,857 e R² médio de 0,839. O E4 foi mantido
somente como produtor experimental de escala, com cobertura de 81,007%, largura
média de 4.501,5 bicicletas/hora e Winkler de 6.530,4 na meta de 90%.

No Notebook 07, essa escala foi recalibrada sem reajustar o estimador pontual. O
calibrador adaptativo normalizado `U4b_g0p01` foi selecionado como candidato
experimental para a camada de incerteza.

| Resultado para cobertura nominal de 90% | E4 | U4b_g0p01 | Ganho |
|---|---:|---:|---:|
| Cobertura observada | 81,007% | 90,146% | +9,139 p.p. |
| Distância absoluta até a meta | 8,993 p.p. | 0,146 p.p. | −98,4% |
| Largura média | 4.501,5 | 3.351,0 | −25,6% |
| Winkler | 6.530,4 | 4.836,9 | −25,9% |

Nos quatro folds de seleção, a cobertura variou de 90,024% a 90,242%. O
bootstrap temporal de 500 repetições produziu intervalo de 89,709% a 90,591%,
incluindo a meta nominal. O ganho foi restrito aos intervalos; MAE, RMSE e R² do
E0 não foram alterados.

Ainda permaneceram alertas de cobertura condicional no pico da manhã, no pico
da tarde, na demanda muito alta e às sextas-feiras às 18h. Assim, o U4b foi
classificado como candidato experimental, não como componente pronto para
produção.

## Demonstração operacional

No Notebook 08, foi simulada uma decisão antes da revelação da demanda real. Na
observação sorteada de 22/10/2022 às 07h, foram produzidos:

| Elemento | Resultado |
|---|---:|
| Previsão pontual E0 | 1.384 aluguéis |
| Intervalo U4b de 90% | 0 a 6.732 |
| Capacidade simulada | 4.000 aluguéis/hora |
| Decisão prévia | Zona de atenção |
| Reserva até o limite superior | 2.732 aluguéis/hora |
| Demanda revelada | 2.469 aluguéis |
| Erro absoluto do ponto | 1.085 aluguéis |

A demanda real permaneceu dentro do intervalo e abaixo da capacidade simulada.
Foi demonstrado, portanto, como a previsão central e a incerteza podem ser
convertidas em uma decisão explícita, sem alegar que o exemplo substitui uma
validação online.

## Estrutura do projeto

| Caminho | Papel |
|---|---|
| `src/` | Dados, EDA, engenharia de atributos, CV, pipelines, tracking, validação, incerteza e relatórios |
| `notebooks/` | Fluxo analítico de oito etapas em PT-BR |
| `tests/` | Testes de leakage, splits temporais, pipelines, relatórios e estrutura dos notebooks |
| `dataset/` | Dataset público, atribuição e artefatos locais ignorados pelo Git |
| `mlruns/` | Tracking local do MLflow, gerado durante os experimentos e não versionado |
| `environment.yml` / `requirements.txt` | Dependências testadas e fixadas |

## Convenções

O viés é definido como `mean(y_pred - y_true)`; valores positivos indicam
superestimação. O resíduo utilizado nos diagnósticos é `y_true - y_pred`;
valores positivos indicam subestimação.

Variáveis, colunas e identificadores Python permanecem em inglês. Narrativas,
títulos, `print()` e tabelas exibidas são apresentados em PT-BR por meio do
módulo `multilang`.

## Instalação e execução

Este projeto é publicado dentro da estrutura:

```text
enioRubens_dataScienceProjects/
├── Bike-Sharing-Demand/
└── ds_toolkit/
    └── multilang/
```

O caminho relativo é utilizado porque o `multilang`, já publicado no mesmo
monorepositório, permanece como dependência irmã deste projeto.

```bash
cd Bike-Sharing-Demand
conda env create -f environment.yml
conda activate Bike-Sharing
python -m ipykernel install --user --name bike-sharing --display-name "Python (Bike-Sharing)"
python -m pytest -q
```

Para reproduzir todo o fluxo, os notebooks devem ser executados em ordem. Os
artefatos volumosos e específicos de runtime não são versionados: o Notebook 06
consome os candidatos congelados pelo Notebook 04; o Notebook 07 consome as
previsões OOF do Notebook 06; e o Notebook 08 consome o manifesto e as previsões
conformes produzidos pelo Notebook 07.

No Windows, o servidor local do MLflow pode ser iniciado por:

```bat
start_mlflow.bat
```

## Limitações e próximos passos

As métricas finais provêm de um único holdout de doze meses. Qualquer troca do
Champion exigiria uma nova janela temporal independente. A camada conformal
também requer melhor cobertura nos regimes de alta demanda antes de ser
promovida para produção.

Os próximos passos previstos são a calibração condicional por regimes, o
congelamento conjunto do E0, da escala E4 e do estado U4b para inferência online,
a incorporação de uma nova janela pública e a edição integral em inglês.

## Dados e licenças

O código-fonte é disponibilizado sob a licença MIT; consulte [LICENSE](LICENSE).
O dataset possui atribuição própria sob KOGL Type 1, descrita em
[dataset/DATASET_README.md](dataset/DATASET_README.md). A licença do código não
substitui nem amplia a licença dos dados.
