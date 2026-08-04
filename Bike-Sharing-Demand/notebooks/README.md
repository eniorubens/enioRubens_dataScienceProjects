# Notebooks por idioma

O português brasileiro é o idioma canônico deste projeto. A edição EN-US é uma
tradução editorial equivalente: código, metodologia, referências e resultados
numéricos devem permanecer alinhados entre os idiomas.

| Etapa | PT-BR | EN-US |
|---|---|---|
| 01 — EDA 2015–2024 | [PT-BR](pt-BR/01_Seoul_Bike_2015-2024_EDA.ipynb) | [EN-US](en-US/01_Seoul_Bike_2015-2024_EDA.ipynb) |
| 02 — Análise multivariada | [PT-BR](pt-BR/02_Seoul_Bike_Multivariate_Analysis.ipynb) | [EN-US](en-US/02_Seoul_Bike_Multivariate_Analysis.ipynb) |
| 03 — Engenharia de atributos | [PT-BR](pt-BR/03_Feature_Engineering_EDA.ipynb) | [EN-US](en-US/03_Feature_Engineering_EDA.ipynb) |
| 04 — Seleção de modelos | [PT-BR](pt-BR/04_Seoul_Bike_Model_Selection.ipynb) | [EN-US](en-US/04_Seoul_Bike_Model_Selection.ipynb) |
| 05 — Validação final | [PT-BR](pt-BR/05_Seoul_Bike_Final_Validation.ipynb) | [EN-US](en-US/05_Seoul_Bike_Final_Validation.ipynb) |
| 06 — Resíduos e incerteza | [PT-BR](pt-BR/06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb) | [EN-US](en-US/06_Seoul_Bike_Residual_and_Uncertainty_Experiments.ipynb) |
| 07 — Calibração conformal | [PT-BR](pt-BR/07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb) | [EN-US](en-US/07_Seoul_Bike_Temporal_Conformal_Calibration.ipynb) |
| 08 — Demonstração operacional | [PT-BR](pt-BR/08_Seoul_Bike_Operational_Forecast_Demo.ipynb) | [EN-US](en-US/08_Seoul_Bike_Operational_Forecast_Demo.ipynb) |

## Contrato de equivalência

As duas edições devem preservar a mesma ordem e os mesmos IDs de células. As
células Python devem ser idênticas, exceto pela seleção explícita do idioma via
`make_lang`. Variáveis, colunas, parâmetros, artefatos e títulos originais de
referências bibliográficas não são traduzidos.

As células Markdown são traduzidas e revisadas separadamente. Gráficos,
tabelas, mensagens e relatórios são renderizados pelo módulo `multilang`, sem
repetir otimizações, treinamentos ou a abertura do holdout selado.
