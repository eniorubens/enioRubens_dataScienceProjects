# NASA C-MAPSS — previsão de vida útil restante — FD001

[English](README.en.md) · [Relatório HTML em português](site/index.html) · [Síntese técnica](notebooks/pt-BR/27_conclusao_entrega_fd001.ipynb)

[Abrir relatório publicado](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/) · [English online](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/en.html)

O projeto foi desenvolvido para estimar a vida útil restante de motores simulados, expressa em ciclos de operação. A etapa FD001 foi concluída com um Extra Trees e uma representação temporal construída exclusivamente a partir do ciclo atual e dos anteriores. O objetivo desta entrega é permitir a reprodução da análise e a execução local da previsão pontual.

## Fonte dos dados

O conjunto utilizado é o **Turbofan Engine Degradation Simulation Data Set**, disponibilizado pelo Prognostics Center of Excellence (PCoE) do NASA Ames Research Center (Saxena e Goebel, 2008). NASA é a sigla de **National Aeronautics and Space Administration**, a agência espacial e aeronáutica dos Estados Unidos. No título deste projeto, ela identifica a origem dos dados; a análise e o modelo são trabalhos autorais de portfólio. [Repositório oficial da NASA](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

As séries foram geradas com o simulador **C-MAPSS — Commercial Modular Aero-Propulsion System Simulation**, que representa a operação e a degradação de motores aeronáuticos turbofan comerciais. São dados de simulação: no treinamento, cada trajetória acompanha um motor até a falha; no teste, o histórico termina antes da falha e o RUL verdadeiro do último ciclo é fornecido separadamente. [Catálogo oficial do dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

Foi utilizado o subconjunto **FD001**, com 100 motores de treinamento e 100 de teste, uma condição operacional e um modo de falha: degradação do compressor de alta pressão (HPC). Essa configuração permite estudar previsão de vida restante sob condições controladas; não demonstra desempenho em uma frota real.

**Referência do dataset:** Saxena, A.; Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames, Prognostics Data Repository. [Fonte e crédito original](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

Os dados locais foram preservados em `data/raw/CMAPSSData.zip` e `data/raw/dataset`. A auditoria inicial e o endereço da distribuição utilizada estão em `CONTEXT_HANDOFF.md`. Este arquivo registra a fase inicial e não descreve o estado final do projeto.

## Resultado registrado

| Modelo | MAE (ciclos) | RMSE (ciclos) | Erro médio assinado (ciclos) |
| --- | ---: | ---: | ---: |
| Extra Trees | 16,23 | 23,80 | +10,08 |
| XGBoost | 17,29 | 25,15 | +9,31 |
| HistGradientBoosting | 17,77 | 27,01 | +8,83 |

O registro corresponde ao benchmark oficial em 100 motores, usando uma previsão no último ciclo disponível de cada trajetória truncada. Os hiperparâmetros dos três candidatos foram congelados antes do benchmark. A escolha final do Extra Trees foi realizada depois da comparação nesse teste; portanto, o resultado deve ser apresentado como benchmark comparativo, sem alegar uma segunda avaliação independente após a seleção. Não foram realizados novos ajustes com base nesses resultados.

O erro positivo indica superestimação da vida restante. O modelo não deve ser interpretado como um limite seguro para operação.

## Percurso de leitura

A síntese está em [27 — conclusão e entrega](notebooks/pt-BR/27_conclusao_entrega_fd001.ipynb). A leitura técnica principal pode ser concentrada em [16 — ablação temporal](notebooks/pt-BR/16_ablation_atributos_temporais_rul_fd001.ipynb), [18 — triagem](notebooks/pt-BR/18_triagem_modelos_atributos_temporais_rul_fd001.ipynb), [19 — otimização](notebooks/pt-BR/19_otimizacao_optuna_atributos_temporais_rul_fd001.ipynb), [21 — benchmark](notebooks/pt-BR/21_benchmark_oficial_finalistas_atributos_temporais_rul_fd001.ipynb), [22 — decisão e erros](notebooks/pt-BR/22_decisao_final_analise_erro_extra_trees_fd001.ipynb), [24 — importância dos atributos](notebooks/pt-BR/24_importancia_features_extra_trees_fd001.ipynb), [25 — alertas e custos](notebooks/pt-BR/25_alertas_modelo_final_extra_trees_fd001.ipynb) e [26 — incerteza](notebooks/pt-BR/26_incerteza_extra_trees_final_fd001.ipynb).

Os demais notebooks preservam o histórico de experimentos e podem ser consultados como apêndice. Reexecutar todos eles não é necessário para utilizar o modelo; alguns executam buscas longas ou experimentos anteriores à configuração final.

## Artefato e contrato de entrada

O pipeline está em `outputs/models/extra_trees_fd001_v1/model.joblib`, acompanhado de `metadata.json`, que registra atributos, parâmetros, versões, hashes e verificações. Foram oferecidos 205 atributos e selecionados 93. As janelas causais são de 5, 10, 20, 40 e 80 ciclos, com médias, desvios-padrão populacionais e deltas por motor. O ciclo atual participa das janelas.

O CSV de entrada deverá conter `unit`, `cycle`, `setting1`, `setting2`, `s2`, `s3`, `s4`, `s6`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20` e `s21`. Todas as colunas obrigatórias deverão ser numéricas, sem valores ausentes ou infinitos. Cada motor deverá possuir ciclos consecutivos desde o ciclo 1 até a medição atual. Os registros poderão estar fora de ordem, pois serão ordenados por motor e ciclo. O histórico será usado para reconstruir as features, e a saída padrão conterá somente a previsão mais recente de cada motor.

O exemplo incluído foi extraído de um prefixo do treinamento e serve exclusivamente para demonstrar o funcionamento da interface e a equivalência do pipeline, sem medir generalização.

## Executar uma previsão

No PowerShell, a partir da raiz do projeto:

```powershell
Set-Location -LiteralPath "D:\Projetos\CMAPSS_Predictive_Maintenance"
$python = "C:\Users\enior\miniforge3\envs\CMAPSS_Py312_Experimental\python.exe"
& $python scripts\predict_final_extra_trees_fd001.py --input outputs\models\extra_trees_fd001_v1\historico_exemplo.csv --output outputs\reports\previsao_demonstracao_fd001.csv
```

A saída conterá `unit`, `cycle` e `rul_predito`. Para prever todos os ciclos do histórico fornecido, poderá ser acrescentado `--all-cycles`. Por padrão, arquivos de saída existentes serão preservados e deverá ser escolhido outro nome.

O ambiente original é descrito em `environment.yml`; as versões efetivamente usadas na entrega estão em `metadata.json`. O arquivo de ambiente inclui um caminho local editável para `multilang`, que deverá ser ajustado em outro computador. O formato joblib deverá ser carregado em um ambiente compatível e somente a partir de um artefato local confiável.

## Reconstruir o artefato

O comando `scripts/export_final_extra_trees_fd001.py` ajusta uma única vez o pipeline com parâmetros congelados, usando apenas o treinamento FD001 e o estudo Optuna existente. Não inicia uma nova busca. A entrega existente é preservada e uma nova exportação para o mesmo diretório é recusada. O script verifica a equivalência de previsões após serialização, a correspondência com a engenharia utilizada no treino, a invariância das previsões anteriores quando o histórico é estendido e a rejeição de históricos inválidos.

## Limites e interpretação

A importância por permutação mostrou forte contribuição do ciclo observado e dos desvios-padrão móveis, especialmente na janela de 80 ciclos. A correlação entre features dificulta interpretar a importância individual como medida exclusiva de informação. A explicabilidade é preditiva, sem identificar causas físicas de falha.

Os alertas foram analisados em trajetórias OOF completas de desenvolvimento, com uma janela útil ilustrativa de 5 a 30 ciclos e confirmação em três ciclos. O limiar 20 produziu alertas oportunos nos 100 motores, com mediana de 16 ciclos de antecedência. Os folds também participaram da escolha de hiperparâmetros, de modo que esse resultado não representa uma política industrial validada. Os custos são relativos e hipotéticos.

As faixas empíricas de erro foram calibradas e avaliadas em motores distintos dentro dos folds OOF. A cobertura média de ciclos por motor foi 85,9%, com largura média de 91,3 ciclos e cobertura de trajetórias inteiras de 46%. A referência de calibração de 90% não é uma garantia de cobertura. O artefato exportado fornece somente a previsão pontual; essas faixas não foram promovidas a intervalos do modelo refitado.

O escopo entregue é FD001, em dados simulados. O README, o relatório HTML e o notebook 27 estão disponíveis em português e inglês; os demais notebooks são mantidos na edição canônica em português. Generalização para FD002–FD004 e validação industrial permanecem extensões. Uma decisão real de manutenção exige custos, antecedência mínima e condições operacionais definidos com a equipe responsável.

## Relatório HTML e publicação

O relatório possui versões em [português](site/index.html) e [inglês](site/en.html), com gráficos interativos locais e sínteses para download. Para reconstruí-lo, execute `python scripts/build_portfolio_pages.py`. Nenhum modelo é treinado por esse comando.

O conteúdo da página está isolado em `site/`; `docs/` é reservado a referências pessoais e não integra o repositório público. No portfólio, o projeto ocupa `CMAPSS_Predictive_Maintenance/`. O fluxo `Publicar portfólio FD001`, na raiz do repositório, publica somente o relatório estático e a página de entrada, sem treinar modelos. [Guia de publicação](site/PUBLICACAO_GITHUB_PAGES.md)
