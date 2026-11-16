# Plano da Fase 9 — relatório e publicação

**Estado:** **aprovado e concluído em 2026-11-16** (P-07 em `REGISTRO-DE-DECISOES.md`). A publicação remota ficou para depois da revisão do responsável.
**Ponto de partida:** fases 1 a 8 concluídas. O F1 (notebook 09) é o resultado oficial e foi **inconclusivo**. O F2 (notebook 10) é a análise complementar pós-hoc. `production_eligible = false`. Julho e agosto de 2018 estão esgotados para novas avaliações.

## 1. Princípio que ordena o trabalho

Toda alteração no código das células de um notebook invalida a execução já feita. Por isso a ordem é:

1. Primeiro, só documentos e arquivos fora dos notebooks.
2. Depois, o notebook novo do relatório.
3. Por último, a execução final e a publicação.

Nenhuma alteração pode mudar resultados do F1 ou do F2. O cálculo deles está congelado, e qualquer mudança em `src/` só gera aviso de fingerprint.

## 2. Frentes de trabalho

| Frente | Conteúdo | Quem faz | Depende de |
|---|---|---|---|
| A. Reexecução | Concluída para os notebooks 01 a 11 em PT e EN. O gate `REQUIRE_NOTEBOOK_OUTPUTS=1` passou 12/12. O notebook 07 reutiliza o cache do torneio; 08, 09 e 10 reutilizam os artefatos congelados conforme o contrato. | Concluído | — |
| B. Documentos | Atualização do que parou em 2026-11-11. São quatro documentos, listados abaixo da tabela. | Eu | — |
| C. Relatório final | Notebook `11_relatorio_final` (PT) e `11_final_report` (EN) que **só lê** saídas gravadas: torneio V1, Optuna, F1, F2 e ledger. O texto é narrativo, sem novo cálculo de modelo. É exportado para HTML. | Agente menor escreve; responsável executa | A (cache do torneio) e P-07 |
| D. Repositório | Os itens são listados abaixo da tabela. | Agente menor | P-07 |
| E. Controle de versão local | `git init` e primeiro commit, só na máquina, para acompanhar as alterações da revisão. Nada é enviado a repositório remoto. | Agente menor, com autorização do responsável | B, C, D |

**Fora da Fase 9:** a publicação em repositório remoto (GitHub). Ela fica para depois da revisão completa do projeto pelo responsável, que pode incluir novos gráficos nos notebooks.

**Documentos da frente B:**
- seção 8 da `REVISAO-TECNICA-ESCOPO-E-GAPS.md`;
- fases 2, 4 e 6 da `SEQUENCIA-DE-TRABALHO.md`;
- cabeçalho e tabela de perguntas do `REGISTRO-DE-DECISOES.md` (perguntas 4, 9 e 15 têm "próximo gate" já cumprido);
- seção "Estado desta migração" da `ARQUITETURA-MODULAR.md`.

**Itens da frente D:**
- Remoção dos 9 CSVs duplicados da raiz de `data/raw/olist/`. Foi verificado em 2026-11-16 que têm o mesmo SHA-256 das cópias em `csv/`, que é a pasta lida pelos notebooks.
- Inclusão de `data/raw/**/*.csv` e `*.zip` no `.gitignore`.
- Script `scripts/verify_data.py`, que confere o SHA-256 do ZIP registrado em `data/raw/olist/README.md` e extrai os CSVs para `csv/`. O download continua manual, pelo Kaggle, porque exige credencial.
- Exceção no `.gitignore` para versionar `outputs/runs/final_evaluation/**/*.csv`, os `metadata.json` e o `LEDGER.jsonl`. Os `.pkl`, o modelo joblib e o SQLite do Optuna não são versionados.
- `pip install -e .` no `environment.yml`.
- `README.md` na raiz, com objetivo, como reproduzir, mapa dos notebooks, resultado (F1 oficial e F2 complementar), limitações e o estado `production_eligible = false`.

## 2a. Frente F — gráficos nos notebooks 05, 07, 09 e 10

A frente F foi concluída em 2026-11-16: os gráficos planejados foram incorporados aos notebooks 05, 07, 09 e 10.

**Restrição técnica que define onde fica o código.** O cache do torneio (`run_cache`) inclui no fingerprint o hash de todos os `*.py` de `src/ecom_delivery_sla/`. Qualquer função nova nesse pacote invalida o cache que o notebook 07 está gerando agora. Com isso, a próxima execução do 07 treinaria tudo de novo. O F1 e o F2 só emitiriam aviso.

Por isso, as funções de gráfico ficam num pacote separado de apresentação, `src/ecom_delivery_sla_viz/`, com rótulos PT e EN próprios. Esse pacote não entra no fingerprint e não importa nada que treine modelo. Os gráficos já existentes em `eda.py`, `geo.py` e `temporal.py` ficam onde estão.

| Notebook e seção | Gráfico | Fonte (sem novo cálculo) |
|---|---|---|
| 05, "Folds expanding-window de desenvolvimento" | Linha do tempo em barras horizontais, uma por fold: treino, embargo de 30 dias e validação, com o número de pedidos. Ao lado, prevalência de treino contra validação por fold. | `expanding_fold_summary(folds)` |
| 05, "Deriva das features entre treino e validação" | Barras horizontais do PSI por feature, em escala log, com linhas de referência em 0,10 e 0,25. As features de calendário ficam em cor neutra, porque o PSI alto delas vem da própria construção. | `psi_table` |
| 07, "Mesmo protocolo" | AP por fold de cada modelo, com a melhor baseline tracejada. Ganho médio sobre a melhor baseline por modelo, com a linha do critério P-01 (5% de AP e 3% de MAE). | `classification_results`, `regression_results`, `tournament_summary` |
| 07, "Ablação da feature de distância" | Métrica principal por variante de distância, com desvio entre folds | `ablation_summary` |
| 09, "Drift entre treino e teste" | Mesmo gráfico de PSI do 05 | `final["psi"]` |
| 09, "Classificação: métricas com intervalo e diferença pareada" | Gráfico de intervalos da AP com IC 95% (CatBoost e baselines), e diferença pareada com a linha do zero | `final["classification_metrics"]`, `final["paired_difference"]` |
| 10, seção 1 | Mediana do prazo prometido (barras) e AUC da regra da promessa (linha) por mês | `promise_regime_table` |
| 10, seção 2 | Barras empilhadas de rótulos conhecidos e desconhecidos por mês, com a prevalência de cada grupo | `posthoc["label_availability"]` |
| 10, seção 4 | AP por fold de cada variante contra a configuração do F1, reutilizando a função do 07 | `posthoc["gate_folds"]` |
| 10, seção 5 | Gráfico de intervalos da AP das variantes e das referências, e diferenças pareadas, reutilizando a função do 09 | `posthoc["classification_metrics"]`, `posthoc["paired_difference"]` |

**Regras da frente F:**
- O gráfico entra na mesma célula de código da tabela, logo depois dela. A sequência Markdown → código → insight se mantém.
- O Markdown de abertura e o insight de cada seção passam a citar o gráfico, sem números novos.
- O código continua idêntico entre PT e EN, mudando só `LANG`.
- Cada função tem teste sintético: devolve `go.Figure`, tem os traços esperados, não altera a tabela de entrada e funciona em PT e EN.
- Um teste garante que `ecom_delivery_sla_viz` não importa módulos de treino.
- Os notebooks 06 e 08 ficam fora desta proposta. Podem ganhar gráficos na revisão do responsável: coeficientes do OLS e histórico do Optuna.

**Ordem executada:**
1. O notebook 05 foi atualizado com os gráficos do contrato temporal e da deriva.
2. O notebook 07 foi atualizado após a estabilização do cache do torneio.
3. Os notebooks 09 e 10 foram atualizados com as visualizações da avaliação oficial e da análise pós-hoc.

**Reexecução realizada:**
- **05:** reconstruiu as features e registrou os gráficos executados.
- **07:** carregou o cache do torneio sem alterar os resultados.
- **09 e 10:** carregaram os resultados congelados, sem reabrir o teste.

O notebook 11 reutiliza essas mesmas funções.

## 3. Conteúdo do relatório (notebook 11)

O relatório segue o padrão Markdown → código → insight, em voz passiva e sem bullets. As seções são:

1. pergunta e escopo;
2. fonte e população;
3. alvo por dia de calendário e censura;
4. features disponíveis na compra e rede as-of;
5. protocolo temporal e baselines obrigatórias;
6. resultado do desenvolvimento (P-01: CatBoost, +13,9% de AP em 8 de 9 folds; regressão sem candidato);
7. F1 oficial (inconclusivo), com tabela, IC e calibração;
8. F2 complementar (exploratório), com o viés de rótulos lentos e a calibração que muda de direção;
9. o que os dados sustentam e o que não sustentam;
10. limitações;
11. hipóteses para um próximo ciclo.

O relatório deve ter gráficos, e não só tabelas, todos construídos a partir dos arquivos gravados. A proposta inicial é:
- AP por fold de cada modelo contra a melhor baseline (torneio V1);
- AP com IC 95% do F1 e do F2 lado a lado com as referências;
- curvas de confiabilidade do F1 e das variantes do F2;
- disponibilidade de rótulos por mês na atualização do F2;
- mediana do prazo prometido e AUC da regra da promessa por mês.

Os gráficos acrescentados na revisão foram incorporados e os notebooks afetados foram reexecutados. Isso não altera o F1 nem o F2, que permanecem congelados.

Todo número citado vem de uma célula que lê os arquivos gravados. Nenhuma célula treina modelo nem abre o teste. O export para HTML é feito por `python scripts/export_report.py`, usando a API do nbconvert no próprio processo, sem o código de entrada. O script também converte os gráficos Plotly, gravados pelo VS Code só em JSON, em HTML com o plotly.js e o contorno do mapa embutidos. Como a exportação é offline, uma variável externa `SSLKEYLOGFILE` é removida durante a conversão para impedir que um endpoint de proxy do ambiente interrompa o nbconvert; isso não altera dados, modelos ou notebooks.

## 4. Itens que não são feitos, e por quê

**Remoção de `sys.path` nos 20 notebooks (resto do R-19).** Exigiria alterar o código de todos os notebooks e reexecutar tudo, inclusive o 08. O ganho é só estético. O item fica registrado como limitação conhecida; o `pip install -e .` resolve a importação para quem reproduz.

**`OneHotEncoder(drop=None)` (resto do R-19).** Mudaria o pré-processamento usado no torneio e no F1 depois de o teste ter sido aberto. O item é encerrado como "não aplicado após a abertura do teste".

**Nova avaliação de qualquer modelo em julho ou agosto de 2018.** Está vetada pelo registro.

## 5. Perguntas do anexo ainda pendentes

| Pergunta | Proposta de fechamento |
|---:|---|
| 12 (Yeo-Johnson no atraso) | Encerrada sem aplicação: a regressão não teve candidato (P-04), e o OLS é inferencial, com erros robustos. |
| 13 (cortes de "maiores atrasos") | Encerrada com top 5% e top 10% como cenários de capacidade. O corte operacional real fica fora do escopo, porque não há dado de capacidade. |
| 19 (scripts externos ou de aula) | Encerrada como não incorporados, salvo indicação do responsável. |
| 22 (padrão editorial) | Aprovada: voz passiva, prosa sem bullets nos notebooks e no relatório, e tabelas permitidas. Esse já é o padrão aplicado. |
| 24 (padrão bibliográfico) | Autor-data simples no fim do relatório: Olist/Kaggle, scikit-learn, CatBoost, Optuna e as referências metodológicas citadas. |
| 25 (formato bilíngue) | Dois arquivos separados (PT e EN), como os notebooks, sem seletor de idioma. |
| 26 (publicação) | Fora da Fase 9: a publicação em repositório remoto é decidida depois da revisão completa do projeto pelo responsável. |

## 6. Sequência executada e estado final

1. O cache do torneio V1 foi verificado e o notebook 07, em PT e EN, carregou os resultados compatíveis.
2. Os documentos da frente B foram atualizados e alinhados aos gates concluídos.
3. A frente D foi concluída: duplicatas de dados foram removidas após comparação de hash, a verificação de dados foi criada e os artefatos versionáveis foram delimitados.
4. O notebook 11 foi criado em PT e EN com testes de contrato, paridade e ausência de novo treinamento.
5. O relatório foi executado nas duas línguas, exportado para HTML e validado com o gate 12/12.
6. Os números e textos foram confrontados com os artefatos gravados; `production_eligible = false` permanece explícito.
7. A publicação remota continua fora desta fase e depende da revisão final do responsável.

## 7. Decisões encerradas e pendências remanescentes

1. O formato aprovado é o notebook 11 bilíngue exportado para HTML.
2. A remoção dos CSVs duplicados e o versionamento apenas dos resultados pequenos do F1 e do F2 foram concluídos.
3. O restante do R-19 (`sys.path` e `OneHotEncoder`) não foi aplicado após a abertura do teste, conforme a seção 4.
4. As perguntas 12, 13, 19, 22, 24 e 25 foram encerradas conforme a seção 5; a pergunta 26 permanece como decisão posterior sobre publicação remota.
