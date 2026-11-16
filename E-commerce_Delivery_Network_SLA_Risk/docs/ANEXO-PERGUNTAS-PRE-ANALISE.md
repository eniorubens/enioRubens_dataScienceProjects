# Anexo — Perguntas para a pré-análise da fonte de dados

Este anexo preserva as perguntas levantadas na leitura inicial do projeto. Elas devem ser respondidas com evidências do dataset antes da aprovação do MVP. Quando a fonte não permitir uma resposta confiável, a escolha da fonte de dados e o escopo do projeto deverão ser reconsiderados.

## Decisões bloqueadoras

1. **População supervisionada:** serão usados apenas pedidos com status `delivered` e data real de entrega válida? Como serão tratados `canceled`, `unavailable`, pedidos sem entrega e entregas com datas inconsistentes?

2. **Cálculo do alvo:** o atraso será medido em dias fracionários ou inteiros? Haverá arredondamento? Qual fuso horário será considerado?

3. **Momento da previsão:** o ponto de corte será a compra, a aprovação do pagamento ou o início da preparação? Quais campos estarão permitidos nesse instante?

4. **Granularidade da tabela analítica:** será uma linha por pedido? Como serão agregados pedidos com vários itens ou vários vendedores?

5. **Distância aproximada:** como será calculada: coordenadas dos CEPs, centroides de cidades, distância entre estados ou outra regra? Como serão tratados CEPs sem geolocalização?

6. **Granularidade do grafo:** os nós serão vendedores e clientes, vendedores e estados, vendedores e regiões, ou outra combinação? O grafo será bipartido e direcionado?

7. **Variáveis do grafo e vazamento:** o atraso médio aparece como peso da aresta e também pode virar variável preditora. Ele será usado apenas na análise descritiva ou também no modelo? Se entrar no modelo, deverá ser calculado somente com histórico anterior ao pedido, nunca com dados do próprio período de previsão.

8. **Janela temporal do grafo:** as métricas serão calculadas cumulativamente até cada data ou em janelas móveis? Como serão tratados vendedores, destinos e rotas ainda não observados no treinamento?

9. **Divisão temporal:** quais são as datas ou proporções exatas de treino, validação e teste? Haverá intervalo de segurança entre os períodos para evitar dependência temporal?

10. **Pedidos cancelados:** cancelamento será uma população excluída, uma análise separada ou algum tipo de violação operacional?

## Modelagem e avaliação

11. No OLS, a intenção é apenas inferência associativa ou haverá alguma interpretação causal? Serão usados erros-padrão robustos ou agrupados?

12. Como será aplicada a transformação Box-Cox se `delivery_delay_days` puder assumir valores negativos?

13. “Maiores atrasos” significa quais casos: top 5%, top 10%, atraso acima de determinado número de dias ou outro corte?

14. “Top-k” na classificação significa top 5%, top 10%, quantidade fixa de pedidos ou capacidade operacional real?

15. Qual métrica será usada para calibração: Brier score, ECE, curva de confiabilidade ou outra?

16. Como será definido o modelo vencedor quando os estimators apresentarem compromissos diferentes entre MAE, estabilidade, interpretabilidade e latência?

17. Qual orçamento será considerado comparável no Optuna: número de trials, tempo máximo por estimator, quantidade de avaliações ou combinação desses critérios?

18. Quais critérios determinam que o dataset Olist “não converge para uma boa análise e estimativa” e deve ser substituído?

## Contratos e entregáveis

19. Onde estão os “scripts da aula” que devem ser avaliados quanto à reutilização conceitual?

20. Os contratos serão implementados em JSON Schema, Great Expectations, testes Python próprios ou outra ferramenta?

21. O que deve ser considerado evidência suficiente para `production_eligible = true`? Ou este projeto permanecerá deliberadamente exploratório, mantendo `false`?

22. A linguagem passiva se aplica apenas ao texto dos notebooks ou também a docstrings, comentários, README e relatório HTML?

23. Os notebooks PT-BR e EN-US terão execução idêntica e resultados equivalentes? Serão dois arquivos independentes ou uma tradução gerada a partir de uma fonte única?

24. Qual padrão de referência bibliográfica deverá ser usado: ABNT, APA ou apenas autor–obra–ano?

25. O relatório HTML bilíngue será uma página com seletor de idioma ou duas páginas separadas?

26. A publicação da Fase 8 está autorizada como parte do projeto ou dependerá de uma aprovação explícita posterior?

## Prioridade para a pré-análise

As perguntas 1–10 são bloqueadoras para decidir se o Olist sustenta o problema proposto. As perguntas 11–18 dependem dos achados da auditoria e da viabilidade da tabela analítica. As perguntas 19–26 podem ser fechadas durante o desenho do MVP, depois que a fonte de dados for considerada adequada.

