# Contrato de dados — Fase 2

**Status:** aprovado como contrato de dados em 2026-11-16 (ver "Estado do gate"); `production_eligible = false`  
**Fonte:** Olist Brazilian E-Commerce Public Dataset  
**Escopo:** população supervisionada, alvo, granularidade, disponibilidade temporal e integridade

Este documento consolida as decisões sustentadas pela auditoria local e separa regras aprovadas, regras provisórias e decisões ainda abertas. A Fase 2 não seria considerada concluída enquanto os itens abertos não tivessem uma decisão registrada e testes correspondentes. Atualização de 2026-11-16: os itens abertos foram fechados ao longo das fases 4 a 9. A seção "Regras que estavam provisórias e como foram resolvidas" registra a solução de cada um, e as seções históricas foram mantidas com notas de atualização.

## Objetivo de negócio

Avaliar o cumprimento do prazo de entrega prometido ao cliente e preparar uma tabela analítica capaz de sustentar:

- regressão do atraso em dias;
- classificação de violação do prazo;
- análise histórica de padrões por tempo, destino, vendedor, rota e categoria;
- comparação entre associação estatística e desempenho preditivo.

O SLA é operacionalizado como uma proxy do prazo prometido: `order_estimated_delivery_date`. O contrato principal de previsão é definido **no momento da compra**. A violação é avaliada por dia de calendário, porque a data estimada não possui componente de hora: há violação quando a data de entrega ao cliente é posterior à data estimada. O dataset não contém o contrato comercial completo da Olist.

## Regras já sustentadas pela evidência

### Unidade e população

- A unidade de análise é o pedido (`order_id`).
- `order_id` deve ser único na tabela de pedidos.
- A população supervisionada provisória contém pedidos com:
  - `order_status == "delivered"`;
  - `order_delivered_customer_date` não nula;
  - `order_estimated_delivery_date` não nula.
- A auditoria local encontrou 96.470 pedidos nessa população.
- Cancelados, indisponíveis e pedidos sem entrega real não entram no alvo supervisionado.
- Esses pedidos não devem ser apagados: devem permanecer disponíveis para uma análise descritiva de pedidos não concluídos.

### Alvo

```text
delivery_delay_days =
    (order_delivered_customer_date - order_estimated_delivery_date).total_seconds() / 86.400

delivery_delay_calendar_days =
    date(order_delivered_customer_date) - date(order_estimated_delivery_date)

sla_breach = delivery_delay_calendar_days > 0
```

- O atraso permanece em dias fracionários; não há arredondamento nesta etapa.
- `delivery_delay_calendar_days` é o atraso inteiro entre as datas de calendário.
- Uma entrega às 15h do dia estimado tem atraso contínuo positivo, mas `delivery_delay_calendar_days == 0` e não é violação.
- A regra binária por calendário foi aprovada após a auditoria da ausência de hora na data estimada. Ela altera a taxa provisória de 8,11% para 6,77%.
- A regra não corrige datas anômalas silenciosamente.
- Entrega antes do prazo gera atraso negativo e não é violação.
- Entrega na data estimada gera atraso zero e não é violação.

### Momento da previsão

O modelo principal responde à pergunta: **quais pedidos apresentam maior risco no momento da compra?** Portanto, somente campos disponíveis no checkout ou derivados de informações conhecidas nesse instante podem entrar na matriz principal. Variáveis que só existem na aprovação do pagamento, preparação ou despacho devem ficar fora do modelo principal ou ser tratadas em um cenário operacional separado.

### Disponibilidade temporal da primeira matriz

O catálogo executável em `src/ecom_delivery_sla/features.py` registra a disponibilidade e a autorização de uso de cada feature. O campo `promised_lead_days` é tratado como disponível na compra porque deriva da data estimada de entrega associada ao pedido no dataset; a [página oficial do dataset no Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce/home/) confirma que a fonte descreve pedidos, prazo estimado e o ciclo posterior de atendimento, mas não constitui o contrato comercial interno da Olist. Os campos de pagamento escolhidos no checkout (`primary_payment_type` e `payment_installments_max`) seguem a mesma regra operacional e devem ser interpretados como informação do pedido, não como prova de aprovação financeira.

`shipping_limit_date` continua fora da matriz principal: é um limite de despacho por item e sua disponibilidade contratual é marcada como `available_at_approval`. `handling_days_min` e `handling_days_max` são calculados para análise operacional, mas não são utilizados no modelo que pontua o pedido no instante da compra.

Também ficam fora da matriz principal `payment_value_sum`, `distance_min_km` e `distance_max_km`: os dois primeiros grupos são mantidos apenas para auditoria de redundância/qualidade, enquanto `distance_mean_km` é a única distância aproximada autorizada. Nenhuma feature com status `quality_control` ou `available_at_approval` pode ser selecionada pelo catálogo de modelagem.

### Informação pós-evento proibida

As seguintes variáveis não podem ser preditoras em um cenário de decisão no momento da compra:

- `order_delivered_carrier_date`;
- `order_delivered_customer_date`;
- `order_status` quando representar evolução posterior do pedido;
- avaliações e datas de avaliação;
- qualquer variável calculada a partir do atraso ou da entrega real do próprio pedido;
- qualquer métrica histórica que utilize o próprio pedido ou observações futuras.

## Investigação de múltiplos vendedores e entrega parcial

A auditoria local confirmou que um pedido pode reunir itens de vendedores diferentes:

- 1.278 pedidos possuem mais de um vendedor na tabela `order_items`;
- 1.275 desses pedidos pertencem à população supervisionada;
- 336 pedidos com múltiplos vendedores possuem mais de uma `shipping_limit_date` distinta;
- 942 pedidos com múltiplos vendedores possuem a mesma `shipping_limit_date` para os itens observados.

Entretanto, `shipping_limit_date` é o prazo-limite de despacho do item pelo vendedor. Ele não é uma data observada de entrega daquele item ao cliente. A tabela `orders` possui apenas uma data de despacho ao transportador, uma data de entrega ao cliente e uma data estimada para cada `order_id`.

Assim, o dataset não permite confirmar se houve entrega parcial nem medir um SLA individual por vendedor ou item. Uma `order_sellers` criada apenas pela explosão de `orders` repetiria o mesmo alvo de pedido em várias linhas e poderia atribuir ao vendedor uma entrega que não foi observada individualmente.

**Decisão de trabalho:** manter o SLA principal no nível do pedido. Podemos construir uma tabela auxiliar pedido–vendedor para descrever composição, quantidade de vendedores, prazos de despacho e origem, mas não trataremos cada linha como uma observação independente de SLA sem uma data de entrega individual. A política de agregação foi definida e implementada: os indicadores de risco e de *cold start* são resumidos pelo máximo entre os vendedores do pedido, as contagens de histórico pelo mínimo, e a média ponderada pelo valor dos itens é mantida como coluna auxiliar, fora do modelo (notebook 04); a distância entra no modelo pela média (P-03).

### Indicador histórico de confiabilidade do vendedor

O cenário proposto é operacionalmente relevante: um vendedor que frequentemente ultrapassa seu prazo de despacho pode aumentar o risco do pedido inteiro. Podemos representar esse sinal no modelo de pedido, desde que o indicador use somente histórico anterior à compra.

Variáveis candidatas para cada vendedor:

- taxa histórica de pedidos que ultrapassaram `shipping_limit_date`;
- mediana histórica do atraso de despacho em dias;
- quantidade de pedidos históricos usados para calcular a taxa;
- recência da última observação histórica.

Para pedidos com vários vendedores, as variáveis podem ser agregadas no nível do pedido, por exemplo:

- maior risco histórico entre os vendedores;
- risco médio ou ponderado pela quantidade/valor dos itens;
- quantidade de vendedores;
- indicador de presença de pelo menos um vendedor de alto risco;
- proporção dos itens associada a vendedores de alto risco.

Há uma limitação crítica: o dataset possui `shipping_limit_date` por item, mas apenas uma `order_delivered_carrier_date` por pedido. Por isso, comparar diretamente essas datas em um pedido com vários vendedores pode atribuir a todos os vendedores o atraso de despacho observado no pedido. Como primeira regra de qualidade, a taxa histórica individual deve ser estimada preferencialmente em pedidos com um único vendedor; pedidos com vários vendedores podem ser usados para gerar sinais agregados, mas não para afirmar qual vendedor causou o atraso.

Esses indicadores só poderão entrar como features depois de testes de disponibilidade temporal: para cada pedido, o histórico do vendedor deve ser calculado excluindo o próprio pedido e qualquer observação posterior à sua compra. Vendedores sem histórico receberão uma política explícita de *cold start*. Atualização: implementado em `network.py`. O histórico usa apenas desfechos conhecidos antes da compra e exclui o próprio pedido; as taxas são suavizadas em direção à taxa global as-of com força m = 20, de modo que uma combinação sem histórico recebe a taxa global, e os indicadores `seller_cold_start` e `edge_cold_start` marcam esses casos (T-01 a T-03).

## Regras que estavam provisórias e como foram resolvidas

| Tema | Regra de trabalho original | O que faltava aprovar | Situação em 2026-11-16 |
|---|---|---|---|
| Múltiplos vendedores | Manter o SLA no nível do pedido; tabela auxiliar pedido–vendedor é possível | Agregação de vendedores, origem, distância e indicadores de multiplicidade | Resolvida: máximo para risco e *cold start*, mínimo para contagens de histórico, média ponderada auxiliar fora do modelo (notebook 04); distância média (P-03) |
| Itens e categorias | Agregar explicitamente para uma linha por pedido | Política para pedidos com várias categorias e empates | Resolvida: contagens de itens, produtos e categorias distintas e vendedor principal por valor (`features.py`) |
| Pagamentos | Agregar por pedido | Campos e tratamento de múltiplos pagamentos | Resolvida: agregação por pedido; o pedido sem registro de pagamento é mantido com `payment_record_missing`, como controle de qualidade |
| Geolocalização | Usar coordenadas agregadas por prefixo de CEP apenas como hipótese | Regra definitiva de agregação e cobertura mínima | Resolvida: mediana das coordenadas por prefixo de CEP; os 476 pedidos sem distância são mantidos com `distance_missing` |
| Disponibilidade de features | Compra, cliente, vendedor, item, produto e destino são candidatos | Evidência de que cada campo estava disponível no momento da compra | Resolvida: catálogo temporal sem itens pendentes, com 14 features de rede `available_at_purchase` |
| Datas | Usar os timestamps originais convertidos pelo pandas | Confirmar tratamento de fuso, precisão e eventuais datas impossíveis | Mantida como regra: timestamps originais, sem conversão de fuso; o alvo por dia de calendário independe da hora, e datas impossíveis são tratadas pela auditoria de anomalias |
| Anomalias | Isolar, contabilizar e não reparar silenciosamente | Decidir inclusão, exclusão ou análise de sensibilidade por tipo | Resolvida: tabela de anomalias abaixo, reproduzida por `audit.py` |
| Divisão temporal | Usar corte temporal, sem divisão aleatória como resultado principal | Fixar períodos de treino, validação e teste | Resolvida: folds mensais expanding de 2017-10 a 2018-06 com embargo de 30 dias; teste único em jul–ago/2018 (F1, P-05) |
| Cold start | Ainda não definido | Política para vendedor, destino ou rota sem histórico | Resolvida: prior global as-of e indicadores de *cold start* (`network.py`, T-01); UFs de vendedor ausentes no treino viram vetor zero no one-hot, igual à referência omitida, e não são sempre marcadas por `seller_cold_start` (T-11) |

## Corte temporal inicial (registro histórico)

Foi implementada em `src/ecom_delivery_sla/temporal.py` e demonstrada no notebook `05_contrato_temporal_e_split.ipynb`. Este bloco registra o primeiro corte exploratório; o protocolo vigente para seleção de modelos é o expanding-window com embargo, e não esta divisão única. Usando `order_purchase_timestamp` como referência, o primeiro corte histórico foi:

- treino: antes de `2018-04-01`, 64.320 pedidos (66,67%);
- validação: de `2018-04-01` até antes de `2018-07-01`, 19.643 pedidos (20,36%);
- teste: a partir de `2018-07-01`, 12.507 pedidos (12,96%).

Os cortes são cronológicos, sem embaralhamento. Eles foram mantidos como referência exploratória. Para seleção de modelos, o corte único foi substituído pelos folds expanding com embargo, e o período de teste ficou reservado à avaliação única F1 (treino até 2018-05-31, teste em jul–ago/2018).

O baseline de modeling utiliza as features candidatas aprovadas para a primeira comparação e mantém os indicadores de controle de qualidade fora da matriz operacional. A execução real inicial produziu matriz de 64.320 linhas de treino por 44 features, sem deficiência de posto no OLS.

## Evolução da tabela analítica (registro histórico)

Foi implementada uma primeira tabela no notebook `04_tabela_analitica_candidata.ipynb`, com código reutilizável em `src/ecom_delivery_sla/features.py`. O registro abaixo documenta sua evolução; a tabela vigente é construída para todos os pedidos elegíveis e preserva uma linha por `order_id`.

Na execução direta com os CSVs locais, a tabela de scoring agora tem 99.441 pedidos, `order_id` único e 39 colunas; o recorte supervisionado tem 96.470 linhas. A função é reutilizável para todos os pedidos e preserva a mesma granularidade; `build_modeling_frame` faz o recorte posterior para a população supervisionada. As novas agregações incluem peso, volume, vendedor principal por valor, UF compartilhada, calendário e distância média Haversine. Os campos derivados da entrega efetiva não são retornados pela tabela de scoring.

O catálogo de features classifica cada coluna como disponível na compra, disponível na aprovação ou controle de qualidade e inclui `use_in_model`. Não há mais status `pending*`. A matriz operacional é derivada do catálogo, e não de listas manuais no módulo de modeling.

### Política de missingness adotada

- Missing em `distance_min_km`, `distance_mean_km` e `distance_max_km` não remove o pedido. A imputação usa estatística ajustada somente no treino e mantém um indicador de ausência.
- Missing nos campos agregados de pagamento permanece como controle de qualidade. O pedido sem registro de pagamento é preservado, sem converter a ausência em comportamento observado ou em valor zero.
- A imputação nunca é calculada sobre treino, validação e teste juntos.
- `distance_missing`, `payment_record_missing` e `coordinate_coverage` permanecem como controle de qualidade e não entram no baseline operacional. O indicador de pagamento representa uma anomalia de cobertura, não um comportamento observado do cliente ou vendedor.

Na fonte local, os missing de distância atingem 476 pedidos (0,4934%) e os de pagamento atingem um pedido. Esse pedido tem status `delivered`, três itens e valor de itens mais frete de 143,46; portanto, a ausência do registro de pagamento não é tratada como valor zero. A tabela preserva o pedido e cria os indicadores `distance_missing` e `payment_record_missing`; a imputação é ajustada somente no treino. A tabela de pedidos continua com 96.470 linhas e `order_id` único.

**Estado:** implementado e usado em todos os notebooks de modelagem; `production_eligible = false`.

## Decisão registrada para a EDA geográfica

Atualização: implementado no notebook 03, com a demonstração Haversine e, desde a Fase 9, o grafo de rotas entre UFs (P-09). Registro original: quando a etapa geográfica for iniciada, apresentaremos um exemplo de origem vendedor–destino cliente e, se tecnicamente viável, uma rota visual pelas vias apenas para contextualizar o problema. Essa rota será identificada como ilustrativa: não temos a rota histórica efetivamente percorrida pelo pedido.

A medida principal para distância será a fórmula de Haversine, calculada a partir das coordenadas aproximadas de origem e destino. Ela será usada por ser determinística, reproduzível, independente de chave de API e compatível com a natureza histórica da fonte.

Não usaremos APIs de roteamento ou tráfego. Embora elas possam calcular rotas atuais, não reconstróem o congestionamento real dos pedidos de 2016–2018 e exigem credenciais, billing e dependência externa. A demonstração visual ficará limitada aos pontos de origem e destino e à linha Haversine, sem tentar representar as vias percorridas.

Essa decisão mantém separado:

- **contexto visual:** pontos de origem e destino ligados pela linha Haversine;
- **medida analítica:** distância Haversine vendedor–cliente;
- **limite:** ausência de tráfego, hubs, modal e rota histórica observada.

## Anomalias classificadas na auditoria inicial

| Regra de auditoria | Casos | Decisão atual |
|---|---:|---|
| Despacho anterior à compra | 166 | Sensibilidade; manter a linha, excluir o evento do histórico as-of |
| Entrega ao cliente anterior ao despacho | 23 | Sensibilidade; manter a linha e reportar a inconsistência |
| Pedido não entregue com data real preenchida | 6 | Excluir do supervisionado até resolver a semântica |
| Status diferente de `shipped`/`delivered` com despacho preenchido | 75 | Sensibilidade; não inferir entrega concluída |
| Data estimada anterior à compra | 0 | Excluir se aparecer; viola o contrato temporal |
| Entrega real anterior à compra | 0 | Excluir se aparecer; é temporalmente impossível |

Os números acima foram reproduzidos diretamente por `audit_temporal_anomalies` em `src/ecom_delivery_sla/audit.py` sobre `olist_orders_dataset.csv`. A função retorna os IDs dos pedidos, uma linha por regra e pedido, e `anomaly_policy_summary` materializa a decisão e a justificativa sem alterar a fonte.

## Critérios de aceitação do contrato

O gate da Fase 2 será considerado aprovado somente quando houver evidência para todos os critérios:

1. A população supervisionada puder ser reconstruída deterministicamente.
2. A definição do alvo estiver coberta por testes unitários.
3. A tabela analítica tiver exatamente uma linha por `order_id`.
4. Cardinalidades e coberturas de cada junção estiverem documentadas.
5. A política para múltiplos vendedores estiver aprovada e testada.
6. Cada feature tiver classificação de disponibilidade no momento da compra.
7. As anomalias tiverem tipo, contagem, tratamento e justificativa registrados.
8. A divisão temporal e o isolamento contra informação futura estiverem testados.
9. A execução repetida com a mesma fonte produzir os mesmos resultados.
10. A fonte original permanecer imutável e sua proveniência continuar registrada.

## Estado do gate

**Aprovado em 2026-11-16 para o contrato de dados.**

Os dez critérios têm evidência. A população supervisionada e o alvo por dia de calendário são reconstruídos por `targets.py`, com testes unitários. A tabela analítica tem uma linha por `order_id`, verificada nos testes de `features.py`. As coberturas das junções estão documentadas no inventário do notebook 01 e na cobertura geográfica do notebook 03. A política de múltiplos vendedores foi implementada, explicada no notebook 04 e testada (P-03). O catálogo classifica todas as features pela disponibilidade no momento da compra, sem itens pendentes. As anomalias têm tipo, contagem, tratamento e justificativa (`audit.py`). Os folds expanding com embargo e a rede as-of têm testes contra informação futura. A repetição com a mesma fonte é controlada pelo fingerprint de `run_cache.py` e pelo congelamento das avaliações F1 e F2, e a proveniência da fonte é verificada por SHA-256 em `scripts/verify_data.py`.

A seleção de modelos (P-04), a avaliação final única (F1, P-05) e a análise complementar (F2, P-06) foram concluídas, e os notebooks 01 a 11 foram executados em português e inglês, com o teste de saídas passando. A aprovação do contrato de dados não altera `production_eligible = false`, que depende do resultado do modelo: o F1 foi inconclusivo.
