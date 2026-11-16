# Pré-análise da fonte Olist

## Estado da decisão

**Fonte viável provisoriamente para o MVP:** sim.

**Modelagem exploratória:** executada nos notebooks 06–08, posteriormente reclassificada como não comparável porque os gates de contrato ainda estavam abertos.

**Modelagem contratual:** não iniciada.

**Optuna, teste final e elegibilidade operacional:** ainda não aplicáveis.

**production_eligible:** `false`.

A fonte Olist sustenta inicialmente o problema de atraso de entrega em e-commerce brasileiro e permite construir uma rede aproximada entre vendedores e destinos. A aprovação do MVP permanece condicionada à resolução das decisões listadas neste documento.

## Evidência da auditoria

- Fonte: Brazilian E-Commerce Public Dataset by Olist.
- Diretório auditado: `data/raw/olist/csv/`.
- ZIP e hash da fonte: registrados em `data/raw/olist/README.md`.
- Data da auditoria: 2026-10-30.
- Runtime usado para leitura: `C:\Users\enior\miniforge3\python.exe`.
- As tabelas brutas não foram alteradas.

## Inventário e relacionamentos

| Tabela | Linhas | Chave ou observação |
|---|---:|---|
| `olist_orders_dataset` | 99.441 | `order_id` único |
| `olist_order_items_dataset` | 112.650 | 98.666 pedidos referenciados |
| `olist_customers_dataset` | 99.441 | `customer_id` único |
| `olist_sellers_dataset` | 3.095 | `seller_id` único |
| `olist_products_dataset` | 32.951 | `product_id` único |
| `olist_order_payments_dataset` | 103.886 | 2.961 pedidos com mais de um pagamento |
| `olist_order_reviews_dataset` | 99.224 | 547 pedidos com mais de uma avaliação |
| `olist_geolocation_dataset` | 1.000.163 | 19.015 prefixos de CEP; múltiplas linhas por prefixo |
| `product_category_name_translation` | 71 | categorias traduzidas |

Foram observadas as seguintes integridades:

- Todas as 99.441 ordens encontram correspondência em clientes.
- Nenhum item aponta para pedido, vendedor ou produto inexistente.
- Não foram observadas linhas duplicadas nas tabelas auditadas.
- 775 pedidos não possuem item associado, mas nenhum deles pertence à população supervisionável válida.
- 1 pedido não possui registro de pagamento.

## População e alvo

Foi considerada população supervisionável provisória a interseção de:

```text
order_status == "delivered"
order_delivered_customer_date não nula
order_estimated_delivery_date não nula
```

Resultado:

- pedidos com status `delivered`: 96.478;
- entregas sem data real: 8;
- população supervisionável provisória: 96.470;
- violações pela regra fracionária histórica: 7.826;
- taxa pela regra fracionária histórica: 8,11%;
- nova taxa contratada por dia de calendário: 6.534 violações, 6,77%.

O alvo foi calculado sem arredondamento nesta auditoria:

```text
delivery_delay_days =
    (order_delivered_customer_date - order_estimated_delivery_date).total_seconds() / 86.400
```

Quantis observados do atraso, em dias:

| Quantil | Atraso |
|---:|---:|
| 0% | -146,016 |
| 1% | -35,145 |
| 25% | -16,244 |
| 50% | -11,948 |
| 75% | -6,390 |
| 95% | 3,817 |
| 99% | 18,940 |
| 100% | 188,975 |

O resultado indica forte concentração de entregas antes da data estimada e uma cauda de atrasos extremos. Esses casos não devem ser removidos automaticamente; primeiro devem ser auditados contra as datas originais.

## Anomalias temporais encontradas

Foram encontradas:

- 166 datas de despacho anteriores à compra;
- 23 datas de entrega ao cliente anteriores ao despacho;
- 6 pedidos não entregues com data real de entrega preenchida;
- 75 pedidos não classificados como `shipped` ou `delivered` com data de despacho preenchida;
- nenhum caso de data estimada anterior à compra;
- nenhum caso de data real anterior à compra.

Esses registros exigem uma política contratual. Não se deve corrigi-los silenciosamente nem incorporá-los ao treinamento sem rastreabilidade.

## Viabilidade da camada de rede

É possível construir uma rede aproximada com:

```text
vendedor -> destino geográfico
```

Evidências:

- 1.275 pedidos supervisionáveis possuem mais de um vendedor;
- 516 pedidos supervisionáveis possuem vendedores associados a mais de um estado de origem;
- 95.994 de 96.470 pedidos supervisionáveis possuem coordenadas aproximadas de origem e destino;
- 3.088 de 3.095 vendedores possuem correspondência geográfica;
- 99.163 de 99.441 clientes possuem correspondência geográfica;
- 22 estados de origem e 27 estados de destino aparecem na agregação observada;
- a distância aproximada por coordenadas medianas de prefixos de CEP é tecnicamente calculável.

A geolocalização possui múltiplas linhas por prefixo de CEP. Nesta auditoria, foi usada a mediana das coordenadas e o estado modal apenas para avaliar cobertura. Essa regra ainda não foi aprovada como contrato definitivo.

## Respostas provisórias às perguntas bloqueadoras

| Pergunta | Resultado da pré-análise | Estado |
|---:|---|---|
| 1 | Usar `delivered` com datas válidas parece viável. Os 8 entregues sem data real devem ser separados. | Provisório |
| 2 | Manter o atraso contínuo em dias fracionários e acrescentar o atraso inteiro por dia de calendário. | Aprovado |
| 3 | O contrato principal será a previsão no momento da compra. | Aprovado |
| 4 | A tabela analítica deve ter uma linha por pedido. Pagamentos e itens precisam ser agregados. | Provisório |
| 5 | Distância por CEP e Haversine é viável para 95.994 pedidos válidos. | Provisório |
| 6 | Rede vendedor–destino é viável. A granularidade final ainda precisa ser aprovada. | Aberto |
| 7 | Métricas de atraso da rede só poderão ser usadas como histórico anterior ao pedido. | Regra obrigatória |
| 8 | Há dados suficientes para janelas temporais, mas a escolha entre janela cumulativa e móvel permanece aberta. | Aberto |
| 9 | A série cobre compras de setembro de 2016 a outubro de 2018, com entregas válidas principalmente em 2017 e 2018. | Aberto |
| 10 | Cancelados, indisponíveis e pedidos sem entrega devem ficar fora do alvo supervisionado e ser analisados separadamente. | Provisório |

## Variáveis pós-evento ou de uso restrito

As seguintes informações não devem ser usadas como preditores do atraso no momento da compra:

- `order_delivered_carrier_date`;
- `order_delivered_customer_date`;
- `order_status` quando representar evolução posterior do pedido;
- avaliações e datas de avaliação;
- qualquer variável agregada com entrega real ou atraso do próprio pedido;
- métricas da rede calculadas usando pedidos futuros ou o próprio período de avaliação.

Informações de itens, produtos, pagamentos, vendedor, cliente, CEP e data da compra são candidatas a variáveis permitidas, mas a disponibilidade operacional de cada campo ainda precisa ser registrada em um contrato de features.

## Pendências antes do MVP

1. Inspecionar individualmente as anomalias temporais.
2. Fixar o ponto de corte de disponibilidade das features.
3. Definir o tratamento de pedidos com múltiplos vendedores.
4. Aprovar a granularidade do grafo: vendedor–cliente, vendedor–CEP, vendedor–estado ou vendedor–região.
5. Definir a regra de agregação da geolocalização por prefixo de CEP.
6. Definir as datas exatas de treino, validação e teste final.
7. Criar a política de cold start para vendedores, destinos e rotas não observados no histórico.
8. Definir limites para auditoria dos atrasos extremos sem removê-los arbitrariamente.

Até que as pendências restantes sejam resolvidas, a fonte está aprovada apenas para auditoria e reconstrução do contrato, não para uma nova competição de estimators, Optuna ou publicação.
