# Dicionário de dados — Olist

Este documento descreve os CSVs locais da fonte Olist usados no projeto. Os nomes das colunas são preservados exatamente como aparecem na fonte, inclusive `product_name_lenght` e `product_description_lenght`.

As definições abaixo são operacionais, baseadas nos nomes das colunas, nos valores observados e nas relações entre tabelas. Quando a fonte não fornece um contrato formal, a definição deve ser validada antes da modelagem.

## Visão geral

| Dataset | Linhas auditadas | Granularidade principal | Chave ou relação |
|---|---:|---|---|
| `olist_orders_dataset.csv` | 99.441 | Pedido | `order_id` |
| `olist_order_items_dataset.csv` | 112.650 | Item de pedido | `order_id` + `order_item_id` |
| `olist_customers_dataset.csv` | 99.441 | Associação pedido–cliente | `customer_id` |
| `olist_sellers_dataset.csv` | 3.095 | Vendedor | `seller_id` |
| `olist_products_dataset.csv` | 32.951 | Produto | `product_id` |
| `olist_order_payments_dataset.csv` | 103.886 | Parcela/meio de pagamento do pedido | `order_id` + `payment_sequential` |
| `olist_order_reviews_dataset.csv` | 99.224 | Avaliação associada ao pedido | `review_id` e `order_id` |
| `olist_geolocation_dataset.csv` | 1.000.163 | Registro geográfico por prefixo de CEP | `geolocation_zip_code_prefix` não é único |
| `product_category_name_translation.csv` | 71 | Tradução de categoria | `product_category_name` |

## 1. `olist_orders_dataset.csv`

Uma linha representa um pedido e seu ciclo de vida logístico. É a tabela principal para o alvo de entrega.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `order_id` | Identificador único do pedido. | Chave da unidade de análise. |
| `customer_id` | Identificador da associação entre o pedido e o cliente. Pode se repetir para pedidos do mesmo cliente. | Junção com clientes; candidato a feature somente com contrato de disponibilidade. |
| `order_status` | Estado registrado do ciclo do pedido, como `delivered`, `shipped`, `canceled`, `unavailable` ou `processing`. | Define a população, mas representa evolução do pedido e não deve ser usado livremente como preditor. |
| `order_purchase_timestamp` | Data e hora em que o pedido foi realizado. | Ponto temporal principal e possível corte de features. |
| `order_approved_at` | Data e hora da aprovação do pedido/pagamento. | Pode ser usada somente se o momento de decisão incluir a aprovação. |
| `order_delivered_carrier_date` | Data em que o pedido foi registrado como entregue ao transportador. Existe uma data por pedido, não por vendedor. | Pós-evento para o modelo de atraso; útil na auditoria de despacho. |
| `order_delivered_customer_date` | Data em que o pedido foi registrado como entregue ao cliente. | Pós-evento; compõe o alvo e não pode ser feature. |
| `order_estimated_delivery_date` | Data estimada/prometida para entrega ao cliente. | Define o prazo do SLA operacionalizado; uso como feature exige contrato explícito. |

### Relação com o SLA

```text
delivery_delay_days = order_delivered_customer_date
                      - order_estimated_delivery_date
sla_breach = delivery_delay_days > 0
```

O dataset não contém o contrato comercial completo de SLA; essa é uma definição analítica baseada na data estimada e na data real de entrega.

## 2. `olist_order_items_dataset.csv`

Uma linha representa um item dentro de um pedido. É a tabela que revela que um pedido pode envolver vários produtos e vendedores.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `order_id` | Identificador do pedido ao qual o item pertence. | Chave de junção com `orders`; precisa ser agregado para voltar ao nível de pedido. |
| `order_item_id` | Número sequencial do item dentro do pedido. | Identifica a posição do item; junto com `order_id`, forma a chave da linha. |
| `product_id` | Identificador do produto vendido. | Junção com produtos; pode haver vários produtos no mesmo pedido. |
| `seller_id` | Identificador do vendedor responsável pelo item. | Permite construir relação pedido–vendedor e indicadores históricos. |
| `shipping_limit_date` | Prazo-limite registrado para o vendedor despachar o item. | Candidato a feature e auditoria de despacho; não é a data real de despacho individual. |
| `price` | Preço do item, sem ser necessariamente o total do pedido. | Agregação por pedido; possível peso econômico. |
| `freight_value` | Valor do frete associado ao item. | Agregação por pedido; possível sinal de distância/custo logístico. |

Um pedido pode ter vários vendedores. Porém, como `order_delivered_carrier_date` e `order_delivered_customer_date` são do pedido inteiro, não é possível atribuir individualmente o atraso de entrega a cada vendedor com esta fonte.

## 3. `olist_customers_dataset.csv`

Uma linha representa a associação de um `customer_id` a uma localização de destino. O `customer_id` é relacionado ao pedido; `customer_unique_id` representa a identidade persistente do cliente entre pedidos.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `customer_id` | Identificador associado ao pedido no dataset. | Chave de junção com `orders`. |
| `customer_unique_id` | Identificador do cliente que pode permanecer entre diferentes pedidos. | Histórico de recorrência; requer cuidado para não usar informação futura do mesmo cliente. |
| `customer_zip_code_prefix` | Prefixo do CEP do destino. | Junção aproximada com geolocalização. |
| `customer_city` | Cidade do destino informada na base. | Feature geográfica categórica, com controle de cardinalidade. |
| `customer_state` | Estado do destino. | Feature geográfica e agrupamento descritivo. |

## 4. `olist_sellers_dataset.csv`

Uma linha representa um vendedor cadastrado e sua localização declarada.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `seller_id` | Identificador único do vendedor. | Chave de junção com itens e base de vendedor. |
| `seller_zip_code_prefix` | Prefixo do CEP de origem do vendedor. | Geolocalização aproximada da origem. |
| `seller_city` | Cidade declarada do vendedor. | Descrição e feature geográfica. |
| `seller_state` | Estado declarado do vendedor. | Distância/origem e agrupamento descritivo. |

O vendedor pode aparecer em muitos pedidos. Indicadores históricos de confiabilidade devem usar somente pedidos anteriores à compra avaliada.

## 5. `olist_products_dataset.csv`

Uma linha representa um produto cadastrado. As medidas físicas e os textos são atributos do catálogo.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `product_id` | Identificador único do produto. | Chave de junção com itens. |
| `product_category_name` | Categoria do produto em português, quando disponível. | Categoria para agregação e análise. |
| `product_name_lenght` | Quantidade de caracteres do nome do produto. O nome da coluna contém o erro ortográfico original `lenght`. | Feature textual simples ou controle de qualidade. |
| `product_description_lenght` | Quantidade de caracteres da descrição. O nome da coluna preserva `lenght`. | Feature de catálogo; pode ter ausências. |
| `product_photos_qty` | Quantidade de fotos cadastradas para o produto. | Feature de catálogo. |
| `product_weight_g` | Peso do produto em gramas. | Feature logística; deve ser validada antes de agregação. |
| `product_length_cm` | Comprimento do produto em centímetros. | Feature dimensional. |
| `product_height_cm` | Altura do produto em centímetros. | Feature dimensional. |
| `product_width_cm` | Largura do produto em centímetros. | Feature dimensional. |

## 6. `olist_order_payments_dataset.csv`

Uma linha representa uma ocorrência de pagamento associada ao pedido. Um pedido pode ter mais de um registro de pagamento.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `order_id` | Pedido associado ao pagamento. | Junção e agregação. |
| `payment_sequential` | Sequência do pagamento dentro do pedido. | Identifica múltiplos pagamentos. |
| `payment_type` | Meio de pagamento, como cartão de crédito, boleto, voucher ou débito. | Feature categórica, se disponível no momento do corte. |
| `payment_installments` | Número de parcelas do pagamento. | Feature numérica. |
| `payment_value` | Valor associado ao registro de pagamento. | Soma/estatísticas por pedido; validar relação com preço e frete. |

## 7. `olist_order_reviews_dataset.csv`

Uma linha representa uma avaliação associada a um pedido. Há pedidos com mais de uma avaliação.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `review_id` | Identificador da avaliação. | Chave da avaliação. |
| `order_id` | Pedido avaliado. | Junção com pedido. |
| `review_score` | Nota atribuída pelo cliente, normalmente de 1 a 5. | Pós-evento; não usar como preditor do atraso. |
| `review_comment_title` | Título textual da avaliação. | Pós-evento; análise de satisfação posterior. |
| `review_comment_message` | Texto da avaliação. | Pós-evento; análise de satisfação posterior. |
| `review_creation_date` | Data de criação da avaliação. | Pós-evento. |
| `review_answer_timestamp` | Data e hora da resposta à avaliação. | Pós-evento. |

## 8. `olist_geolocation_dataset.csv`

Uma linha representa uma observação geográfica associada a um prefixo de CEP. O prefixo não é uma chave única: há várias coordenadas para o mesmo prefixo.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `geolocation_zip_code_prefix` | Prefixo do CEP ao qual a coordenada é associada. | Chave de aproximação com clientes e vendedores, não chave única. |
| `geolocation_lat` | Latitude observada. | Cálculo de localização/distância após agregação. |
| `geolocation_lng` | Longitude observada. | Cálculo de localização/distância após agregação. |
| `geolocation_city` | Cidade associada ao registro geográfico. | Validação e descrição. |
| `geolocation_state` | Estado associado ao registro geográfico. | Validação e descrição. |

Para usar essa tabela, é necessário definir uma regra de agregação por prefixo, como mediana de latitude/longitude e moda do estado. Essa regra ainda é provisória no projeto.

## 9. `product_category_name_translation.csv`

Uma linha representa a tradução de uma categoria do português para o inglês.

| Campo | Definição operacional | Uso no projeto |
|---|---|---|
| `product_category_name` | Nome da categoria original em português. | Chave de junção com produtos. |
| `product_category_name_english` | Nome traduzido da categoria em inglês. | Rótulo para relatórios e notebooks EN-US. |

Categorias sem tradução devem permanecer identificadas como ausentes ou não traduzidas; não devem receber uma tradução inventada.

## Regras gerais de uso no projeto

- `orders` define a unidade principal do SLA.
- `order_items`, `payments` e `reviews` são tabelas de granularidade potencialmente um-para-muitos e precisam de agregação explícita.
- Campos de entrega real, despacho ao transportador, avaliação e status evolutivo são pós-evento.
- Features históricas de vendedor, rota ou destino devem ser calculadas somente com observações anteriores ao pedido atual.
- A geolocalização é aproximada e depende de uma regra de agregação por prefixo.
- O dicionário descreve a fonte; ele não aprova sozinho a tabela analítica final nem encerra o gate da Fase 2.
