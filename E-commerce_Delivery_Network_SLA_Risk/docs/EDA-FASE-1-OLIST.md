# EDA inicial — atraso e risco de SLA no Olist

**Status:** exploração descritiva inicial. Houve modelagem exploratória nos notebooks 06–08, mas seus resultados foram reclassificados como não comparáveis porque os contratos de alvo, rede e validação ainda estavam abertos. Nenhuma variável pós-evento foi aprovada como feature.

**Fonte:** Brazilian E-Commerce Public Dataset by Olist, com nove tabelas relacionais preservadas em `data/raw/olist/csv/`.

## Pergunta desta etapa

Antes de construir a tabela analítica e os modelos, queremos saber se o atraso de entrega apresenta variação suficiente para justificar uma análise de regressão e uma classificação complementar de violação de SLA.

O alvo contratado para a próxima execução é:

```text
delivery_delay_days = data_entrega_real - data_entrega_estimada
delivery_delay_calendar_days = data_entrega_real.normalize() - data_entrega_estimada.normalize()
sla_breach = delivery_delay_calendar_days > 0
```

A população supervisionada é formada por pedidos com `order_status == delivered`, data real de entrega preenchida e data estimada preenchida.

## Escopo mensurável e limites da fonte

O EDA mede diretamente o cumprimento do prazo de entrega no nível do pedido:

- `delivery_delay_days`: diferença entre a data real e a data estimada de entrega;
- `sla_breach`: indicador de atraso positivo;
- prazo prometido e prazo realizado desde a compra, como medidas descritivas;
- taxas e distribuições por período, destino e características agregadas do pedido.

Também podem ser construídas variáveis auxiliares, como distância aproximada entre vendedor e cliente, quantidade de vendedores e indicadores históricos de despacho. Essas variáveis são proxies e não representam a rota real nem um SLA individual do vendedor.

O dataset não permite observar diretamente:

- entrega individual por item ou vendedor;
- localização de transportadoras e hubs;
- rota efetivamente percorrida ou modal de transporte;
- motivo operacional do atraso;
- capacidade, estoque ou eventos de rastreamento intermediários.

Assim, o projeto analisa risco histórico de descumprimento do prazo prometido, não a reconstrução completa da operação logística. Diferenças entre grupos serão tratadas como associações descritivas, sem afirmações causais automáticas.

## Retrato do alvo

| Medida | Resultado |
|---|---:|
| Pedidos supervisionáveis | 96.470 |
| Violações de SLA pela regra de calendário | 6.534 |
| Taxa de violação pela regra de calendário | 6,77% |
| Violações pela regra fracionária anterior (histórico) | 7.826 |
| Taxa pela regra fracionária anterior (histórico) | 8,11% |
| Mediana do atraso | -11,948 dias |
| Percentil 95 do atraso | 3,817 dias |
| Percentil 99 do atraso | 18,940 dias |
| Atraso maior que 30 dias | 360 |
| Antecipação menor que -30 dias | 2.352 |

O atraso é concentrado em valores negativos. Em termos descritivos, a maior parte dos pedidos foi entregue antes da data estimada, enquanto uma minoria forma uma cauda de atrasos positivos e extremos. Isso torna inadequado tratar a média isoladamente ou remover extremos sem auditoria.

O alvo binário por calendário tem desbalanceamento relevante, mas não extremo: aproximadamente um em cada quinze pedidos supervisionáveis viola o SLA. A classificação deverá usar métricas que não dependam apenas de acurácia, como precisão, recall, F1, average precision e calibração. As taxas e os agrupamentos abaixo ainda contêm resultados da regra fracionária anterior e serão reprocessados antes de qualquer conclusão.

## Tempo prometido versus tempo realizado

Entre os pedidos supervisionáveis:

- o prazo prometido mediano, entre compra e data estimada, é de aproximadamente 23,23 dias;
- o prazo realizado mediano, entre compra e entrega, é de aproximadamente 10,22 dias;
- a correlação descritiva entre atraso relativo e prazo prometido é aproximadamente -0,50.

O prazo prometido é um candidato a feature somente se confirmarmos que a data estimada estava disponível no momento da compra. O prazo realizado é informação pós-evento e não poderá entrar no modelo de risco.

## Variação temporal

A taxa mensal de violação não é estável. Entre os meses com volume suficiente, observamos, por exemplo:

| Mês da compra | Pedidos | Taxa de violação |
|---|---:|---:|
| 2017-11 | 7.288 | 14,31% |
| 2018-01 | 7.069 | 6,56% |
| 2018-02 | 6.555 | 15,99% |
| 2018-03 | 7.003 | 21,36% |
| 2018-04 | 6.798 | 5,31% |
| 2018-06 | 6.096 | 1,36% |
| 2018-08 | 6.351 | 10,39% |

Essas diferenças são sinais para investigação, não explicações. Podem refletir sazonalidade, mudanças operacionais, composição de rotas, disponibilidade de vendedores, qualidade do registro ou efeitos de truncamento do período. A divisão de treino, validação e teste deverá ser temporal, e o último intervalo de tempo não deve ser misturado ao treino.

Os primeiros meses têm poucos pedidos entregues. Suas taxas não devem ser comparadas diretamente com meses de maior volume sem intervalos de incerteza ou um limite mínimo de observações.

## Variação geográfica descritiva

Agrupando por estado do cliente e mantendo apenas estados com pelo menos 500 pedidos supervisionáveis, as maiores taxas observadas foram:

| Estado do cliente | Pedidos | Taxa de violação |
|---|---:|---:|
| MA | 717 | 19,67% |
| CE | 1.279 | 15,32% |
| BA | 3.256 | 14,04% |
| RJ | 12.350 | 13,47% |

Entre as menores taxas:

| Estado do cliente | Pedidos | Taxa de violação |
|---|---:|---:|
| PR | 4.923 | 5,00% |
| MG | 11.354 | 5,61% |
| SP | 40.494 | 5,89% |
| MT | 886 | 6,77% |

O contraste geográfico justifica investigar distância, origem do vendedor, rota e composição de pedidos. Ele não sustenta, sozinho, uma afirmação de que o estado causa o atraso. A análise deverá controlar volume, período, categoria, múltiplos vendedores e possíveis diferenças de composição.

Para a futura EDA geográfica, a distância principal será calculada por Haversine entre as coordenadas aproximadas do vendedor e do cliente. Uma rota visual pelas vias poderá ser apresentada apenas como contexto; ela não representará a rota histórica real nem será usada como feature de tráfego. APIs de mapas atuais não serão usadas para inferir o congestionamento dos pedidos históricos.

O notebook geográfico implementa essa demonstração para um único pedido apresentando os pontos de origem e destino e a linha Haversine. Não haverá chamada a APIs de rota ou tráfego: além de não reconstruírem as condições históricas, elas introduziriam credenciais, billing e dependência externa desnecessários.

## Categorias de produto: sinal preliminar

Como exploração inicial, cada pedido recebeu a categoria modal entre seus itens. Essa é uma regra provisória e não deve ser tratada como contrato final para pedidos com múltiplas categorias.

Entre categorias com pelo menos 1.000 pedidos, as maiores taxas descritivas foram `eletronicos` (9,84%), `moveis_escritorio` (9,24%) e `bebes` (9,23%). As menores foram `malas_acessorios` (5,45%), `pet_shop` (6,31%) e `fashion_bolsas_e_acessorios` (6,67%). A categoria desconhecida representa pedidos com informação de produto ausente ou não traduzida e deve permanecer identificada.

Essas diferenças são exploratórias. A regra de categoria dominante pode perder informação e produzir uma atribuição arbitrária para pedidos heterogêneos.

## Geografia e composição do pedido

A geolocalização aproximada cobre 99,51% dos pedidos supervisionados. A mediana da distância vendedor–cliente, calculada por Haversine entre coordenadas agregadas por prefixo de CEP, é aproximadamente 433,83 km. A fórmula usa `R = 6.371,0088 km` e calcula a distância do arco terrestre entre origem e destino. Essa medida representa distância geográfica em linha reta, não rota rodoviária, tráfego histórico ou trajeto efetivamente percorrido.

| Faixa de distância aproximada | Pedidos | Taxa de violação |
|---|---:|---:|
| 0–100 km | 17.823 | 6,43% |
| 100–300 km | 13.429 | 6,37% |
| 300–600 km | 31.483 | 7,77% |
| 600–1.000 km | 17.919 | 8,45% |
| acima de 1.000 km | 15.340 | 11,82% |

Há uma associação descritiva entre faixas mais distantes e maior taxa de violação, mas o padrão pode refletir estado, período, categoria, vendedor e composição do pedido. Não é uma conclusão causal.

Também foram calculadas quantidade de itens, produtos distintos, categorias distintas e vendedores por pedido. Pedidos com um produto apresentaram 8,22% de violação; os grupos com dois, três e quatro produtos apresentaram 5,06%, 4,03% e 1,45%. Pedidos com dois vendedores somaram 1.216 observações e apresentaram 1,48%; pedidos com três ou mais vendedores foram raros.

Essas taxas menores em pedidos mais complexos não significam que a complexidade reduza o risco. Os grupos são desbalanceados, podem ter composição operacional distinta e a categoria pode estar ausente em parte dos produtos. As variáveis serão mantidas como hipóteses explicativas para a tabela analítica, não como conclusões isoladas.

## Anomalias que permanecem abertas

Na população supervisionada foram observados:

- 23 pedidos em que a data de entrega ao cliente antecede a data de entrega ao transportador;
- 1 pedido sem data de entrega ao transportador;
- nenhum caso de aprovação anterior à compra;
- nenhum caso de entrega ao cliente anterior à compra.

Esses registros não serão corrigidos silenciosamente. A política para datas inconsistentes será definida no contrato de dados e os casos serão mantidos em uma auditoria separada.

## O que o EDA já responde

1. O Olist permite definir um alvo contínuo de atraso e um alvo binário de violação.
2. A população supervisionada é grande o suficiente para uma exploração estatística e preditiva inicial.
3. O risco de violação é desigual no tempo, por estado e, provisoriamente, por categoria.
4. A assimetria e a cauda do atraso exigem métricas robustas e auditoria de extremos.
5. O problema não deve ser tratado como causal apenas porque há diferenças entre grupos.

## O que ainda não está respondido

- qual granularidade de destino será usada no grafo;
- como tratar pedidos com múltiplos vendedores;
- quais métricas históricas da rede estavam disponíveis antes de cada pedido;
- qual política será aplicada a vendedores, destinos e rotas sem histórico;
- quais datas formam treino, validação e teste final;
- se a data estimada estava disponível exatamente no momento da compra;
- qual especificação OLS será interpretável e quais modelos preditivos serão competitivos.

## Próxima entrega

Construir a tabela analítica de pedidos, com uma linha por `order_id`, agregando itens, pagamentos, vendedores, produtos e localização com regras explícitas. Antes de criar features de rede, será produzido um inventário de disponibilidade no momento da compra e uma checagem de cardinalidade.

Até que esses contratos sejam aprovados, `production_eligible = false`.
