# Plano do grafo de rotas origem–destino (notebook 03)

**Estado:** **aprovado em 2026-11-16** (P-09 em `REGISTRO-DE-DECISOES.md`), com as quatro propostas da seção 8. Implementado e executado em PT e EN, com insights escritos a partir das saídas.
**Origem:** pedido do responsável para mostrar, como rede, quantos pedidos saem de cada origem para cada destino e qual é a taxa de atraso de cada rota, com todas as propriedades de nós e arestas disponíveis no *hover* do Plotly.

## 1. Evidência que define o nível do grafo

A medição foi feita em 2026-11-16 sobre os 96.470 pedidos supervisionados, contados como 97.811 pares pedido–vendedor.

| Nível | Rotas | Rotas com 1 pedido | Rotas com ≥ 100 pedidos | Volume coberto por essas rotas |
|---|---:|---:|---:|---:|
| Cidade → cidade | 35.628 | 25.855 | 70 | 20,7% |
| UF → UF | 412 | 68 | 71 | 94,8% |

Entre cidades, a maioria das taxas seria calculada com um a cinco pedidos e o gráfico seria ilegível. **O grafo é construído entre UFs.** As cidades aparecem só num gráfico das maiores rotas. A origem é concentrada: 70,9% dos pares saem de SP. O grafo terá a forma de estrela centrada em SP, e o texto apresenta isso como achado.

## 2. Definições

| Item | Definição |
|---|---|
| População | Pedidos supervisionados (P-02), todo o período, como nos demais gráficos descritivos dos notebooks 02 e 03 |
| Unidade da aresta | Par pedido–vendedor. Um pedido com dois vendedores conta uma vez em cada rota, e o desfecho do pedido é atribuído a cada par. Isso é declarado no texto, porque é a mesma limitação da agregação discutida no notebook 04. |
| Nó | UF. A coordenada é a mediana da latitude e da longitude da tabela de geolocalização da Olist para a UF, sem dado externo. |
| Aresta | Par ordenado (UF de origem, UF de destino), com direção. Rotas dentro da mesma UF (laços, como SP → SP) não são desenhadas como linhas e aparecem como propriedades do nó. |
| Taxa de atraso | Proporção de pares com `sla_breach = 1`, com IC 95% de Wilson (Wilson, 1927) |
| Volume mínimo | Arestas com pelo menos 100 pares (parâmetro `min_orders`, padrão 100). Na matriz, células com menos de 30 pares ficam em branco. |

## 3. Gráficos (Plotly, com *hover* completo)

**3.1 Mapa da rede por UF (`go.Scattergeo`, sem dado externo nem tiles).**
- **Arestas:**
  - Cada aresta é uma curva levemente arqueada entre os nós, para que A → B e B → A não se sobreponham.
  - A espessura é proporcional ao volume, e a cor à taxa de atraso, numa escala contínua com barra de cores.
  - Como linhas do Scattergeo não têm *hover* ao longo do traço, cada aresta ganha um marcador no ponto médio. Ele indica a direção com uma seta, se a versão do Plotly suportar `angleref="previous"`; senão, com um triângulo.
  - O marcador carrega o *hover* da aresta: origem, destino, pares, violações, taxa com IC 95%, participação no volume total, mediana da distância Haversine, mediana do prazo prometido e mediana do prazo realizado.
- **Nós:**
  - Círculo sobre o centro da UF, com rótulo da sigla, tamanho pelo volume total (saída + entrada) e cor pela taxa de atraso dos pedidos recebidos.
  - O *hover* do nó mostra: pares enviados e recebidos; taxa de atraso de saída e de entrada; pares e taxa dentro da própria UF; número de vendedores e de clientes; grau de saída e de entrada (número de rotas acima do volume mínimo) e força (volume ponderado).
- **Filtro:** um menu do Plotly (`updatemenus`) alterna o volume mínimo entre 100, 300 e 1.000 pares, sem reexecutar a célula.

**3.2 Matriz origem × destino (`go.Heatmap`).** As linhas são as 8 UFs de origem com maior volume, e as colunas são as 27 UFs de destino. A cor é a taxa de atraso, o número de pares fica escrito na célula, e células com menos de 30 pares ficam em branco. O *hover* traz as mesmas propriedades da aresta do mapa.

**3.3 Maiores rotas entre cidades (`go.Bar` horizontal).** As 15 rotas cidade → cidade de maior volume, com a taxa de atraso e barra de erro do IC de Wilson. O *hover* mostra pares, violações, IC e mediana da distância. É aqui que entra o exemplo São Paulo → Belo Horizonte.

## 4. Onde fica o código

As funções ficam em `src/ecom_delivery_sla_viz/routes.py`, um módulo novo do pacote de apresentação. A razão é a mesma da frente F do P-07: qualquer arquivo novo em `src/ecom_delivery_sla/` muda o fingerprint e invalidaria o cache do notebook 07. O módulo não importa `ecom_delivery_sla`. Ele recebe as tabelas que o notebook 03 já carrega (pedidos supervisionados, itens, vendedores, clientes e geolocalização) e usa só pandas, numpy e plotly.

Funções:
- `route_pairs(supervised, items, sellers, customers, geolocation)`: pares pedido–vendedor com UF e cidade de origem e destino, coordenadas, distância Haversine, prazos e `sla_breach`;
- `route_table(pairs, level="state" | "city", min_orders=...)`: tabela de arestas;
- `node_table(pairs, edges)`: tabela de nós;
- `wilson_interval(breaches, n)`;
- `plot_route_network(...)`, `plot_route_matrix(...)` e `plot_top_city_routes(...)`, com rótulos PT/EN.

A distância Haversine é recalculada no módulo com a mesma fórmula do notebook 03, e um teste compara o resultado com `ecom_delivery_sla.geo.haversine_km`. A importação acontece só no teste, para manter o isolamento do pacote.

## 5. Notebook 03

- **Onde entra:** três blocos novos, no padrão Markdown → código → insight, entre a demonstração Haversine e "Limites e próximo passo".
- **Texto de abertura de cada bloco:** explica a unidade (par pedido–vendedor), o volume mínimo, o IC de Wilson e como usar o *hover*. Ele cita (Wilson, 1927), e a referência entra na célula de referências do notebook.
- **Insights:** são escritos depois da execução pelo responsável, só com números produzidos pelas células, como foi feito no notebook 10.
- **Cuidado declarado no texto:** a taxa de uma rota mistura destino e período, porque os picos de atraso de novembro de 2017 e de fevereiro e março de 2018 não se distribuem igualmente entre as rotas. A leitura é descritiva, e o grafo é a versão visual das features de rede as-of vendedor → UF usadas no modelo.
- **Paridade:** PT e EN com código idêntico, exceto `LANG`.

## 6. Testes

Os testes usam dados sintéticos em `tests/test_viz_routes.py` e verificam:
- contagem por par pedido–vendedor em pedidos com vários vendedores;
- a taxa e o IC de Wilson contra valores calculados à mão;
- o filtro de volume mínimo;
- a separação dos laços dentro da mesma UF;
- os graus e a força dos nós;
- a distância igual à de `haversine_km`;
- figuras com os traços esperados e com *hover* contendo todos os campos;
- nenhuma mutação das entradas.

O teste de isolamento do pacote continua valendo. O teste de contrato de citações passa a exigir a referência de Wilson no 03.

## 7. Execução e resultado

- **Implementação:** feita no pacote de apresentação, sem tocar no cache do notebook 07 nem em `src/ecom_delivery_sla/`.
- **Execução:** o notebook 03 foi executado em PT e EN; a geografia foi recalculada sem cache.
- **Insights:** foram escritos a partir das saídas executadas.

## 8. Decisões encerradas no P-09

1. Foi adotado o nível UF para o grafo, com cidades apenas no gráfico das maiores rotas.
2. Foi adotada a unidade par pedido–vendedor, preservando a informação de pedidos com múltiplos vendedores.
3. Foram adotados os volumes mínimos de 100 pares no mapa e 30 na matriz, com o menu 100/300/1.000.
4. O grafo foi incorporado ao notebook 03, depois da demonstração Haversine.
