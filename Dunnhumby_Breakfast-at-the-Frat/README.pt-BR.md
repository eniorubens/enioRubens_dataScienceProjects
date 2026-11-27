# Precificação e promoção no varejo com modelos multinível

[English](README.md) · [Relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Dunnhumby_Breakfast-at-the-Frat/)

> A seleção de modelos e a avaliação final do P1 foram concluídas, assim como as análises que usaram apenas o treino nos notebooks 05 e 06.

## Pergunta de negócio

Como **preço** e **promoção** se relacionam com as vendas em unidades entre lojas e produtos? O estudo compara encarte, display e redução temporária de preço (TPR). Também compara cenários de preço e promoção e verifica com que frequência condições semelhantes aparecem nos dados. Margem, custo da promoção, ROI e recomendações de preço com base em efeitos causais ficam fora do escopo.

Uma regressão por mínimos quadrados ordinários (OLS) para cada par loja–produto pode gerar estimativas instáveis da elasticidade ao desconto, inclusive positivas. Com **agrupamento parcial (partial pooling)**, o modelo multinível usa informações do painel inteiro para estimar diferenças entre lojas e produtos. O estudo testa se isso melhora as previsões fora da amostra e se deixar a inclinação do desconto variar por loja e produto traz um ganho adicional.

## Abordagem (resumo)

- A variável resposta do modelo linear misto é `log(UNITS)`. A análise começa com modelos nulos e correlações intraclasse (ICCs), com testes de razão de verossimilhança ajustados para parâmetros na fronteira.
- Os efeitos fixos incluem profundidade do desconto, encarte, display, TPR, sazonalidade de Fourier e tendência. O preço-base entra como média do par e desvio em relação a essa média. O modelo inclui interceptos aleatórios cruzados para loja, produto e loja × produto.
- O P1 testou inclinações aleatórias do desconto por loja e produto (C1) pelo erro fora da amostra. Um teste posterior de razão de verossimilhança usou bootstrap paramétrico nas semanas de treino. Outras análises de treino testaram as interações planejadas da inclinação do desconto com segmento da loja, marca própria e categoria com correção de Holm, avaliaram a sensibilidade a choques de categoria × semana e compararam cenários de preço e promoção.
- A comparação preditiva inclui OLS por par loja–produto, OLS agrupado, modelo multinível com BLUPs e GPBoost. O P1 fixou as regras antes de ajustar os candidatos. A seleção de modelos usou duas dobras internas nas primeiras 104 semanas. Os modelos da comparação final foram ajustados nessas 104 semanas.
- As últimas 52 semanas formam uma **janela de avaliação retrospectiva**. O OLS exploratório do briefing já havia usado o painel completo, como registra a decisão D11 revisada. A especificação selecionada, os parâmetros do GPBoost e os hashes do código e do painel foram congelados antes da avaliação formal, feita uma única vez na janela final. As previsões usam os preços e as promoções observados em cada semana.
- O Python chama o `lme4` via `rpy2`. O R é instalado pelo conda-forge no ambiente conda. O `statsmodels.MixedLM` serve de conferência para o modelo nulo com intercepto aleatório por loja.
- Semanas sem registro podem indicar venda zero ou ruptura. O modelo usa apenas semanas com vendas registradas.

## Como obter os dados

A reprodução usa a planilha original da dunnhumby e o guia que acompanha os dados.

1. Cadastre-se e baixe *Breakfast at the Frat: A Time Series Analysis* em https://www.dunnhumby.com/source-files/.
2. Coloque `dunnhumby - Breakfast at the Frat.xlsx` (e o guia em PDF) em `dataset/`.

*Breakfast at the Frat* faz parte da coleção dunnhumby Source Files.

## Reprodução

~~~
conda env create -f environment.yml
conda activate retail-multilevel
python scripts/ingest.py        # Excel -> data/interim/*.parquet (~1 min, uma vez)
python scripts/build_panel.py   # regras de limpeza -> data/processed/panel.parquet
python scripts/validate_data.py # contrato de qualidade no painel real (só local)
pytest                          # só fixtures sintéticas; é o que o CI roda
~~~

Os scripts ajustam os modelos uma vez por experimento e salvam as execuções em `outputs/runs/`. As duas versões dos notebooks leem as execuções salvas.

No Windows, o `retail_multilevel.rbridge` aponta `R_HOME` para o R do ambiente conda, mesmo quando existe uma instalação de R no sistema. O notebook 01 mostra as regras de limpeza e as contagens de linhas de cada regra.

## Mapa dos notebooks

| Etapa | PT-BR | EN-US | O que apresenta |
| :--- | :---: | :---: | :--- |
| 01 — Painel e exploração | [PT-BR](notebooks/pt_br/01_eda_painel.ipynb) | [EN-US](notebooks/en_us/01_panel_eda.ipynb) | briefing do negócio, regras de limpeza, corte temporal, EDA descritiva, inclinações do desconto estimadas par a par |
| 02 — Modelo nulo | [PT-BR](notebooks/pt_br/02_modelo_nulo.ipynb) | [EN-US](notebooks/en_us/02_null_model.ipynb) | modelos nulos, ICCs, testes de razão de verossimilhança com correção de fronteira |
| 03 — Efeitos fixos | [PT-BR](notebooks/pt_br/03_efeitos_fixos.ipynb) | [EN-US](notebooks/en_us/03_fixed_effects.ipynb) | preço e promoção, cenários, sensibilidades, dependência residual |
| 04 — Protocolo P1 | [PT-BR](notebooks/pt_br/04_protocolo_p1.ipynb) | [EN-US](notebooks/en_us/04_protocol_p1.ipynb) | protocolo pré-registrado, inclinações aleatórias, avaliação final única, teste das inclinações por bootstrap paramétrico |
| 05 — Contrastes e sensibilidade | [PT-BR](notebooks/pt_br/05_contrastes_e_sensibilidade.ipynb) | [EN-US](notebooks/en_us/05_contrasts_and_sensitivity.ipynb) | contrastes planejados da inclinação do desconto (segmento, marca própria, categoria) com Holm, choques de categoria × semana |
| 06 — Cenários de preço e promoção | [PT-BR](notebooks/pt_br/06_cenarios_preco_promocao.ipynb) | [EN-US](notebooks/en_us/06_pricing_promotion_scenarios.ipynb) | oito cenários com covariância conjunta, suporte observado, dispersão da inclinação entre produtos |

## Principais resultados (protocolo P1)

Nas duas dobras internas, o C1 reduziu o erro de previsão nos pares com menos de 26 semanas de treino. No conjunto dos pares elegíveis, o C1 ficou abaixo do critério de seleção pré-registrado. O P1 selecionou o C0, com interceptos aleatórios e uma inclinação comum para o desconto. O C0 foi a especificação multinível avaliada na janela final. Ainda fica em aberto se as inclinações aleatórias ajudariam os pares com pouco histórico nessa janela.

Na janela final, o C0 teve erro menor no agregado que o OLS agrupado e o OLS por par loja–produto. Entre os pares vistos no treino, o GPBoost teve erro absoluto médio (MAE) no logaritmo das unidades vendidas 0,47% menor com peso igual por par (0,4520 ante 0,4541 do C0). Com peso igual por linha, o C0 teve o menor MAE (0,4446 ante 0,4478). O C0 também teve o menor MAE por par nos pares inéditos (0,626 ante 0,688). A incerteza das diferenças entre C0 e GPBoost ainda não foi estimada. Para os pares com pouco histórico, o notebook 04 e o registro de decisões mostram como o resultado muda conforme o conjunto comparado e a ponderação.

## Contrastes e cenários (notebooks 05 e 06, só no treino)

Após a correção de Holm, cada um dos três testes globais planejados rejeitou a hipótese de inclinação comum para o desconto. Os moderadores testados foram segmento da loja, marca própria em comparação com marca de fabricante e categoria do produto. As estimativas por categoria foram de cerca de −1,1 em BAG SNACKS a −2,4 em FROZEN PIZZA. O intercepto aleatório de categoria × semana absorveu os choques semanais comuns, com pouca mudança nos coeficientes de preço e promoção.

Com desconto de 20%, o modelo estimou vendas em unidades 119% acima de uma semana regular no cenário com display e 40% acima no cenário com TPR. A razão estimada das vendas com display sobre as vendas com TPR foi de 1,57. Os oito cenários tiveram observações comparáveis no treino. Cada cenário promocional correspondeu a 0,3% a 3% das linhas de treino.

## Inclinações aleatórias do desconto no treino (notebook 04, seção 9)

Um teste complementar depois do P1 comparou C0 e C1 nas semanas de treino. O teste ajustou os dois modelos por máxima verossimilhança e calibrou a estatística da razão de verossimilhanças com 39 réplicas de bootstrap paramétrico sob o C0. A estatística observada foi 12.974 e o maior valor simulado foi 5,18. O valor-p de Monte Carlo de 0,025 é o menor possível com esse desenho. Sob a estrutura residual do C0, o teste conjunto rejeita a hipótese de que ambas as variâncias das inclinações sejam zero. A autocorrelação residual diagnosticada no notebook 03 limita essa inferência.

O C1 estimou um desvio-padrão da inclinação do desconto de 0,88 entre produtos e de 0,27 entre lojas. O valor-p se refere à hipótese conjunta sobre as duas variâncias. O C1 se ajustou melhor aos dados de treino. A seleção no P1 se baseou no erro fora da amostra, em que os ganhos do C1 foram inconsistentes. Os coeficientes descrevem associações nas semanas com vendas registradas. Estimar efeitos causais exigiria um desenho de identificação causal.

## Itens em aberto e não executados

Duas análises aguardam aprovação. Uma avaliaria o C1 na janela final como exploração posterior. A outra usaria um bootstrap longo por par para estimar a incerteza da diferença de erro preditivo. A auditoria manteve a semana 61. A regra geral para semanas com vendas anormalmente baixas, que podem indicar ruptura, segue em aberto.

Ficaram por executar a checagem de robustez com GLMM binomial negativa e três análises de sensibilidade. Elas restringiriam a amostra aos pares com painel completo de 156 semanas, excluiriam as linhas com preço-base imputado ou incluiriam as linhas `OUTLIER_UPV`. A análise executada excluiu as linhas de fallback. O notebook 01 apresenta as inclinações por par do OLS. Para compará-las com inclinações por par do multinível, seria preciso ajustar uma especificação com inclinação por par. Pela regra de seleção do P1, a avaliação parou no C1, antes das extensões C2 a C4. As inclinações para o lift promocional, as interações com tamanho da loja e as extensões da estrutura de covariância ficaram sem ajuste.

## Licença

O código está sob a licença MIT. Os dados seguem os termos da dunnhumby e não são redistribuídos neste repositório.
