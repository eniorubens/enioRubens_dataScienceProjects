# Precificação e promoção no varejo com modelos multinível

[English](README.md) · [Relatório executivo](site/index.html)

> **Status:** o protocolo P1 (seleção de modelos e avaliação final) e as análises de treino dos notebooks 05 e 06 estão concluídos. Os itens em aberto e os não executados estão no fim deste README.

## Pergunta de negócio

Quanto o **preço** e a **promoção** (encarte, display e desconto temporário, o TPR) estão associados à venda em unidades **entre lojas e produtos**, e como se comparam cenários de preço e promoção, dado o suporte observado de cada um? Margem, custo de promoção, ROI e recomendação causal de preço ficam fora do escopo.

O argumento central é o **encolhimento (partial pooling)**. A elasticidade estimada loja a loja por OLS é ruidosa, e às vezes positiva, o que não faz sentido econômico. O modelo multinível estabiliza essas estimativas usando a informação do painel inteiro. O projeto testa essa afirmação com evidência, inclusive fora da amostra.

## Abordagem (resumo)

- Resposta: `log(UNITS)` (modelo linear misto). A checagem de robustez com GLMM binomial negativa, prevista, não foi executada.
- Passos executados: modelos nulos e ICCs, com testes de razão de verossimilhança corrigidos para a fronteira → efeitos fixos (profundidade do desconto, preço-base decomposto em desvio dentro do par e média do par, encarte, display, TPR, sazonalidade de Fourier e tendência), com interceptos aleatórios cruzados (loja, produto, loja×produto) → inclinações aleatórias do desconto por loja e produto (C1), julgadas pelo erro fora da amostra no protocolo P1 e depois por um teste de razão de verossimilhança com bootstrap paramétrico no treino → interações planejadas da inclinação do desconto com segmento da loja, marca própria e categoria (Holm) → sensibilidade a choques de categoria × semana → cenários. As inclinações do lift de promoção, as interações com o tamanho da loja e as extensões da estrutura de covariância não foram ajustadas.
- Comparação: OLS separado por loja×produto, OLS agrupado e multinível (BLUP), mais um desafiante GPBoost. A seleção de modelos usa duas dobras internas dentro das primeiras 104 semanas. A comparação final usa as últimas 52 semanas, que são uma **janela de avaliação temporal retrospectiva com exposição exploratória prévia** (decisão D11, revisada). Os modelos da avaliação formal do P1 foram ajustados nas primeiras 104 semanas. O OLS exploratório do brief usou o painel inteiro. A comparação seguiu o protocolo P1, registrado antes do ajuste de qualquer candidato. **O P1 está concluído.** A especificação selecionada, os parâmetros do GPBoost e os hashes do código e do painel foram congelados **antes da pontuação formal** da janela final, que foi pontuada uma única vez. As previsões são condicionais ao preço e à promoção realizados em cada semana.
- Motor: `lme4` (R, instalado dentro do env conda a partir do conda-forge), chamado do Python via `rpy2`. O `statsmodels.MixedLM` confere o modelo nulo só com loja.
- As semanas sem linha ficam fora do modelo, porque não dá para distinguir venda zero de ruptura, e as estimativas são condicionais a haver venda registrada. A sensibilidade prevista com os pares de painel completo de 156 semanas não foi executada.

## Como obter os dados

Os dados **não** estão neste repositório (licença da dunnhumby).

1. Cadastre-se e baixe *Breakfast at the Frat: A Time Series Analysis* em https://www.dunnhumby.com/source-files/.
2. Coloque `dunnhumby - Breakfast at the Frat.xlsx` (e o guia em PDF) em `dataset/`.

Dados: dunnhumby Source Files – Breakfast at the Frat.

## Reprodução

~~~
conda env create -f environment.yml
conda activate retail-multilevel
python scripts/ingest.py        # Excel -> data/interim/*.parquet (~1 min, uma vez)
python scripts/build_panel.py   # regras de limpeza -> data/processed/panel.parquet
python scripts/validate_data.py # contrato de qualidade no painel real (só local)
pytest                          # só fixtures sintéticas; é o que o CI roda
~~~

Os modelos são ajustados por script, uma vez por experimento, em `outputs/runs/`. Os notebooks PT e EN só leem essas execuções gravadas e nunca ajustam de novo.

No Windows, o env conda traz o próprio R, e o `retail_multilevel.rbridge` aponta o `R_HOME` para ele mesmo quando há um R instalado no sistema. As regras de limpeza e as contagens de linhas estão no notebook 01.

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

Nas dobras internas, as inclinações aleatórias do desconto por loja e produto (C1) melhoraram os pares com menos de 26 semanas de histórico nas duas dobras, mas falharam na regra global pré-registrada. Por isso, foi selecionado o modelo multinível só com interceptos (C0, com inclinação de desconto comum). A janela final avaliou o C0, e a questão das inclinações aleatórias para pares curtos fica **em aberto**.

Na janela final, o modelo multinível superou com folga o OLS agrupado e o OLS por loja×produto. O C0 e o desafiante GPBoost tiveram desempenho próximo, e o vencedor muda conforme a métrica. O GPBoost teve MAE por par 0,47% menor nos pares vistos (0,4520 contra 0,4541). O C0 teve menor MAE por linha (0,4446 contra 0,4478) e menor erro nos pares inéditos (0,626 contra 0,688). As diferenças são observadas, sem análise de incerteza. Nos pares curtos, o resultado depende da população e do peso; ver o notebook 04 e o registro de decisões.

## Contrastes e cenários (notebooks 05 e 06, só no treino)

Os três testes globais planejados rejeitaram, após Holm, a hipótese de uma inclinação comum ao desconto entre segmentos de loja, entre marca própria e marca de fabricante, e entre categorias. As inclinações foram de cerca de −1,1 em BAG SNACKS a −2,4 em FROZEN PIZZA. Ao acrescentar um intercepto aleatório de categoria × semana, os choques semanais comuns foram absorvidos, e os coeficientes de preço e promoção quase não mudaram. Nos cenários, display com 20% de desconto foi associado a vendas 119% acima de uma semana regular, e TPR com 20% a 40% acima. A razão modelada entre os dois foi de 1,57. Os oito cenários têm suporte observado no treino, ainda que de 0,3% a 3% das linhas nos promocionais.

## Inclinações aleatórias do desconto no treino (notebook 04, seção 9)

Um teste conjunto complementar, feito depois do P1 e só nas semanas de treino, comparou C0 e C1 por máxima verossimilhança e calibrou a razão de verossimilhanças por bootstrap paramétrico sob o C0 (39 réplicas). A estatística observada foi 12.974, contra uma maior simulada de 5,18, e o valor-p de Monte Carlo foi 0,025, a menor resolução do desenho: a hipótese de que as duas variâncias de inclinação sejam zero foi rejeitada. O C1 estima um desvio-padrão da inclinação do desconto de 0,88 entre produtos e de 0,27 entre lojas. O C1 ajusta muito melhor o treino, mas não reduziu o erro fora da amostra de forma consistente, e por isso o C0 continua sendo o modelo selecionado.

Os coeficientes são associações condicionais nas semanas com transações registradas, não efeitos causais. Continuam adiados, dependendo de aprovação, a pontuação do C1 na janela final e o bootstrap longo por par.

## Itens em aberto e não executados

Dependem de aprovação a pontuação do C1 na janela final, como exploração posterior, e o bootstrap longo por par para a incerteza da diferença de erro preditivo. Também fica em aberto uma regra geral para semanas com venda anormalmente baixa (possível ruptura); a semana 61 foi auditada e mantida.

Foram previstos e não executados, sem nenhum resultado publicado que dependa deles: a checagem de robustez com GLMM binomial negativa; as sensibilidades com os pares de painel completo de 156 semanas, sem as linhas de preço-base imputado e com as linhas `OUTLIER_UPV` (só a sensibilidade sem as linhas de fallback foi feita); a comparação das inclinações do OLS por par com as do multinível, já que o C0 não tem inclinação por par (as inclinações do OLS por par estão no notebook 01); as extensões C2 a C4 do protocolo P1, interrompidas pela regra após a rejeição do C1; e as interações com o tamanho da loja e as extensões da estrutura de covariância.

## Licença

Código sob licença MIT. Os dados da dunnhumby seguem os termos da própria dunnhumby e não são redistribuídos.
