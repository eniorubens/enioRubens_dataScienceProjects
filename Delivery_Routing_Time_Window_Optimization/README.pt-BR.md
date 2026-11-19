# Otimização de Rotas de Entrega com Capacidade e Janelas de Tempo (VRPTW)

[English](README.md) · [Relatório executivo](site/index.html)

Este projeto estuda como alocar clientes a veículos e ordenar as entregas sob capacidade dos veículos e
janelas de atendimento, minimizando **primeiro a frota e depois a distância total**. É uma análise
**prescritiva**: o produto é um plano de rotas verificável de forma independente e comparável com
soluções de referência publicadas, não uma previsão.

O código reutilizável fica no pacote `delivery_routing_vrptw` (`src/`); a análise é conduzida em
notebooks bilíngues, em português e inglês, que compartilham o mesmo código executável.

Os dados são **benchmarks acadêmicos sintéticos** (Solomon, 1987), não telemetria logística real. A
ligação com o projeto Olist é apenas temática: o Olist não contém frota, capacidade, janelas nem rotas.
`production_eligible = false`: o método está validado no benchmark; uma operação real exigiria
distâncias viárias, a frota real, tempos de serviço medidos e reotimização ao longo do dia.

## Pergunta

Como alocar clientes a veículos e ordenar as entregas minimizando a frota e depois a distância, sem
violar capacidade, janelas de tempo nem o retorno ao depósito?

## Dados

- **Instâncias:** as 56 instâncias de Solomon com 100 clientes (famílias C1, C2, R1, R2, RC1, RC2), do
  SINTEF TOP (`data/raw/sintef/solomon-100.zip`, baixado manualmente porque o site bloqueia clientes
  automatizados) e do CVRPLIB (`data/raw/cvrplib_solomon_100/Solomon.7z`). As duas fontes são
  numericamente idênticas. Proveniência e SHA-256 estão no `MANIFEST.md` de cada pasta.
- **Referências:** melhores soluções conhecidas do SINTEF (SINTEF, 2008; `data/reference/sintef_bks_100.csv`,
  transcritas da página do SINTEF) e soluções do CVRPLIB (CVRPLIB, s.d.;
  `data/reference/cvrplib_bks_100.csv`).
- **Licença:** nenhuma licença explícita foi encontrada; citar Solomon (1987), *Operations Research*
  35(2), SINTEF TOP e CVRPLIB.
- Não são usados Google Maps, dados de trânsito nem APIs pagas. As distâncias são euclidianas e fazem
  parte do contrato de dados.

### Obtenção dos dados (arquivos de terceiros não são redistribuídos)

A versão pública deste repositório **não** inclui arquivos de terceiros: o arquivo do SINTEF
(`data/raw/sintef/solomon-100.zip`), os arquivos do CVRPLIB (`data/raw/cvrplib_solomon_100/*.7z`), a página
salva do SINTEF (`docs/Solomon-html descriptions.pdf`) e as rotas das soluções do CVRPLIB
(`data/reference/cvrplib_bks_100.csv`, regenerado a partir dos arquivos do CVRPLIB). Os `MANIFEST.md`, a tabela
transcrita do SINTEF com veículos e distância (`data/reference/sintef_bks_100.csv`) e a divisão
desenvolvimento/avaliação são mantidos.

~~~
python scripts/fetch_data.py              # confere SHA-256, baixa os arquivos do CVRPLIB, regenera o csv do CVRPLIB
python scripts/fetch_data.py --check-only # apenas confere os arquivos já presentes (sem rede)
~~~

O site do SINTEF bloqueia clientes automatizados (Cloudflare): baixe `solomon-100.zip` manualmente em
https://www.sintef.no/projectweb/top/vrptw/100-customers/ e salve como `data/raw/sintef/solomon-100.zip`
(o script mostra o SHA-256 esperado). Os testes que não dependem desses arquivos rodam sem eles.

Fontes a citar: Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time
window constraints. *Operations Research, 35*(2), 254-265. SINTEF. (2008). *VRPTW - 100 customers: Solomon
benchmark, best known solutions*. SINTEF TOP. https://www.sintef.no/projectweb/top/vrptw/100-customers/ .
CVRPLIB. (s.d.). *VRPTW instances - Solomon set*. PUC-Rio. https://galgos.inf.puc-rio.br/cvrplib/

### Convenções de objetivo (risco central de comparabilidade)

| | SINTEF (usada aqui) | CVRPLIB |
|---|---|---|
| Objetivo | hierárquico: veículos, depois distância | apenas distância |
| Distância | precisão dupla, total arredondado a 2 casas | truncada em 1 casa |

Um verificador independente mostrou que 8 das 56 soluções do CVRPLIB tornam-se **infactíveis** com
distância exata e que o SINTEF usa menos veículos em 37 instâncias. Os resultados são comparados apenas
com o SINTEF, na convenção dele; o gap de distância só é reportado quando a frota é igual à da referência.

## Reprodução

O ambiente conda usa apenas o conda-forge; o OR-Tools vem do PyPI (não é publicado no conda-forge):

~~~
set CONDA_CHANNELS=conda-forge
set CONDA_CHANNEL_PRIORITY=strict
conda env create -f environment.yml
conda activate delivery-routing-vrptw
~~~

O `environment.yml` instala o pacote em modo editável e define `MKL_THREADING_LAYER=SEQUENTIAL`
(evita um crash do numpy/MKL no Windows). As versões exatas estão em `environment.lock.yml`.

Testes e lint (os testes unitários usam fixtures sintéticas; os testes de contrato leem os notebooks e
`data/reference/split.csv`; sem rede ou interface gráfica):

~~~
python -m ruff check src tests scripts
python -m pytest -q --cov=delivery_routing_vrptw
~~~

Os parâmetros dos experimentos (tempo do solver, orçamentos, fatores dos cenários, réplicas) ficam em
`config/experiment.toml` e podem ser sobrescritos na linha de comando. Experimentos, na ordem em que
foram executados:

| Etapa | Script | Instâncias |
|---|---|---|
| Divisão desenvolvimento/avaliação (semente 20261117) | `scripts/make_split.py` | todas |
| Gate (baselines + OR-Tools) | `scripts/run_gate.py` | 1 dev por família |
| Torneio pré-registrado (18 configurações) | `scripts/run_dev_grid.py`, `scripts/select_config.py` | 14 dev |
| Sensibilidade ao tempo de busca | `scripts/run_time_sensitivity.py` | 14 dev |
| Janelas alargadas (cenário de negócio) | `scripts/run_window_scenarios.py` | 14 dev |
| Entrega antecipada (cenário de negócio) | `scripts/run_early_delivery.py` | 14 dev |
| Avaliação final pré-registrada | `scripts/run_evaluation.py --confirm-eval` | 42 eval |
| Auditoria, fase A: métricas de tempo e resumos da camada B derivados dos resultados congelados (sem solver) | `scripts/derive_eval_time_metrics.py`, `scripts/derive_layerb_summaries.py` | 42 eval / 14 dev |
| Auditoria, fase B: diagnósticos v2 (B1 escala, B2 custo fixo, B3 estratégia de frota, B4 carga da máquina) | `scripts/run_v2_solver_grid.py`, `scripts/select_v2.py` | 14 dev |
| Auditoria, fase B: partida a quente entre níveis da camada B (B5) | `scripts/run_v2_layerb_warm.py` | 14 dev |
| Fase C1: robustez dos planos congelados da avaliação sob atrasos (sem solver) | `scripts/run_robustness.py` | 42 planos de avaliação (descritivo) |
| Fase C2: capacidade reduzida, frota limitada com prioridade, janelas flexíveis | `scripts/run_phase_c_solver.py` | 14 dev |
| Fase C3: reotimização após remover ou inserir 10% dos clientes | `scripts/run_reoptimisation.py` | 14 dev |
| Fase D: reavaliação v2 pós-hoc e comparação v1 × v2 | `scripts/run_evaluation.py --version v2 --confirm-eval --post-hoc`, `scripts/compare_eval_versions.py` | 42 eval (pós-hoc) |
| Execução dos notebooks | `scripts/execute_notebooks.py` | — |

Os scripts que retomam uma pasta são protegidos por uma impressão digital da configuração (`delivery_routing_vrptw.runs`;
pastas legadas exigem `--allow-legacy-resume`). Ela cobre a configuração que determina os resultados, **inclusive o
número de processos** (limites de tempo de relógio fazem o resultado depender da contenção de CPU; o bloco B4
compara 4 e 8), mas não datas nem caminhos. No C3, o agregado do protocolo é `aggregate_representative.csv` (réplica
representativa); `aggregate.csv` junta todas as réplicas e é mantido por rastreabilidade. Para as pastas C3
congeladas, o agregado do protocolo é derivado com `python scripts/derive_phase_c_corrections.py` (sem solver).

As instâncias de avaliação são protegidas: todo script as recusa sem confirmação explícita. Execuções
cronometradas não devem dividir a máquina com outras cargas (o limite por relógio torna o resultado
dependente da carga). Os resultados usados pelos notebooks estão congelados em `outputs/frozen/`
(ver o `README.md` da pasta).

## Mapa dos notebooks

| Etapa | PT-BR | EN-US | O que apresenta |
| :--- | :---: | :---: | :--- |
| 01 — Apresentação e pré-análise | [PT-BR](notebooks/pt_br/01_apresentacao_e_pre_analise.ipynb) | [EN-US](notebooks/en_us/01_presentation_and_pre_analysis.ipynb) | problema, proveniência, contrato de dados, convenções de objetivo |
| 02 — Análise exploratória das instâncias | [PT-BR](notebooks/pt_br/02_eda_instancias.ipynb) | [EN-US](notebooks/en_us/02_instance_eda.ipynb) | geografia, janelas, frota limitada por capacidade × por janelas |
| 03 — Verificador e baselines | [PT-BR](notebooks/pt_br/03_verificador_e_baselines.ipynb) | [EN-US](notebooks/en_us/03_verifier_and_baselines.ipynb) | verificador independente, heurísticas construtivas |
| 04 — Torneio de desenvolvimento | [PT-BR](notebooks/pt_br/04_torneio_desenvolvimento.ipynb) | [EN-US](notebooks/en_us/04_development_tournament.ipynb) | torneio pré-registrado, configuração congelada, orçamento de tempo, diagnósticos v2 (fase B) |
| 05 — Avaliação final | [PT-BR](notebooks/pt_br/05_avaliacao_final.ipynb) | [EN-US](notebooks/en_us/05_final_evaluation.ipynb) | avaliação final nas 42 instâncias inéditas; reavaliação v2 pós-hoc ao lado |
| 06 — Cenários de negócio | [PT-BR](notebooks/pt_br/06_cenarios_de_negocio.ipynb) | [EN-US](notebooks/en_us/06_business_scenarios.ipynb) | janelas alargadas, entrega antecipada, realocação, partida a quente entre níveis (B5) |
| 07 — Robustez e cenários operacionais | [PT-BR](notebooks/pt_br/07_robustez_e_cenarios_operacionais.ipynb) | [EN-US](notebooks/en_us/07_robustness_and_operational_scenarios.ipynb) | camada B: robustez de planos fixos sob atrasos, capacidade reduzida, frota limitada com prioridade, janelas flexíveis, reotimização |

## Método

- **Verificador:** todo plano é conferido em precisão dupla (capacidade, visita única, janelas, retorno
  ao depósito, limite de frota); as métricas reportadas vêm do verificador, nunca do objetivo do solver.
- **Baselines:** gulosa por prazo (regra do prazo mais cedo, Jackson, 1955; *first-fit*, Johnson, 1974) e
  vizinho mais próximo orientado ao tempo de Solomon (Solomon, 1987).
- **Solver:** roteamento do OR-Tools (Perron & Furnon, 2026) com custo fixo por veículo (frota primeiro;
  objetivo hierárquico como em Bräysy & Gendreau, 2005a), custos inteiros e
  tempos de viagem arredondados para cima (um plano factível para o solver é factível em precisão dupla).
- **Anti-vazamento:** divisão estratificada 14/42 sorteada antes de qualquer execução do solver;
  configuração escolhida nas 14 instâncias de desenvolvimento por uma regra registrada antes; protocolo
  de avaliação (orçamento, réplicas, resultado principal) registrado antes de abrir as 42 instâncias de
  avaliação.

## Resultados

Configuração congelada: custo fixo por veículo + `SAVINGS` (heurística das economias; Clarke & Wright, 1964)
+ `GUIDED_LOCAL_SEARCH` (busca local guiada; Voudouris & Tsang, 1999). Avaliação final: 42
instâncias de avaliação, 120 s × 5 réplicas, resultado principal = réplica mediana, 294/294 planos
factíveis.

| Escopo | Veículos a mais que o SINTEF (média) | Frota igual ao SINTEF | Gap mediano de distância (frota igual) |
|---|---|---|---|
| **Geral** | **0,50** | **24 / 42** | 3,40% |
| C1 | 0,00 | 7 / 7 | 0,00% |
| C2 | 0,00 | 6 / 6 | 0,00% |
| R1 | 0,89 | 1 / 9 | 0,52% |
| R2 | 0,25 | 6 / 8 | 5,02% |
| RC1 | 1,50 | 0 / 6 | — |
| RC2 | 0,33 | 4 / 6 | 4,63% |

- A frota nunca fica abaixo da referência: +1 veículo em 15 instâncias, +2 em 3 (todas RC1).
- Nas famílias C a frota de referência é atingida nas 13 instâncias, e 10 reproduzem a distância de
  referência ao centésimo; R1 e RC1, onde as janelas são a restrição ativa, são o limite da abordagem.
- A frota foi idêntica nas 5 réplicas em todas as 42 instâncias.
- Baselines: vizinho mais próximo +1,36 veículo (12/42 com frota igual), gulosa +3,98 (2/42).

**Cenários de negócio** (instâncias de desenvolvimento, comparados com o próprio plano original, nunca
com o SINTEF):

- **Janelas alargadas (×2):** 0,79 veículo economizado em média (8 de 14 instâncias reduzem a frota).
  A distância nunca é média entre frotas diferentes: −2,0% onde a frota não muda (6 instâncias) e
  −12,3% onde é reduzida (8). **Nenhuma** economia nas famílias C, cuja frota é determinada pela
  capacidade.
- **Entrega antecipada permitida** (o benchmark obriga a esperar a abertura da janela; na operação real a
  entrega antecipada costuma ser aceita): a antecipação livre economiza 0,50 veículo em média; onde a
  frota não muda (7 instâncias), a duração das rotas varia −20,8% e a distância −18,1%, e onde é reduzida
  (6 instâncias), −11,7% e −11,0%, ambas medidas com saída ótima do depósito (folga de tempo; Savelsbergh, 1992) (a medida ingênua "todos
  saem às 0" exageraria o ganho de duração em 1,0 a 7,9 pontos). Em R2/RC2 a duração cai cerca de 37% e
  35% entre as instâncias de frota igual. Cerca de 40% dos clientes mudam de veículo: o ganho vem de
  replanejar, o que tem custo operacional.

## Auditoria e v2

Depois da avaliação final, uma auditoria externa (`reports/plano-sanidade-auditoria.md`) revisou o
projeto. Os dez achados foram triados por leitura do código e dos resultados congelados: a maioria foi
confirmada, entre eles que a escala inteira 100 com arredondamento para cima poda rotas válidas (R106 e
R110), que o nosso diagnóstico da falha de `reduce_fleet` estava errado, que os níveis da camada B
resolvidos sem partida a quente não são monotônicos (6 + 14 células), que a avaliação não registrava
duração e espera das rotas e que algumas médias da camada B misturavam frotas diferentes; a afirmação de
que um custo fixo por veículo muito alto distorce a busca era apenas uma hipótese. O trabalho seguiu
quatro fases, cada uma com o protocolo registrado antes da execução (`reports/protocolo-v2.md`,
`reports/protocolo-fase-c.md`, `reports/protocolo-fase-d.md`), e os resultados da v1 continuam congelados
como resultado original:

- **A, correções sem executar o solver:** duração e espera derivadas das rotas congeladas (notebook 05),
  réplicas representativas lexicográficas e separação por classe de frota na camada B (notebook 06),
  barra de cores normalizada no notebook 02 e documentação corrigida.
- **B, diagnósticos do solver nas 14 instâncias de desenvolvimento (v2):** B1 escala 10 000 contra 100
  (posição média 1,32 contra 1,68; R110 passa de 12 para 11 veículos), B2 custo fixo k = 1, 10, 100 (mesma
  frota nos três níveis; k = 10 mantido), B3 `reduce_fleet` corrigido (iguala o SINTEF em R202, 4 → 3, mas
  posição média 1,68 contra 1,32, e por isso `fixed_cost` é mantido), B4 4 contra 8 processos (mesma frota
  em 42/42 células), B5 partida a quente entre níveis da camada B (células não monotônicas 6 → 0 e
  14 → 0). **v2 = a configuração da v1 com escala inteira 10 000 e custo fixo relativo k = 10, cuja magnitude é
  ~50–66% do custo fixo absoluto da v1** (corrigido em 2026-11-19; só a escala foi isolada, no B1).
- **C, camada B (ver abaixo).**
- **D, reavaliação v2 pós-hoc das 42 instâncias de avaliação** (já vistas, por isso rotulada "v2, pós-hoc"
  e apresentada ao lado da v1, nunca no lugar dela): 210/210 factíveis; veículos a mais que o SINTEF
  0,50 → 0,48; frota igual ao SINTEF em **25 / 42** instâncias (24 na v1), e a única mudança de frota é
  **RC207, 4 → 3**. Com frota igual, a distância varia nos dois sentidos (de −4,3% a +2,1%; 18 de 41
  instâncias dentro de ±0,5%, 11 mais curtas, 12 mais longas), de modo que a queda da mediana do gap de
  3,40% para 2,23% reflete essas oscilações e a mudança do conjunto de frota igual, não um ganho
  consistente. A expectativa registrada de ganho em instâncias de janelas apertadas, como R106, não se
  confirmou (13 veículos nas duas versões). A v2 corrige um defeito demonstrado, mas o impacto agregado é
  marginal; a conclusão da v1 se mantém. **Correção (2026-11-19):** a v2 difere da v1 na escala **e** na
  magnitude do custo fixo (de 49,9% a 66,2% do valor da v1 em unidades de distância, mediana 57,7%); a
  comparação é, portanto, entre duas configurações, e a mudança em RC207 é atribuível à configuração v2, e
  não à escala isoladamente.

### Fase C: robustez e cenários operacionais (camada B, sem comparação com o SINTEF)

Notebook 07 (`reports/fase-c-robustez.md`; cada cenário é comparado com o plano de referência da mesma
instância; C2 e C3 usam as 14 instâncias de desenvolvimento, 30 s, 3 réplicas):

- **C1, planos fixos sob atrasos** (planos congelados da avaliação, usados de forma descritiva, 1 000
  sorteios por plano e cenário): com viagem +0–10%, 6,3% dos clientes do OR-Tools são atendidos com atraso,
  contra 9,3% (vizinho mais próximo) e 21,1% (gulosa), e 13,4% dos retornos ao depósito são tardios, contra
  32,9% e 43,3%; por rota, o OR-Tools tem mais rotas com algum atraso sob choque de serviço (84,2% contra
  71,1% e 72,5%). A folga nominal explica a fragilidade (correlação de Spearman com a taxa de atraso de
  −0,37 a −0,78).
- **C2.1, capacidade reduzida:** 90% e 80% de Q exigem 0,21 e 0,57 veículo a mais por instância (3 e 6 das
  14 instâncias), com cerca de +14% de distância onde a frota cresce.
- **C2.2, frota limitada com prioridade:** com K − 1, a penalidade uniforme deixa 111 clientes sem
  atendimento (23 prioritários) e a prioridade deixa 122 (0 prioritários); com K − 2, 344 (69) contra 363 (3). O número descartado varia um pouco entre réplicas da mesma célula
  (RC108, K − 2, penalidade uniforme: 9, 7 e 7), porque a penalidade incentiva o atendimento sem garantir
  um descarte mínimo.
- **C2.3, janelas flexíveis: exploratório, pouco informativo:** só 3 de 14 instâncias foram resolvidas (4
  infactíveis por capacidade e 7 com falha na construção inicial), nenhuma conclusão geral é tirada e o
  resultado não deve ser citado sem esta ressalva.
- **C3, reotimização após remover ou inserir 10% dos clientes:** a partida a quente muda o veículo de 26,7%
  (remoção) e 30,0% (inserção) dos clientes comuns, contra 40,6% e 36,7% a frio (réplica representativa de
  cada instância; corrigido em 2026-11-19); o patamar de ruído tem mediana 0%, mas não é zero na remoção (4
  de 42 comparações entre réplicas a frio realocaram clientes, até 23,3%, em R111 e RC203; 0 de 42 na
  inserção). Replanejar tem custo operacional relevante nos dois casos.

## Limitações

- Benchmark sintético (distância euclidiana no plano, velocidade 1, frota homogênea).
- Os valores do SINTEF são as melhores soluções conhecidas, não ótimos provados; não se alega
  otimalidade.
- O limite por relógio torna o resultado dependente da carga da máquina: com 30 s, execuções idênticas
  diferiram em ±1 veículo em instâncias R/RC; com 120 s e a máquina ociosa, a frota foi estável.
- A estratégia de redução de frota (`reduce_fleet`) não foi testada de forma efetiva no torneio. A
  tentativa com um veículo a menos falhou quase de imediato (tempo total de cerca de 10 s dos 30 s de
  orçamento com `PARALLEL_CHEAPEST_INSERTION` e `SAVINGS`; só `PATH_CHEAPEST_ARC` usou o passo inteiro
  de 5 s) e o orçamento restante foi descartado, de modo que a comparação com a estratégia de custo fixo
  é inconclusiva e enviesada contra `reduce_fleet`. A explicação anterior (passo de 5 s curto demais)
  estava errada (corrigida em 2026-11-18). O experimento B3 repetiu depois a comparação com um `reduce_fleet`
  corrigido (partida a quente, descarte penalizado, 120 s): ele iguala o SINTEF em R202, mas perde na
  posição média, e `fixed_cost` é mantido (ver Auditoria e v2).
- Nas execuções da camada B da v1, os níveis dos cenários foram resolvidos sem partida a quente entre si, e
  por isso os resultados não são monotônicos: 20 células (6 de janelas alargadas e 14 de entrega
  antecipada) são lexicograficamente piores (veículos, depois distância) que o melhor plano de um nível
  menos permissivo da mesma instância. Com partida a quente (experimento B5) essas células desaparecem,
  mas as tabelas principais do notebook 06 continuam sendo as da v1.
- A reavaliação v2 pós-hoc reutiliza instâncias já vistas; a diferença marginal em relação à v1 é
  descritiva e não substitui o resultado da v1.
- Fase C: o C2.3 (janelas flexíveis) é exploratório (3 de 14 instâncias resolvidas): falhou em 7 das 10
  instâncias viáveis por capacidade por causa da construção inicial (não por tempo) e não é conclusivo; o C1 usa choques independentes, sem a correlação
  espacial e temporal do trânsito real; C2 e C3 usam 14 instâncias de desenvolvimento.
- Desvio da regra da pré-análise de que os valores do SINTEF das instâncias de avaliação só seriam lidos
  no torneio final: os notebooks 01 e 02 leem os 56 valores publicados de forma descritiva (convenções e
  limites de frota); eles não foram usados para escolher configuração, orçamento ou parâmetro.
- Os cenários de negócio usam apenas as 14 instâncias de desenvolvimento e tratam o ganho de tempo como
  efeito colateral (o solver otimiza frota e distância, não duração).

## Documentação

`reports/` reúne a pré-análise de viabilidade, a avaliação da proposta, os protocolos pré-registrados
(grade, avaliação, v2, fase C, fase D) com seus resultados, o plano de sanidade da auditoria, o relatório de
robustez, o estudo de sensibilidade ao tempo e às janelas e o cenário de entrega antecipada. `docs/Solomon-html descriptions.pdf` é a página do SINTEF salva, com a tabela de
referência.

## Referências

As obras citadas no README e nos notebooks, no formato autor-ano, estão listadas em
[`docs/referencias.md`](docs/referencias.md) (lista curada; cada notebook termina com uma seção Referências).
