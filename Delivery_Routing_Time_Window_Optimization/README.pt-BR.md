# Otimização de rotas de entrega com capacidade e janelas de tempo (VRPTW)

[English](README.md) · [Relatório executivo](site/index.html)

Este projeto estuda a alocação de clientes a veículos e o sequenciamento de entregas sob restrições de capacidade e janelas de atendimento, minimizando primeiro a frota e depois a distância total. Trata-se de uma análise prescritiva cujo produto é um plano de rotas verificável de forma independente e comparável a soluções de referência publicadas. O código reutilizável reside no pacote `delivery_routing_vrptw` (`src/`), enquanto a análise é documentada em notebooks bilíngues (português e inglês) que compartilham o mesmo código executável.

Os dados pertencem ao benchmark acadêmico sintético de Solomon (1987). No portfólio, o trabalho dá continuidade temática ao projeto Olist no domínio de entregas, utilizando exclusivamente os dados do benchmark, já que a base do marketplace não possui registros de frota, capacidade, janelas ou rotas. Como o método foi validado apenas no benchmark (`production_eligible = false`), uma operação real exigiria distâncias viárias, frota real, tempos de serviço medidos e reotimização ao longo do dia.

## Pergunta

Como alocar clientes a veículos e ordenar as entregas minimizando a frota e depois a distância, sem violar capacidade, janelas de tempo nem o retorno ao depósito?

## Dados

Foram utilizadas as 56 instâncias de Solomon com 100 clientes (famílias C1, C2, R1, R2, RC1 e RC2), obtidas do SINTEF TOP (`data/raw/sintef/solomon-100.zip`, baixado manualmente porque o site bloqueia clientes automatizados) e do CVRPLIB (`data/raw/cvrplib_solomon_100/Solomon.7z`). As duas fontes são numericamente idênticas, e sua proveniência e hashes SHA-256 estão registrados no `MANIFEST.md` de cada pasta. As soluções de referência vêm do SINTEF (SINTEF, 2008; `data/reference/sintef_bks_100.csv`, transcritas da página do SINTEF) e do CVRPLIB (CVRPLIB, s.d.; `data/reference/cvrplib_bks_100.csv`). Nenhuma licença explícita foi encontrada nos arquivos originais, devendo-se citar Solomon (1987), *Operations Research* 35(2), SINTEF TOP e CVRPLIB. As distâncias são euclidianas no plano e fazem parte do contrato de dados, sem uso de Google Maps, dados de trânsito ou APIs pagas.

### Obtenção dos dados (arquivos de terceiros não são redistribuídos)

A versão pública deste repositório omite arquivos de terceiros: o arquivo do SINTEF (`data/raw/sintef/solomon-100.zip`), os arquivos do CVRPLIB (`data/raw/cvrplib_solomon_100/*.7z`), a página salva do SINTEF (`docs/Solomon-html descriptions.pdf`) e as rotas das soluções do CVRPLIB (`data/reference/cvrplib_bks_100.csv`, regenerado a partir dos arquivos do CVRPLIB). Permanecem no repositório os arquivos `MANIFEST.md`, a tabela transcrita do SINTEF com veículos e distância (`data/reference/sintef_bks_100.csv`) e a divisão desenvolvimento/avaliação.

~~~
python scripts/fetch_data.py              # confere SHA-256, baixa os arquivos do CVRPLIB, regenera o csv do CVRPLIB
python scripts/fetch_data.py --check-only # apenas confere os arquivos já presentes (sem rede)
~~~

Como o site do SINTEF bloqueia clientes automatizados via Cloudflare, é necessário baixar `solomon-100.zip` manualmente em https://www.sintef.no/projectweb/top/vrptw/100-customers/ e salvá-lo em `data/raw/sintef/solomon-100.zip` (o script exibe o SHA-256 esperado). Os testes que não dependem desses arquivos executam normalmente sem eles.

As fontes a citar são Solomon, M. M. (1987), Algorithms for the vehicle routing and scheduling problems with time window constraints (*Operations Research, 35*(2), 254-265); SINTEF (2008), *VRPTW - 100 customers: Solomon benchmark, best known solutions* (SINTEF TOP, https://www.sintef.no/projectweb/top/vrptw/100-customers/); e CVRPLIB (s.d.), *VRPTW instances - Solomon set* (PUC-Rio, https://galgos.inf.puc-rio.br/cvrplib/).

### Convenções de objetivo (risco central de comparabilidade)

| | SINTEF (usada aqui) | CVRPLIB |
|---|---|---|
| Objetivo | hierárquico: veículos, depois distância | apenas distância |
| Distância | precisão dupla, total arredondado a 2 casas | truncada em 1 casa |

Um verificador independente demonstrou que 8 das 56 soluções do CVRPLIB tornam-se infactíveis com distância exata e que o SINTEF usa menos veículos em 37 instâncias. Por isso, os resultados são comparados apenas com o SINTEF, sob sua convenção, e o gap de distância só é reportado quando a frota iguala a da referência.

## Reprodução

O ambiente conda utiliza apenas o canal `conda-forge`, enquanto o OR-Tools é obtido do PyPI:

~~~
set CONDA_CHANNELS=conda-forge
set CONDA_CHANNEL_PRIORITY=strict
conda env create -f environment.yml
conda activate delivery-routing-vrptw
~~~

O arquivo `environment.yml` instala o pacote em modo editável e define `MKL_THREADING_LAYER=SEQUENTIAL` para evitar um erro do numpy/MKL no Windows; as versões exatas constam em `environment.lock.yml`. Os testes unitários utilizam fixtures sintéticas e os testes de contrato leem os notebooks e `data/reference/split.csv`, sem rede nem interface gráfica:

~~~
python -m ruff check src tests scripts
python -m pytest -q --cov=delivery_routing_vrptw
~~~

Os parâmetros dos experimentos (tempo do solver, orçamentos, fatores dos cenários e réplicas) ficam em `config/experiment.toml` e podem ser sobrescritos na linha de comando. A tabela resume as execuções realizadas, em ordem cronológica:

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

Os scripts que retomam uma pasta são protegidos por uma impressão digital da configuração (`delivery_routing_vrptw.runs`; pastas legadas exigem `--allow-legacy-resume`). Essa verificação cobre os parâmetros que determinam os resultados, incluindo o número de processos (visto que limites por tempo de relógio dependem da contenção de CPU; o bloco B4 compara 4 e 8 processos), mas ignora datas e caminhos. No C3, o agregado do protocolo é `aggregate_representative.csv` (réplica representativa), enquanto `aggregate.csv` reúne todas as réplicas para rastreabilidade; para as pastas C3 congeladas, o agregado do protocolo é derivado com `python scripts/derive_phase_c_corrections.py` sem executar o solver.

As instâncias de avaliação são protegidas e todo script as recusa sem confirmação explícita. Execuções cronometradas não devem compartilhar a máquina com outras cargas porque o limite por relógio torna o resultado sensível à carga. Os resultados consumidos pelos notebooks estão congelados em `outputs/frozen/` (detalhados no `README.md` interno da pasta).

Os metadados dessas execuções registravam o caminho absoluto da pasta do projeto na máquina local. Antes da publicação, esses caminhos foram substituídos pelo marcador neutro PROJECT_ROOT, e a verificação pode ser repetida a qualquer momento (o código de saída é 1 quando ainda resta algum caminho local):

~~~
python scripts/anonymize_local_paths.py --check
~~~

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

Cada plano gerado é conferido por um verificador independente em precisão dupla quanto a capacidade, visita única, janelas de atendimento, retorno ao depósito e limite de frota; todas as métricas reportadas provêm desse verificador, e nunca do valor objetivo interno do solver. As baselines compreendem uma heurística gulosa por prazo (regra do prazo mais cedo de Jackson, 1955, com alocação *first-fit* de Johnson, 1974) e o algoritmo do vizinho mais próximo orientado ao tempo de Solomon (1987).

Para a otimização principal utilizou-se o roteamento do OR-Tools (Perron & Furnon, 2026) com custo fixo por veículo, implementando o objetivo hierárquico de frota seguido de distância (Bräysy & Gendreau, 2005a), com custos inteiros e tempos de viagem arredondados para cima de modo que toda solução factível no solver seja factível em precisão dupla. Como medida contra vazamento de dados, sorteou-se a divisão estratificada 14/42 antes de qualquer execução; a configuração foi escolhida apenas nas 14 instâncias de desenvolvimento por regra registrada previamente, e o protocolo da avaliação final (orçamento, réplicas e critério do resultado principal) foi fixado antes da abertura das 42 instâncias de avaliação.

## Resultados

A configuração congelada combina custo fixo por veículo, a heurística construtiva `SAVINGS` (Clarke & Wright, 1964) e a metaheurística `GUIDED_LOCAL_SEARCH` (Voudouris & Tsang, 1999). Na avaliação final sobre as 42 instâncias inéditas (120 s × 5 réplicas, com resultado principal definido na réplica mediana), todos os 294 planos obtidos foram factíveis.

| Escopo | Veículos a mais que o SINTEF (média) | Frota igual ao SINTEF | Gap mediano de distância (frota igual) |
|---|---|---|---|
| **Geral** | **0,50** | **24 / 42** | 3,40% |
| C1 | 0,00 | 7 / 7 | 0,00% |
| C2 | 0,00 | 6 / 6 | 0,00% |
| R1 | 0,89 | 1 / 9 | 0,52% |
| R2 | 0,25 | 6 / 8 | 5,02% |
| RC1 | 1,50 | 0 / 6 | — |
| RC2 | 0,33 | 4 / 6 | 4,63% |

A frota do OR-Tools nunca ficou abaixo da referência, registrando +1 veículo em 15 instâncias e +2 veículos em 3 instâncias (todas de RC1), com resultado de frota idêntico entre as 5 réplicas em todas as 42 instâncias. Nas famílias C, a frota de referência foi atingida nas 13 instâncias e 10 delas reproduziram a distância do SINTEF ao centésimo; já R1 e RC1, onde as janelas são a restrição ativa, marcaram o limite da abordagem. As baselines terminaram atrás: o vizinho mais próximo registrou +1,36 veículo em média (12/42 com frota igual) e a gulosa por prazo +3,98 (2/42).

Nos cenários de negócio (executados apenas nas 14 instâncias de desenvolvimento e comparados ao próprio plano original, sem comparação com o SINTEF), o alargamento simétrico das janelas por fator 2 economizou 0,79 veículo em média (redução em 8 de 14 instâncias). Separando por classe de frota, a distância variou −2,0% onde a frota permaneceu igual (6 instâncias) e −12,3% onde foi reduzida (8 instâncias), sem qualquer economia nas famílias C, nas quais a capacidade determina a frota. Quando a entrega antecipada foi permitida sem restrição de abertura (tolerância livre), a economia média foi de 0,50 veículo. Com saída ótima ajustada pela folga temporal (Savelsbergh, 1992), a duração das rotas variou −20,8% e a distância −18,1% onde a frota se manteve igual (7 instâncias), e −11,7% e −11,0% onde a frota diminuiu (6 instâncias); medir a duração com todos saindo no instante zero superestimaria esse ganho em 1,0 a 7,9 pontos percentuais. Em R2 e RC2 a duração caiu cerca de 37% e 35% nos casos de frota igual, porém com realocação de cerca de 40% dos clientes entre veículos, indicando que o ganho decorre de replanejamento com custo operacional.

## Auditoria e v2

Depois da avaliação final, uma auditoria externa (`reports/plano-sanidade-auditoria.md`) revisou o projeto e teve a maioria dos dez achados confirmada por inspeção de código e arquivos congelados: a escala inteira 100 com arredondamento para cima podava rotas válidas (R106 e R110), o diagnóstico anterior para a falha de `reduce_fleet` estava errado, os níveis da camada B sem partida a quente apresentavam 20 células não monotônicas (6 em janelas e 14 em entrega antecipada), a avaliação final não gravava duração e espera das rotas e algumas médias da camada B misturavam frotas diferentes; a hipótese de que um custo fixo alto distorcia a busca exigiu experimento próprio. As correções seguiram quatro fases com protocolos pré-registrados (`reports/protocolo-v2.md`, `reports/protocolo-fase-c.md` e `reports/protocolo-fase-d.md`), mantendo os resultados originais da v1 congelados.

Na Fase A (sem executar o solver), derivaram-se duração e espera a partir das rotas congeladas da avaliação (notebook 05), adotaram-se réplicas representativas lexicográficas e separação por classe de frota na camada B (notebook 06), normalizou-se a barra de cores no notebook 02 e corrigiu-se a documentação. Na Fase B (diagnósticos do solver nas 14 instâncias de desenvolvimento), o experimento B1 comparou escala 10 000 contra 100 e escolheu 10 000 (posição média 1,32 contra 1,68, com R110 caindo de 12 para 11 veículos). O B2 avaliou o custo fixo relativo k ∈ {1, 10, 100} e manteve k = 10 por posição média, com frota idêntica nos três níveis. O B3 testou o `reduce_fleet` corrigido (partida a quente e descarte penalizado), que igualou o SINTEF em R202 (4 para 3 veículos), mas obteve posição média inferior (1,68 contra 1,32) e manteve `fixed_cost`. O B4 comparou 4 contra 8 processos e obteve frota idêntica em 42/42 células. O B5 aplicou partida a quente entre os níveis da camada B e zerou a não monotonicidade (6 para 0 e 14 para 0). Assim, a configuração v2 ficou definida pela configuração v1 com escala inteira 10 000 e custo fixo relativo k = 10, cuja magnitude equivale a cerca de 50% a 66% do custo fixo absoluto da v1.

Na Fase D, conduziu-se a reavaliação pós-hoc da v2 nas 42 instâncias de avaliação (já abertas, rotulada "v2, pós-hoc" ao lado da v1, sem substituí-la), com 210/210 execuções factíveis. A média de veículos extras caiu de 0,50 para 0,48, elevando as instâncias com frota igual ao SINTEF de 24 para 25 em razão de uma única mudança em RC207 (4 para 3 veículos). Em frota igual, a distância oscilou nos dois sentidos (de −4,3% a +2,1%, com 18 de 41 instâncias dentro de ±0,5%, 11 mais curtas e 12 mais longas), de modo que a queda da mediana do gap de 3,40% para 2,23% decorre dessas oscilações e da entrada de RC207 no conjunto de frota igual. A expectativa prévia de ganho em janelas apertadas como R106 não se confirmou (13 veículos nas duas versões). Como corrigido em 2026-11-19, a v2 difere da v1 tanto na escala quanto na magnitude do custo fixo (49,9% a 66,2% do valor da v1 em unidades de distância, mediana 57,7%), razão pela qual a mudança em RC207 é atribuída à configuração v2 como um todo. Embora a v2 corrija um defeito comprovado de truncamento, o impacto agregado é marginal e a conclusão principal da v1 permanece inalterada.

### Fase C: robustez e cenários operacionais (camada B, sem comparação com o SINTEF)

Documentada no notebook 07 (`reports/fase-c-robustez.md`), esta fase compara cada cenário ao plano de referência da própria instância, usando 30 s e 3 réplicas nas 14 instâncias de desenvolvimento para C2 e C3. No bloco C1 (robustez dos planos fixos congelados da avaliação sob atrasos aleatórios em 1 000 sorteios por plano e cenário), choques de viagem de +0–10% causaram atraso em 6,3% dos clientes do OR-Tools contra 9,3% do vizinho mais próximo e 21,1% da gulosa, além de 13,4% de retornos tardios ao depósito contra 32,9% e 43,3%; por rota, o OR-Tools acumulou mais rotas com algum atraso sob choque de serviço (84,2% contra 71,1% e 72,5%). A fragilidade associou-se à folga nominal (correlação de Spearman entre −0,37 e −0,78).

Nos cenários de restrição com o solver, reduzir a capacidade para 90% e 80% de Q (C2.1) exigiu em média 0,21 e 0,57 veículo adicional por instância (3 e 6 de 14 instâncias), elevando a distância em cerca de 14% quando a frota cresceu. Com frota limitada (C2.2), a penalidade uniforme com K − 1 deixou 111 clientes sem atendimento (23 prioritários), enquanto a penalidade prioritária deixou 122 clientes e 0 prioritários; com K − 2, a contagem foi de 344 (69 prioritários) contra 363 (3 prioritários), havendo pequena variação entre réplicas (como 9, 7 e 7 descartados em RC108 com K − 2 uniforme), pois a penalidade incentiva o atendimento sem garantir mínimo exato. O bloco de janelas flexíveis (C2.3) teve caráter exploratório e baixa informatividade, resolvendo apenas 3 de 14 instâncias (4 infactíveis por capacidade e 7 com falha na construção inicial), o que impede conclusões gerais sem ressalva explícita. Por fim, na reotimização após remoção ou inserção de 10% dos clientes (C3, avaliado na réplica representativa conforme correção de 2026-11-19), a partida a quente realocou 26,7% (remoção) e 30,0% (inserção) dos clientes comuns, contra 40,6% e 36,7% na partida a frio; embora a mediana do patamar de ruído seja 0%, a remoção apresentou realocação em 4 de 42 comparações entre réplicas a frio (até 23,3% em R111 e RC203), confirmando custo operacional relevante ao replanejar.

## Limitações

O estudo baseia-se em um benchmark sintético com distância euclidiana no plano, velocidade unitária e frota homogênea, cujos valores de referência do SINTEF são melhores soluções conhecidas e não ótimos provados. Limites de tempo por relógio tornam os resultados sensíveis à carga da máquina: com 30 s ocorreram variações de ±1 veículo em instâncias R e RC, ao passo que com 120 s em máquina ociosa a frota manteve-se estável.

No torneio inicial de desenvolvimento, a estratégia `reduce_fleet` falhou quase de imediato na construção da primeira redução com `PARALLEL_CHEAPEST_INSERTION` e `SAVINGS` (cerca de 10 s totais dos 30 s disponíveis; apenas `PATH_CHEAPEST_ARC` consumiu todo o passo de 5 s), descartando o restante do orçamento e enviesando a comparação original contra essa estratégia; a afirmação anterior de que o passo de 5 s era curto foi corrigida em 2026-11-18, e o experimento B3 repetiu depois a comparação com partida a quente e descarte penalizado a 120 s, mantendo `fixed_cost` por posição média superior. Nos cenários de negócio v1, os níveis foram resolvidos a frio e produziram 20 células não monotônicas (6 em janelas alargadas e 14 em entrega antecipada), eliminadas na rodada com partida a quente do experimento B5, ainda que as tabelas principais do notebook 06 preservem a v1. A reavaliação v2 pós-hoc reutiliza instâncias já conhecidas e tem papel puramente descritivo ao lado da v1.

Na Fase C, o experimento C2.3 de janelas flexíveis resolveu apenas 3 de 14 instâncias devido à falha na construção inicial em 7 das 10 instâncias viáveis por capacidade; as simulações de C1 usam choques independentes sem correlação espacial ou temporal de trânsito real, e os blocos C2 e C3 restringem-se às 14 instâncias de desenvolvimento. Os cenários de negócio da camada B tratam ganhos de duração como efeito colateral da minimização de frota e distância. Registra-se também um desvio formal em relação à regra da pré-análise que restringia a leitura de valores SINTEF das instâncias de avaliação ao torneio final: os notebooks 01 e 02 leem os 56 valores publicados exclusivamente de forma descritiva para estudar convenções e limites de frota, sem qualquer uso na escolha de configurações, orçamentos ou parâmetros.

## Documentação

A pasta `reports/` reúne a pré-análise de viabilidade, a avaliação da proposta, os protocolos pré-registrados (grade, avaliação, v2, fase C e fase D) com seus resultados, o plano de sanidade da auditoria, o relatório de robustez, o estudo de sensibilidade ao tempo e às janelas e o cenário de entrega antecipada. O arquivo `docs/Solomon-html descriptions.pdf` guarda a página salva do SINTEF com a tabela de referência.

## Referências

As obras citadas neste documento e nos notebooks, no formato autor-ano, constam na lista curada em [`docs/referencias.md`](docs/referencias.md), além da seção de referências ao fim de cada notebook.
