# Protocolo pré-registrado v2 — diagnósticos do solver (Fase B)

Registrado em 2026-11-18, **antes** de implementar ou executar qualquer experimento desta fase.
Origem: `reports/plano-sanidade-auditoria.md` (achados 4, 5, 6 e 8 da auditoria externa). A v1
(tag `v1`, `outputs/frozen/`) permanece congelada e não é substituída.

## Escopo e regras gerais

- **Somente as 14 instâncias de desenvolvimento** (`split.csv`, `set = dev`). Nenhuma instância de
  avaliação é lida ou resolvida nesta fase.
- Base: configuração congelada v1 (`fixed_cost` + `SAVINGS` + `GUIDED_LOCAL_SEARCH`), **120 s**,
  **3 réplicas** por célula, 8 processos, máquina sem outra carga, experimentos em sequência (nunca dois ao
  mesmo tempo).
- Toda solução é verificada em precisão dupla contra a instância original; métricas vêm do verificador.
- **Resultado por instância** = réplica representativa lexicográfica (ordenar réplicas por
  (factível, veículos, distância) e tomar a posição 2 de 3).
- **Comparação entre níveis** = mesma regra do torneio v1: posição média lexicográfica (veículos, depois
  distância) por instância, sobre as 14 instâncias; desempates por Δ veículos médio vs SINTEF e depois
  mediana do gap entre as de frota igual.

## Experimentos e regra de decisão

### B1 — Escala inteira (achado 5)

- Níveis: `scale` ∈ {100, 10 000}. Custo fixo por veículo **proporcional à escala** (ver B2), para que só
  a escala mude: nível de dominância 10× (equivale ao 10^7 da v1 com escala 100).
- Decisão: escolhe-se o nível de melhor posição média; **empate → 10 000** (garante factibilidade em
  double sem podar rotas válidas, demonstrado para R106/R110).
- Verificação adicional registrada: R110 reduz frota com escala 10 000?

### B2 — Custo fixo por veículo (achado 8)

- Definição independente da escala: `vehicle_fixed_cost = k × scale × D_ub`, com
  `D_ub = Σ_i 2·d(0, i)` (limite superior simples da distância total). Níveis k ∈ {1, 10, 100}
  (k = 10 é o nível de B1 e é reutilizado, não reexecutado).
- Escala: a escolhida em B1.
- Decisão: menor posição média; **empate → manter k = 10** (padrão v1).

### B3 — Redução de frota corrigida (achado 4)

- `minimize_fleet` v2: primeira resolução com 1/3 do orçamento; depois, a cada passo, **partida a quente**
  a partir do melhor plano, esvaziando a rota com menos clientes, com `AddDisjunction` de penalidade muito
  maior que o custo fixo durante a redução; o passo é aceito só se o plano final servir todos os clientes
  (verificado). Passos de 20 s; orçamento **total** de 120 s respeitado; o tempo não usado num passo
  aceito passa ao próximo; o laço não descarta orçamento por falha rápida (tenta esvaziar a próxima menor
  rota até esgotar o tempo).
- Comparação: `reduce_fleet` v2 × `fixed_cost`, ambos com a escala de B1 e o k de B2, 120 s.
- Decisão: `reduce_fleet` v2 só substitui `fixed_cost` se tiver posição média **estritamente** melhor.

### B4 — Carga da máquina / *throttling* (achado 8)

- Configuração escolhida após B1–B3, executada com `workers` = 4; comparada com a mesma configuração com
  `workers` = 8 (reutilizada de B1/B2/B3, conforme o caso).
- Pergunta descritiva: muda frota ou distância? Sem decisão automática; se a frota mudar em alguma
  instância, a avaliação v2 usará 4 processos.

### B5 — Partida a quente entre níveis da Camada B (achado 6)

- Repetir os cenários de janelas alargadas (fatores 1,0/1,25/1,5/2,0) e de entrega antecipada
  (0/25%/50%/100%/livre; 3 réplicas) com **30 s** (o mesmo orçamento da v1), cada nível partindo da solução
  do nível anterior (que é factível no nível mais permissivo por construção), com a configuração v1 —
  para isolar o efeito da partida a quente.
- Medida: número de células lexicograficamente não monótonas (v1: 6 + 14), veículos economizados e
  variação de duração/distância por classe de frota.
- Descritivo; não altera a configuração da Camada A.

## Configuração v2 e próxima decisão

A configuração v2 = escala de B1 + k de B2 + estratégia de frota de B3 (+ número de processos de B4).
Se a v2 for idêntica à v1, nada muda. Se diferir, a reavaliação nas 42 instâncias (Fase D) só ocorre
por decisão do usuário e será rotulada **"v2, pós-hoc"**, apresentada ao lado da v1, sem substituí-la.

## Orçamento estimado (8 processos, máquina ociosa)

| Bloco | Execuções | Tempo de relógio |
|---|---|---|
| B1 (2 escalas × 14 × 3 × 120 s) | 84 | ~21 min |
| B2 (k = 1 e 100) | 84 | ~21 min |
| B3 (`reduce_fleet` v2) | 42 | ~11 min |
| B4 (`workers` = 4) | 42 | ~21 min |
| B5 (janelas 56 + antecipação 210, 30 s) | 266 | ~17 min |
| **Total** | | **~1,5 h** |

## Esclarecimentos (registrados antes da execução, 2026-11-18)

Pontos ambíguos do texto acima, resolvidos durante a implementação e **fixados antes de rodar B1**:

1. **Ordem dos desempates em B1/B2.** Aplica-se a regra geral completa (posição média; depois Δ veículos
   médio vs SINTEF; depois mediana do gap entre as de frota igual). A preferência do protocolo
   (B1 → 10 000; B2 → k = 10) só decide se os três critérios empatarem. `decision.json` registra qual
   critério decidiu (`decided_by`).
2. **"Estritamente melhor" em B3** = posição média estritamente menor de `reduce_fleet` v2, sem desempates.
3. **Tempo em B3.** O tempo não usado numa tentativa (aceita ou com falha rápida) passa à seguinte; o laço
   para quando nenhuma rota candidata é esvaziada com sucesso ou o orçamento acaba; rotas já tentadas no
   mesmo plano não são repetidas.
4. **Réplicas em B5.** Janelas alargadas com 1 réplica (como na v1, 56 células); entrega antecipada com
   3 réplicas; cada réplica percorre sua própria escada de níveis.
5. **Comparação B4.** Feita por instância contra as linhas de 8 processos da mesma configuração já
   produzidas em B1/B2/B3.

---

## Resultados (executado em 2026-11-18, após o registro e os esclarecimentos acima)

Todos os blocos em sequência, máquina sem outra carga; todas as soluções factíveis no verificador
(B1 84/84, B2 84/84, B3 42/42, B4 42/42, B5 56 + 210). Dados em `outputs/frozen/v2/`.

| Bloco | Níveis (posição média) | Decisão (regra) |
|---|---|---|
| B1 escala | 10 000: 1,32 · 100: 1,68 | **10 000** (posição média) |
| B2 custo fixo | k=10: 1,89 · k=100: 1,96 · k=1: 2,14 | **k = 10** (posição média) |
| B3 frota | `fixed_cost`: 1,32 · `reduce_fleet` v2: 1,68 | **`fixed_cost`** (v2 não é estritamente melhor) |
| B4 processos | frota idêntica em 42/42 células; distância difere em 9 células (C203, R111, RC106) | **8 processos** (frota não mudou) |

- **B1:** com escala 10 000, Δ veículos médio vs SINTEF 0,57 (contra 0,64) e mediana do gap em frota igual
  1,22% (contra 1,53%). **R110 cai de 12 para 11 veículos** nas 3 réplicas — confirma o achado 5.
- **B2:** a frota é a mesma nos três níveis de k; k = 1 só piora a distância. A hipótese de que o custo
  fixo alto prejudica a redução de frota (achado 8) **não se confirmou**.
- **B3:** `reduce_fleet` v2 iguala a frota do SINTEF em **R202 (4 → 3)**, levando a 8/14 instâncias com
  frota igual e Δ médio 0,50; mas, gastando tempo em tentativas de redução, piora a distância em outras
  (ex.: RC203 +12%, R207 +4%) sem ganhar veículos, e perde na posição média. R1 e RC1 continuam com +1 a +2
  veículos em ambas.
- **B4:** sem efeito na frota; pequenas diferenças de distância (ex.: C203 591,17 com 4 processos — igual
  ao SINTEF — contra 594,85 com 8). A carga afeta a distância marginalmente, não a frota.
- **B5:** a partida a quente **elimina a não monotonicidade**: janelas 6 → 0, entrega antecipada 14 → 0
  (lexicográfica). Veículos economizados com antecipação livre: 0,71 (v2) contra 0,50 (v1); janelas ×2:
  0,86 contra 0,79.

### Configuração v2

`fixed_cost` + `SAVINGS` + `GUIDED_LOCAL_SEARCH`, **escala 10 000**, k = 10 (custo fixo relativo),
120 s, 8 processos. Difere da v1 apenas na escala (e na definição relativa do custo fixo, de magnitude
equivalente). Reavaliação nas 42 instâncias (Fase D): **decisão do usuário**, rotulada "v2, pós-hoc".

---

## Correção (2026-11-19)

Duas frases deste protocolo estão erradas e **não foram reescritas** (o texto original acima fica como foi
registrado):

- B1: "nível de dominância 10× (equivale ao 10^7 da v1 com escala 100)". A v1 usava o custo fixo **absoluto**
  10^7 com escala 100, isto é, 10^5 unidades de distância; a v2 usa `10 × escala × D_ub` (D_ub = Σ 2·d(0, i)),
  isto é, 10·D_ub unidades de distância. Nas 42 instâncias de avaliação o custo fixo da v2 vale de **49,9% a
  66,2%** do da v1 em unidades de distância (mediana 57,7%; `final_eval.fixed_cost_ratio_table`, notebook 05,
  seção 11). Não são equivalentes.
- "Configuração v2": "Difere da v1 apenas na escala (e na definição relativa do custo fixo, de magnitude
  equivalente)". A v2 difere da v1 na escala **e** na magnitude do custo fixo.

Consequência: a comparação v1 × v2 (Fase D) é uma comparação de **duas configurações**; a mudança de RC207
(4 → 3 veículos) é atribuível à configuração v2 como um todo, e não isolada à escala. A evidência controlada
sobre a escala é o B1 (os dois níveis usaram a mesma regra relativa k = 10); o B2 encontrou a mesma frota com
k ∈ {1, 10, 100} nas instâncias dev, o que sugere, mas não prova, que a mudança do custo fixo não foi o que moveu
RC207 (instância de avaliação, fora do B2).
