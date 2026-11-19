# Protocolo pré-registrado — Fase D: reavaliação v2 (pós-hoc)

Registrado em 2026-11-18, antes de qualquer execução da v2 nas instâncias de avaliação.

## Natureza

As 42 instâncias de avaliação **já foram vistas** na avaliação v1 (`reports/protocolo-avaliacao.md`).
Esta reavaliação é **pós-hoc**: não é um conjunto intocado. A v1 permanece como resultado original; a v2 é
apresentada ao lado, nunca no lugar, e nenhuma escolha entre as duas é feita olhando os resultados.

## Desenho (idêntico ao da v1, salvo a configuração)

- Instâncias: as 42 `eval` de `data/reference/split.csv`.
- Configuração **v2** (`reports/protocolo-v2.md`): `fixed_cost` + `SAVINGS` + `GUIDED_LOCAL_SEARCH`,
  **escala 10 000**, custo fixo relativo k = 10. Única diferença material em relação à v1: a escala.
- 120 s × 5 réplicas, 8 processos, máquina sem outra carga.
- Baselines: **não reexecutadas** (determinísticas e independentes da configuração do OR-Tools); reutilizam-se
  as linhas da v1.
- Resultado principal: réplica mediana lexicográfica (3ª de 5); "melhor de 5" à parte.
- Métricas: as mesmas da v1 (Δ veículos vs SINTEF, frota igual, gap só com frota igual, variabilidade entre
  réplicas) + duração/espera derivadas das rotas.

## Comparação v1 × v2 (fixada antes de ver resultados)

- Por instância: diferença de veículos (v2 − v1) e, com frota igual, diferença de distância.
- Agregados geral e por família, lado a lado.
- Nenhum teste de significância; leitura descritiva. Diferenças de distância < 0,5% entre versões são
  tratadas como dentro da variação observada entre réplicas e execuções.
- Expectativa registrada (não usada para decidir): ganho concentrado em instâncias de janelas apertadas
  (ex.: R106), onde a escala 100 podava rotas válidas.

---

## Resultados (executado em 2026-11-18, após o registro acima)

210/210 execuções factíveis, 54 min, 120,0 s por execução; frota idêntica nas 5 réplicas em todas as 42
instâncias (amplitude máxima de distância 3,0%). Dados em `outputs/frozen/evaluation_v2/`.

| Escopo | Δ veículos médio vs SINTEF (v1 → v2) | Frota igual ao SINTEF (v1 → v2) | Mediana do gap, frota igual (v1 → v2) |
|---|---|---|---|
| **Geral** | 0,50 → **0,48** | 24 → **25** | 3,40% → 2,23% |
| C1 / C2 | 0,00 → 0,00 | 13 → 13 | 0,00% → 0,00% |
| R1 | 0,89 → 0,89 | 1 → 1 | 0,52% → 0,52% |
| R2 | 0,25 → 0,25 | 6 → 6 | 5,02% → 4,95% |
| RC1 | 1,50 → 1,50 | 0 → 0 | — |
| RC2 | 0,33 → **0,17** | 4 → **5** | 4,63% → 6,29% |

- **Frota:** muda em uma única instância — **RC207, 4 → 3**, igualando o SINTEF. A expectativa registrada
  (ganho em R106) **não se confirmou**: R106 fica com 13 veículos nas duas versões.
- **Distância (frota igual nas duas versões, 41 instâncias):** 18 dentro de ±0,5%; das 23 restantes, 11
  melhoram e 12 pioram (de −4,3% a +2,1%). Não há efeito sistemático da escala sobre a distância. A queda da
  mediana do gap (3,40% → 2,23%) vem dessas oscilações em ambos os sentidos e da mudança de composição do
  conjunto de frota igual (RC207 entra), não de uma melhora consistente.
- **Leitura:** a escala 10 000 corrige um defeito real (rotas válidas podadas, demonstrado em R106/R110 e com
  efeito em R110 e RC207), mas seu impacto agregado na avaliação é **marginal**. A conclusão da v1 se mantém:
  famílias C e séries 2 próximas do SINTEF; **R1 e RC1 com +1 a +2 veículos** são o limite desta abordagem.

---

## Correção (2026-11-19)

O texto acima **não foi reescrito**. Duas afirmações precisam de correção:

- Em "Desenho": "Única diferença material em relação à v1: a escala." A v2 difere da v1 na escala **e** na
  magnitude do custo fixo: v1 = 10^7 absoluto com escala 100 (10^5 unidades de distância); v2 = `10 × escala ×
  D_ub` (10·D_ub unidades de distância), que nas 42 instâncias de avaliação vale de 49,9% a 66,2% do custo fixo da
  v1 (mediana 57,7%; notebook 05, seção 11).
- Na "Leitura": "a escala 10 000 corrige um defeito real … com efeito em R110 e RC207". O defeito da escala 100
  (rotas válidas podadas em R106/R110) é demonstrado, e o B1 isolou a escala (k = 10 nos dois níveis). A mudança de
  RC207 na reavaliação, porém, é atribuível à **configuração v2** (escala e custo fixo juntos): a comparação
  pós-hoc é entre duas configurações e não isola a escala. O B2 (mesma frota para k ∈ {1, 10, 100} nas instâncias
  dev) sugere, mas não prova, que o custo fixo não foi o que moveu RC207. A conclusão agregada (impacto marginal;
  a v1 permanece o resultado principal) não muda.
