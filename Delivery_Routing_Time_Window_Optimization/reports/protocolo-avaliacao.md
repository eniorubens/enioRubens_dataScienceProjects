# Protocolo pré-registrado — avaliação final

Registrado em 2026-11-17, **antes** de qualquer execução sobre as instâncias de avaliação. Decisão do
orçamento tomada pelo usuário com base em `reports/sensibilidade-tempo-e-janelas.md` (curva de tempo nas
instâncias de desenvolvimento: ganho até ~120 s, estável depois; variabilidade de ±1 veículo por relógio).

## Escopo

- Instâncias: as **42** com `set = eval` em `data/reference/split.csv`. Nenhuma foi usada antes para
  escolher configuração, orçamento ou parâmetro.
- Configuração: a congelada em `outputs/frozen/selected_config.json` — `fixed_cost` + `SAVINGS` +
  `GUIDED_LOCAL_SEARCH`, sem nenhuma alteração.
- **Orçamento: 120 s** por execução (o torneio usou 30 s; a mudança é declarada e foi decidida nas
  instâncias de desenvolvimento).
- **Réplicas: 5** por instância (variabilidade por relógio). 42 × 5 = 210 execuções do OR-Tools, 8 processos,
  máquina sem outra carga.
- Baselines (gulosa por prazo e vizinho mais próximo com janelas): 1 execução por instância (determinísticas).
- Toda solução é verificada em precisão dupla; métricas vêm do verificador.
- Camada A apenas (convenção SINTEF). Nenhum cenário de negócio é executado nas instâncias de avaliação
  nesta etapa.

## Métricas e resultado principal (fixados antes de ver resultados)

1. Por instância, as 5 réplicas são ordenadas lexicograficamente por (factível, veículos, distância).
   **Resultado principal = réplica mediana (3ª de 5).** A melhor das 5 é reportada à parte, rotulada como
   "melhor de 5", nunca como resultado principal.
2. Contra o SINTEF, por instância: Δ veículos; gap de distância % **somente** quando a frota é igual;
   indicador lexicográfico "supera a referência".
3. Agregados (geral e por família): média de Δ veículos; nº de instâncias com frota igual à referência;
   mediana do gap de distância entre as de frota igual; nº de instâncias que superam a referência.
4. Variabilidade: amplitude de veículos e de distância entre as 5 réplicas por instância.
5. Baselines: as mesmas métricas, para situar o ganho do OR-Tools.
6. Nenhuma alegação de otimalidade; diferenças agregadas são descritas, sem teste de significância
   (não pré-registrado).

## O que não será feito

- Não haverá nova seleção de configuração, ajuste de parâmetro ou reexecução seletiva após ver os
  resultados de avaliação. Se uma execução falhar por erro técnico (não por qualidade), ela pode ser
  repetida, e isso será registrado.

---

## Resultados (executado em 2026-11-17, após o registro acima)

Execução: `scripts/run_evaluation.py --confirm-eval --workers 8 --out outputs/evaluation` — 294 células
(210 OR-Tools + 84 baselines), **294/294 factíveis** no verificador em double, 54 min, tempo médio por
execução do OR-Tools 120,0 s. Nenhuma reexecução, nenhuma alteração de configuração. Artefatos congelados
em `outputs/frozen/evaluation/`.

Ressalva de execução: durante a avaliação um agente editou o notebook 04 sob a regra de não executar
notebooks, testes ou solvers; o uso de CPU foi residual. A frota não variou entre réplicas em nenhuma
instância (ver abaixo), o que indica que essa atividade não afetou o resultado principal.

### Resultado principal (réplica mediana) contra o SINTEF

| Escopo | Instâncias | Δ veículos médio | Frota igual | Mediana do gap (frota igual) | Supera a referência |
|---|---|---|---|---|---|
| **Geral** | 42 | **0,50** | **24** | 3,40% | 0 |
| C1 | 7 | 0,00 | 7 | 0,00% | 0 |
| C2 | 6 | 0,00 | 6 | 0,00% | 0 |
| R1 | 9 | 0,89 | 1 | 0,52% | 0 |
| R2 | 8 | 0,25 | 6 | 5,02% | 0 |
| RC1 | 6 | 1,50 | 0 | — | 0 |
| RC2 | 6 | 0,33 | 4 | 4,63% | 0 |

- Distribuição de Δ veículos: 0 em 24 instâncias, +1 em 15, +2 em 3 (RC101, RC102, RC105); nunca abaixo
  da referência.
- Famílias C: frota igual nas 13; **10 reproduzem a distância de referência ao centésimo**; C104 (+7,7%),
  C109 (+7,9%) e C204 (+3,6%) ficam acima.
- R1: só R101 iguala a frota (19 veículos, gap 0,52%); as demais ficam com +1.
- RC1 é a família mais difícil: nenhuma iguala a frota (+1 a +2).
- Distâncias menores que a referência (ex.: R103, R204, R211, RC202, RC207) ocorrem sempre com mais
  veículos e não são ganho na convenção hierárquica.

### Melhor de 5 (reportado à parte)

Idêntico à mediana em frota (Δ médio 0,50; 24 com frota igual); mediana do gap entre as de frota igual
2,65% contra 3,40%.

### Variabilidade entre réplicas

- **Frota: amplitude 0 em todas as 42 instâncias** — com 120 s e a máquina sem outra carga, o número de
  veículos foi estável.
- Distância: amplitude mediana 0,00%; máxima 3,21% (R203); variação em 6 de 42 instâncias (R203, R206, R209, R211, RC204, RC205).

### Baselines

| Método | Δ veículos médio | Frota igual |
|---|---|---|
| OR-Tools (configuração congelada, mediana) | 0,50 | 24 |
| Vizinho mais próximo com janelas | 1,36 | 12 |
| Gulosa por prazo | 3,98 | 2 |

### Consistência com o desenvolvimento

Nas instâncias de desenvolvimento a mesma configuração com 120 s teve Δ médio 0,64 e frota igual em 7/14;
na avaliação, 0,50 e 24/42. O padrão por família se repete: C sem perda de frota, R1 e RC1 com +1 a +2,
séries 2 próximas da referência em frota com gap de distância de 3% a 8%.
