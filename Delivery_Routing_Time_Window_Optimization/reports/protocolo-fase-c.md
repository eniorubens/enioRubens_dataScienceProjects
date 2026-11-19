# Protocolo pré-registrado — Fase C: robustez e cenários operacionais (Camada B)

Registrado em 2026-11-18, antes de implementar ou executar. Origem: achados 1 e 2 da auditoria
(`reports/plano-sanidade-auditoria.md`) e seções 2 e 6 de `reports/pre-analise-proposta.md`.
**Camada B:** nenhuma comparação com o SINTEF; cada cenário é comparado com o próprio plano de referência.
Todos os resultados são descritivos (não há seleção de configuração nesta fase).

Regras gerais: sementes fixas e registradas; verificador em precisão dupla; máquina sem outra carga;
experimentos com solver em sequência; configuração do solver = **v2** (`fixed_cost` + `SAVINGS` + GLS,
escala 10 000, k = 10), 30 s por execução, 3 réplicas, réplica representativa lexicográfica (2ª de 3).

## C1 — Robustez do plano fixo sob choques (sem solver)

Pergunta: quão frágeis são planos já otimizados quando a execução real atrasa, **sem replanejar**?

- Planos: os planos congelados da avaliação v1 (`outputs/frozen/evaluation/results.csv`): réplica mediana do
  OR-Tools e as duas baselines, nas 42 instâncias de avaliação. (Uso descritivo de planos finais; nada é
  ajustado.)
- Execução simulada: o veículo segue a sequência planejada; começa o serviço em `max(chegada, ready)`;
  chegada depois de `due` é registrada como **atraso** e o serviço acontece mesmo assim; partida do
  depósito ajustada como no plano (saída ótima de Savelsbergh calculada no plano nominal).
- Choques (por trecho/cliente, independentes): tempo de viagem × (1 + ε), ε ~ U(0, a), com
  a ∈ {0,05; 0,10; 0,20}; tempo de serviço × (1 + η), η ~ U(0; 0,20); e a combinação viagem a = 0,10 +
  serviço. **1 000 sorteios por plano e cenário**, semente 20261118.
- Métricas por plano: fração de clientes atendidos com atraso; fração de rotas com ≥ 1 atraso; atraso
  médio e P95 entre os atrasados; fração de retornos ao depósito após o horizonte. Agregação por família
  e por método.
- Explicação: folga temporal do plano nominal (`due − início` por cliente) contra a taxa de atraso.

## C2 — Cenários operacionais com o solver (somente 14 instâncias dev)

1. **Capacidade reduzida:** Q × {1,0; 0,9; 0,8}; modelo rígido. Métrica: veículos adicionais, distância
   (separada por classe de frota).
2. **Frota limitada com descarte penalizado e prioridade:** K = (veículos do plano de referência com
   Q × 1,0) − 1 e − 2; `AddDisjunction` em todo cliente. 20% dos clientes marcados como prioritários
   (sorteio por instância, semente 20261118) com penalidade 10× a dos comuns; comparado com o mesmo K sem
   prioridade (penalidade uniforme). Penalidades acima do custo de qualquer desvio para que o solver só
   descarte quando a frota impede. Métricas: clientes não atendidos (total e prioritários), demanda não
   atendida, distância.
3. **Janelas flexíveis com atraso penalizado:** K = referência − 1; fim de janela macio
   (`SetCumulVarSoftUpperBound`) com custo por unidade de atraso c ∈ {baixo, alto} (definidos em relação
   ao custo por unidade de distância: 1× e 10×); retorno ao depósito continua rígido. Métricas: clientes
   atrasados, atraso total e máximo, distância.

## C3 — Estabilidade da reotimização (somente 14 instâncias dev)

- **Remoção:** sortear 10% dos clientes (semente fixa); reotimizar sem eles de duas formas: (i) partida a
  quente a partir do plano de referência sem esses clientes, (ii) a frio.
- **Inserção:** planejar com 90% dos clientes; inserir os 10% restantes e reotimizar, a quente (plano dos
  90% como partida, com descarte não permitido) e a frio.
- Métricas: `reallocation` (clientes que mudam de veículo, pares mantidos, trechos mantidos) entre o plano
  original e o reotimizado sobre os clientes comuns; variação de veículos e distância; patamar de ruído =
  duas réplicas a frio do mesmo cenário.

## Orçamento estimado

C1 alguns minutos (sem solver); C2 ≈ 25 min; C3 ≈ 11 min. Total ≈ 40 min de relógio.

## Saída esperada

Relatório `reports/fase-c-robustez.md`, dados congelados em `outputs/frozen/phase_c/` e atualização dos
notebooks (seção nova no NB06 ou NB07 dedicado — decidido na implementação, sem alterar os números).

## Esclarecimentos (registrados antes da execução, 2026-11-18)

Pontos ambíguos ou com defeito identificados na implementação, fixados **antes de rodar**:

1. **Réplica representativa por bloco.** A regra geral (factível, veículos, distância) favoreceria, em
   C2.2 e C2.3, um plano que descarta mais clientes ou atrasa mais. Por isso:
   - C2.2 (frota limitada e prioridade): ordenar por (factível, prioritários descartados, clientes
     descartados, demanda descartada, veículos, distância);
   - C2.3 (janelas flexíveis): ordenar por (factível, clientes atrasados, atraso total, veículos, distância);
   - demais blocos e planos de referência: regra geral. A ordenação usada fica registrada nos metadados.
2. **C2.3 nas famílias C.** Com K = referência − 1, a frota fica abaixo do limite por capacidade
   ⌈Σq/Q⌉ em C103, C106, C203 e C206 (frota de referência já no limite): o problema é infactível por
   capacidade, o que janela flexível não resolve. Essas células são mantidas, marcadas
   `capacity_infeasible = True` e reportadas à parte — não como falha de busca. Em C2.2, nessas instâncias,
   K − 1 e K − 2 só são atendíveis com descarte, o que é o objeto do bloco.
3. **Penalidade de descarte (C2.2).** Base `P = F + escala·D_ub + 1` (F = custo fixo por veículo;
   `D_ub = Σ 2·d(0,i)`): atender um cliente a mais nunca custa mais que abrir um veículo e uma ida e volta,
   então o solver só descarta quando a frota impede. Prioritários: 10·P.
4. **Custo do atraso (C2.3).** Custo de arco = `round(d·escala)`, logo uma unidade de distância escalada
   custa 1; "1× e 10× o custo por unidade de distância" = custo por unidade de atraso escalada de 1 e 10.
   Como o custo fixo por veículo domina esses valores, o solver pode usar menos que K veículos; isso é
   reportado.
5. **Plano de referência (C2/C3)** = configuração v2 com **30 s** e 3 réplicas na instância original,
   calculado no bloco `capacity` (as linhas v2/B1 têm 120 s e não são equivalentes).
6. **Inserção a quente (C3).** O OR-Tools rejeita partida com clientes obrigatórios fora de rota; a
   partida é o plano dos 90% completado por inserção de menor custo factível (aritmética inteira do
   solver), sem descarte. Registrado como `warm_start_input = built`.
7. **Sorteios.** Prioritários e amostra de 10% com semente derivada de (20261118, instância, finalidade),
   registrados nas linhas; remoção e inserção usam a mesma amostra. C1 usa números aleatórios comuns entre
   cenários (mesma semente por instância e método).
