# Referências bibliográficas do projeto (lista curada)

Fonte única das citações usadas nos notebooks e READMEs. Formato da citação no texto: **(Autor, ano)**;
dois autores: (A & B, ano); três ou mais: (A et al., ano). Entradas no estilo APA (7ª ed.).
**Só estas referências são citadas**; um conceito sem referência nesta lista não recebe citação.

## Mapa conceito → citação

| Conceito no texto | Citação |
|---|---|
| Problema de roteamento de veículos (VRP), origem | (Dantzig & Ramser, 1959) |
| VRPTW: definição, janelas rígidas, capacidade; visão geral | (Toth & Vigo, 2014) |
| Instâncias de Solomon (C, R, RC; séries 1 e 2); heurística do vizinho mais próximo orientada ao tempo e seus pesos; heurísticas de inserção | (Solomon, 1987) |
| Melhores soluções conhecidas e convenção hierárquica (veículos, depois distância; precisão dupla) | (SINTEF, 2008) |
| Convenção de distância truncada em 1 casa e soluções do conjunto Solomon no CVRPLIB | (CVRPLIB, s.d.) |
| Objetivo hierárquico frota-primeiro em heurísticas de VRPTW; heurísticas construtivas e de busca local para VRPTW | (Bräysy & Gendreau, 2005a) |
| Metaheurísticas para VRPTW (visão geral, inclui busca tabu, recozimento, GLS) | (Bräysy & Gendreau, 2005b) |
| OR-Tools (solver de roteamento usado) | (Perron & Furnon, 2026) |
| Heurística das economias (`SAVINGS`) | (Clarke & Wright, 1964) |
| Busca local guiada (`GUIDED_LOCAL_SEARCH`) | (Voudouris & Tsang, 1999) |
| Busca tabu (`TABU_SEARCH`) | (Glover, 1986) |
| Recozimento simulado (`SIMULATED_ANNEALING`) | (Kirkpatrick et al., 1983) |
| Regra do prazo mais cedo (base da heurística gulosa por prazo) | (Jackson, 1955) |
| Encaixe no primeiro que couber (*first-fit*) | (Johnson, 1974) |
| Folga de tempo / atraso máximo da partida sem perder factibilidade (saída ajustada do depósito) | (Savelsbergh, 1992) |
| Janelas de tempo flexíveis com atraso penalizado | (Taillard et al., 1997) |
| Atendimento opcional com penalidade / problemas com lucros (descarte penalizado de clientes) | (Archetti et al., 2014) |
| Tempos de viagem e serviço estocásticos; robustez de rotas | (Gendreau et al., 1996) |
| Roteamento dinâmico e reotimização | (Pillac et al., 2013) |
| Método de Monte Carlo | (Metropolis & Ulam, 1949) |
| Correlação de postos de Spearman | (Spearman, 1904) |
| Índice de Rand (concordância entre pares; `pair_agreement`) | (Rand, 1971) |
| Desenho e relato de experimentos computacionais com heurísticas (réplicas, orçamento de tempo, relato de variabilidade) | (Barr et al., 1995) |
| Pré-registro de análises | (Nosek et al., 2018) |
| Vazamento de informação (*leakage*) na avaliação | (Kaufman et al., 2012) |

Limite inferior de frota por capacidade (⌈Σq/Q⌉): citar (Toth & Vigo, 2014). Métricas definidas neste
projeto (densidade de compatibilidade temporal, réplica representativa lexicográfica, contagem de não
monotonicidade) **não recebem citação** e devem ser apresentadas como "definida neste projeto".

## Entradas

- Archetti, C., Speranza, M. G., & Vigo, D. (2014). Vehicle routing problems with profits. In P. Toth & D. Vigo (Eds.), *Vehicle routing: Problems, methods, and applications* (2nd ed.). SIAM.
- Barr, R. S., Golden, B. L., Kelly, J. P., Resende, M. G. C., & Stewart, W. R. (1995). Designing and reporting on computational experiments with heuristic methods. *Journal of Heuristics, 1*(1), 9–32.
- Bräysy, O., & Gendreau, M. (2005a). Vehicle routing problem with time windows, Part I: Route construction and local search algorithms. *Transportation Science, 39*(1), 104–118.
- Bräysy, O., & Gendreau, M. (2005b). Vehicle routing problem with time windows, Part II: Metaheuristics. *Transportation Science, 39*(1), 119–139.
- Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research, 12*(4), 568–581.
- CVRPLIB. (s.d.). *VRPTW instances — Solomon set*. PUC-Rio. https://galgos.inf.puc-rio.br/cvrplib/
- Dantzig, G. B., & Ramser, J. H. (1959). The truck dispatching problem. *Management Science, 6*(1), 80–91.
- Gendreau, M., Laporte, G., & Séguin, R. (1996). Stochastic vehicle routing. *European Journal of Operational Research, 88*(1), 3–12.
- Glover, F. (1986). Future paths for integer programming and links to artificial intelligence. *Computers & Operations Research, 13*(5), 533–549.
- Jackson, J. R. (1955). *Scheduling a production line to minimize maximum tardiness* (Research Report 43). Management Science Research Project, University of California, Los Angeles.
- Johnson, D. S. (1974). Fast algorithms for bin packing. *Journal of Computer and System Sciences, 8*(3), 272–314.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science, 220*(4598), 671–680.
- Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. *Journal of the American Statistical Association, 44*(247), 335–341.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606.
- Perron, L., & Furnon, V. (2026). *OR-Tools* (Version 9.15) [Computer software]. Google. https://developers.google.com/optimization/
- Pillac, V., Gendreau, M., Guéret, C., & Medaglia, A. L. (2013). A review of dynamic vehicle routing problems. *European Journal of Operational Research, 225*(1), 1–11.
- Rand, W. M. (1971). Objective criteria for the evaluation of clustering methods. *Journal of the American Statistical Association, 66*(336), 846–850.
- Savelsbergh, M. W. P. (1992). The vehicle routing problem with time windows: Minimizing route duration. *ORSA Journal on Computing, 4*(2), 146–154.
- SINTEF. (2008). *VRPTW — 100 customers: Solomon benchmark, best known solutions*. SINTEF TOP. https://www.sintef.no/projectweb/top/vrptw/100-customers/
- Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research, 35*(2), 254–265.
- Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology, 15*(1), 72–101.
- Taillard, É., Badeau, P., Gendreau, M., Guertin, F., & Potvin, J.-Y. (1997). A tabu search heuristic for the vehicle routing problem with soft time windows. *Transportation Science, 31*(2), 170–186.
- Toth, P., & Vigo, D. (Eds.). (2014). *Vehicle routing: Problems, methods, and applications* (2nd ed.). SIAM.
- Voudouris, C., & Tsang, E. (1999). Guided local search and its application to the traveling salesman problem. *European Journal of Operational Research, 113*(2), 469–499.
