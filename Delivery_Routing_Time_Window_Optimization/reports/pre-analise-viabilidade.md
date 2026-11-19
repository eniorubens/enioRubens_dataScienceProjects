# Pré-análise de viabilidade — Delivery Routing & Time-Window Optimization

Data: 2026-11-17 · Fase: auditoria de dados, contrato e ambiente (sem notebooks nem solver).

## 1. O que o benchmark mede (e o que não mede)

**Mede:** a qualidade de soluções para o VRPTW em 56 instâncias **sintéticas** de Solomon (1987), com
100 clientes, um depósito, frota homogênea, distância euclidiana no plano e tempo de viagem igual à
distância. Classes: C (clientes agrupados), R (aleatórios), RC (mistos). Séries: 1 (horizonte curto,
poucos clientes por rota) e 2 (horizonte longo, rotas longas).

**Não mede:** rede viária, trânsito, tempo variável, custo de motorista, frota heterogênea, pedidos
dinâmicos, janelas múltiplas ou violação tolerada com penalidade. Não é telemetria logística real.

## 2. Contrato de dados (verificado nas 56 instâncias)

| Campo | Unidade / regra | Verificação |
|---|---|---|
| `VEHICLE NUMBER` (K) | limite superior da frota | K = 25 em todas |
| `CAPACITY` (Q) | unidades de demanda | 200 (C1, R1, RC1), 700 (C2), 1000 (R2, RC2) |
| Nó 0 | depósito; demanda 0; `DUE DATE` = fim do horizonte | horizonte: C1 1236, C2 3390, R1 230, R2 1000, RC1 240, RC2 960 |
| `XCOORD`, `YCOORD` | inteiros, plano euclidiano | distância = tempo de viagem (velocidade 1) |
| `DEMAND` | inteiro ≥ 0 | — |
| `READY TIME` / `DUE DATE` | janela rígida para o **início** do serviço | chegada antecipada espera; chegada após `DUE DATE` é infactível |
| `SERVICE TIME` | constante por família | 90 (C), 10 (R, RC) |
| IDs | 0..100 contíguos | verificado |

Regra de retorno: o veículo deve voltar ao depósito até o `DUE DATE` do nó 0.
Instâncias de 25 e 50 clientes, por convenção, são os primeiros 25/50 clientes do arquivo de 100
(ainda não baixadas; o SINTEF as publica em páginas separadas).

## 3. Proveniência

| Fonte | Conteúdo | SHA-256 |
|---|---|---|
| CVRPLIB (PUC-Rio), set Solomon | 56 instâncias + 56 soluções `.sol` | `55ba67b5…2a37` (instâncias), `88a5d385…394e` (soluções) |
| SINTEF TOP, `solomon-100.zip` (download manual) | 56 instâncias, sem soluções | `8a0a72cb…7747` |
| SINTEF TOP, página "100 customers" salva em PDF ([docs/Solomon-html descriptions.pdf](../docs/Solomon-html%20descriptions.pdf)) | tabela de BKS + convenção, publicada em 18/04/2008 | transcrita em [data/reference/sintef_bks_100.csv](../data/reference/sintef_bks_100.csv) |

As instâncias das duas fontes são **numericamente idênticas** (diferenças só de espaçamento).
Licença: nenhuma licença explícita encontrada; usar com citação de Solomon (1987, *Operations Research*
35(2)), SINTEF TOP e CVRPLIB. O SINTEF bloqueia acesso automatizado (Cloudflare), portanto o CI não
pode baixar dados de lá: os arquivos brutos precisam ser versionados no repositório ou obtidos do CVRPLIB.

## 4. Convenções de objetivo — risco central de comparabilidade

| | SINTEF | CVRPLIB |
|---|---|---|
| Objetivo | hierárquico: 1º nº de veículos, 2º distância total | **somente distância total** (rotas ≤ K = 25) |
| Distância e tempo | **double precision**; distância total arredondada a 2 casas | euclidiana **truncada em 1 casa decimal** |
| Status das referências | melhores conhecidas (heurísticas, 1995–2009), conferidas pelo *solution checker* do SINTEF | marcadas como ótimas sob essa convenção (métodos exatos) |

O próprio SINTEF afirma (convenção confirmada na página salva): métodos exatos costumam usar objetivo
monolítico de distância e cálculo inteiro ou de baixa precisão, *"hence, results are not directly
comparable"*. A página também registra que valores antigos de R101, R104, R106, R110, R208, R210,
RC201, RC204 e RC205 eram ligeiramente melhores por **baixa precisão aritmética** e foram corrigidos.

Cruzamento SINTEF × CVRPLIB (56 instâncias):

- em **37** instâncias o SINTEF usa **menos veículos** (até 5 a menos, ex.: R202 3 vs 8, RC201 4 vs 9),
  ao custo de distância maior (+7,9 em R101 até +270,0 em RC202); em **19** a frota coincide e em nenhuma o SINTEF usa mais;
- nas 19 de frota igual (C1, C2, RC103, RC104), recalcular as rotas do CVRPLIB com distância exata
  reproduz **exatamente** o valor do SINTEF (diferença 0,00) — validação cruzada do verificador;
- portanto as famílias C são o único ponto em que as duas convenções coincidem; R e RC divergem quase sempre.

Evidência medida com verificador independente ([scripts/audit_cvrplib_bks.py](../scripts/audit_cvrplib_bks.py)):

- as 56 soluções do CVRPLIB são factíveis e reproduzem exatamente o custo publicado **com distância truncada**;
- **8 das 56 tornam-se infactíveis com distância exata** (violam janelas): R102, R105, R107, R108,
  R112, R211, RC101, RC105;
- sob distância exata, o custo das mesmas rotas sobe de 1,6 a 5,2 unidades;
- R101: o CVRPLIB usa 20 rotas (1637,7); o SINTEF usa 19 veículos (1650,80).

**Consequência:** um resultado só pode ser comparado à referência calculada sob a **mesma** função
objetivo e a **mesma** regra de distância. Misturar as duas convenções é um erro metodológico, não um
detalhe de arredondamento.

## 5. Ambiente local

| Item | Situação |
|---|---|
| CPU | i9-11900H, 16 threads — adequado |
| RAM | 15,7 GB — folgada (matriz 101×101) |
| GPU | RTX 3050 Laptop, 4 GB — irrelevante para OR-Tools (CPU) |
| Python global | 3.15.1 |
| OR-Tools | 9.15 publica wheels apenas até cp314 → não instala no 3.15 |
| numpy / pandas / pytest | ausentes no Python global |

**Resolvido (2026-11-17):** ambiente conda `delivery-routing-vrptw` (Miniforge, padrão dos demais
projetos) com Python 3.13.18. Especificação em `environment.yml`, versões exatas em
`environment.lock.yml` (ortools 9.15.6755, numpy 2.5.5, pandas 3.1.1, matplotlib 3.11.2,
pytest 9.2.0). Teste de fumaça do roteamento OR-Tools executado com sucesso.

Observações do ambiente:
- o OR-Tools **não é publicado no conda-forge** (só no canal `defaults` da Anaconda); instalado via
  seção `pip:` do `environment.yml`, com a wheel oficial do PyPI;
- o `.condarc` do usuário lista `rapidsai` e `defaults` antes de `conda-forge`; o ambiente foi criado
  com `CONDA_CHANNELS=conda-forge` e `CONDA_CHANNEL_PRIORITY=strict` (só nesta execução) para ficar
  100% conda-forge (200 pacotes conda-forge + 4 PyPI);
- neste ambiente de execução as conexões HTTPS exigem `CONDA_SSL_VERIFY=truststore` (certificado
  intermediário não reconhecido pelo bundle do conda).

Reproduzir:

```
set CONDA_CHANNELS=conda-forge
set CONDA_CHANNEL_PRIORITY=strict
conda env create -f environment.yml
```

A estimativa de tempo por família (passo 7 do briefing) ainda não foi feita.

## 6. Status do gate de continuidade

| Critério | Status |
|---|---|
| Dados reproduzíveis | ✅ duas fontes, hash registrado, conteúdo idêntico |
| Parser testado | ✅ `delivery_routing_vrptw.io` com testes sintéticos (64 testes, cobertura 99%, Ruff limpo) |
| Solução factível validada | ✅ `verify` reproduz a auditoria (8 BKS CVRPLIB infactíveis em double; 19 valores SINTEF ao centésimo); 18/18 soluções do gate factíveis |
| Comparação objetiva com o benchmark | ✅ possível: tabela SINTEF transcrita e convenção confirmada (hierárquica, double, 2 casas) |

**Gate executado (2026-11-17)** — `scripts/run_gate.py --time-limit 10`, uma instância de
desenvolvimento por família (split com semente 20261117 sorteado antes de qualquer execução):

| Instância | Gulosa (veíc. / dist.) | NN-TW (veíc. / dist.) | OR-Tools 10 s (veíc. / dist.) | SINTEF |
|---|---|---|---|---|
| C103 | 12 / 3472,14 | 11 / 2078,88 | **10** / 853,83 | 10 / 828,06 |
| C203 | 5 / 3127,54 | 4 / 1765,66 | **3** / 600,21 | 3 / 591,17 |
| R108 | 21 / 3059,11 | 12 / 1324,15 | 10 / 976,40 | 9 / 960,88 |
| R201 | 6 / 2719,21 | 5 / 2150,38 | **4** / 1283,08 | 4 / 1252,37 |
| RC106 | 18 / 2678,60 | 14 / 1882,69 | 14 / 1530,84 | 11 / 1424,73 |
| RC203 | 7 / 3344,55 | 4 / 1919,57 | 4 / 1017,93 | 3 / 1049,62 |

Leitura: com 10 s o OR-Tools iguala a frota de referência em 3/6 (gap de distância 1,5–3,1%), e fica
+1 veículo em R108 e RC203 e +3 em RC106. A distância menor em RC203 não é ganho: usa um veículo a mais.
O gargalo é a **minimização de frota**, não a distância — confirma o risco previsto na pré-análise e
orienta a grade de desenvolvimento (custo fixo × laço de redução de frota).

Limitação: as soluções detalhadas do SINTEF (links por instância) não foram baixadas; para R e RC a
comparação será por valor (veículos, distância), não por rota.

## 7. Pendências e decisões

1. ~~Salvar a página do SINTEF~~ — feito (PDF em `docs/`, tabela em `data/reference/`).
2. ~~Criar ambiente Python 3.13 com OR-Tools fixado~~ — feito (conda `delivery-routing-vrptw`, `environment.lock.yml`).
3. Decidir a convenção pública: recomendo **SINTEF hierárquica com distância exata** como principal e
   CVRPLIB truncada apenas como análise secundária, sempre em tabelas separadas.
4. OR-Tools aceita apenas custos inteiros: será preciso escalar distâncias (ex.: ×1000) e documentar
   o erro de arredondamento resultante, verificando a solução final com distância em double.
5. Definir o split desenvolvimento/avaliação (decisão aberta 1 do briefing) antes de qualquer ajuste.
