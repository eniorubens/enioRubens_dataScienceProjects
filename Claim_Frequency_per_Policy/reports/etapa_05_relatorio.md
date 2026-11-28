# Etapa 5 — Avaliação final e apresentação

`production_eligible = false`. Depois do congelamento, uma única execução oficial avaliou os **101.859 registros** da partição de teste. Foram usados o Poisson com offset fixo e a taxa global de referência definidos na etapa 4, sem novo ajuste. Os resultados agregados estão em `outputs/etapa_05/`; o notebook é `04_avaliacao_final.ipynb` (PT-BR e EN-US).

## 1. Integridade antes do teste

O portão (`stage5.integrity_gate`, 20 verificações em `integrity_gate.json`) foi satisfeito **antes** de as linhas de teste serem disponibilizadas ao avaliador. O leitor do ARFF interpreta o arquivo integral (inclusive a verificação prévia `verify-freeze`, que usa esse leitor e devolve somente a validação); o campo `test_rows_read = 0` do registro do portão significa zero linhas de teste disponibilizadas ao avaliador, não ausência de leitura física do ARFF:

| Verificação | Resultado |
| --- | --- |
| SHA-256 do modelo congelado contra o valor aprovado no parecer da etapa 4 (`8ac94b7e…efe80a`) | confere |
| SHA-256 do roteiro contra o valor aprovado (`c0365ad5…c1ed8`) | confere |
| ARFF, atribuições, configurações de partição e protocolo, e os seis arquivos de origem da etapa 3, contra os hashes do modelo | conferem |
| Hashes dos oito módulos registrados no modelo | conferem |
| Módulos complementares aprovados: `freeze.py` `ad44566d…af31c`, `arff.py` `a24d0fb2…a923`, `profiles.py` `58411364…aca3` | conferem |
| `verify-freeze` (código de saída 0, 77 agregados dentro da tolerância, `code_modules_unchanged = true`) | passou |

Os valores aprovados são constantes no código (`stage5.APPROVED`), independentes dos arquivos verificados. Os módulos congelados não foram alterados: a etapa só adiciona `stage5.py` e extensões não congeladas (`artifacts.py`, `plots.py`, `labels.py`, `cli.py`).

**Execução única.** A trava `official_run.lock` é criada de forma atômica depois do portão e antes de o teste ser disponibilizado ao avaliador. Uma segunda chamada é recusada (código 4) sem disponibilizar o teste (verificado em execução real e em teste sintético). A conferência do número de linhas de teste (101.859) se refere às linhas disponibilizadas ao avaliador. Uma falha depois dessa disponibilização preservaria os artefatos e gravaria o ponto de falha em `official_run.json`. Nenhuma linha de teste é persistida: só agregados.

**Ensaio sem tocar o teste.** Antes da execução oficial, `stage5.evaluate` foi executada sobre a **validação** real e reproduziu exatamente os números da etapa 4 (desvio 0,318988; diferença pareada −0,009638 com IC idêntico). A execução ponta a ponta com fixtures sintéticas cobre integridade, adulteração do modelo, do roteiro, do código e dos dados, trava, falha simulada, reconciliação e verificação offline.

## 2. Execução oficial (`official_run.json`)

Comando: `python -I -m freq_sinistros --root . stage5`, código de saída 0, início 2026-11-21 17:12:21 UTC, fim 17:13:49 UTC. Tempo de parede 88,5 s; pico de memória 236 MiB (CPU). Teste: 101.859 registros (igual à etapa 2), 53.839,09 apólices-ano, 5.415 sinistros, 96.726 zeros; SHA-256 dos IDs `11808b81…83fb8`. 1.000 réplicas, semente 20261121. O registro guarda também o hash dos módulos usados (conferidos depois, todos iguais).

## 3. Comparação primária (`test_metrics.csv`, `test_intervals.csv`)

| Candidato | Desvio médio de Poisson | Log-verossim. média | Taxa prevista /100 | Prevista − observada |
| --- | ---: | ---: | ---: | ---: |
| Taxa global | 0,33079 | −0,21663 | 10,084 | +0,027 |
| Poisson (offset fixo) | 0,31902 | −0,21074 | 10,108 | +0,050 |

A taxa observada é 10,058 por 100 apólices-ano. A diferença pareada Poisson − taxa global é **−0,01177 (IC 95% −0,01317; −0,01018)**; na sensibilidade por perfil, **−0,01177 (−0,01338; −0,01028)**. Como ambos os intervalos ficam inteiramente abaixo de zero, o teste **sustenta um ganho** do Poisson sobre a referência em desvio médio. Os intervalos são condicionados aos modelos ajustados.

A calibração global foi avaliada pela diferença entre as taxas prevista e observada de cada candidato. Cada IC 95% se refere à diferença **prevista − observada**: Poisson −0,233 a +0,317 e taxa global −0,253 a +0,288. Ambos incluem zero. Esses intervalos individuais não testam equivalência entre os candidatos nem comparam seus erros absolutos, cujos valores pontuais são 0,050 (Poisson) e 0,027 (taxa global). Nenhum teste adicional foi criado sobre a partição reservada.

## 4. Calibração por segmento (`panel.csv`, `test_calibration.csv`)

Diferença prevista − observada por 100 apólices-ano, Poisson, previsões diretas e exposição total de cada faixa. Faixa de exposição (teste; validação entre colchetes):

| Faixa | Registros | Observada | Prevista | Diferença (IC 95%) | Validação |
| --- | ---: | ---: | ---: | --- | ---: |
| `(0; 0,02]` | 2.921 | 255,51 | 14,04 | −241,5 (−296,7; −190,3) | −224,8 |
| `(0,02; 0,08]` | 14.341 | 39,81 | 12,40 | −27,4 (−31,6; −23,4) | −23,8 |
| `(0,08; 0,25]` | 16.168 | 21,62 | 12,05 | −9,6 (−11,4; −8,0) | −11,6 |
| `(0,25; 0,50]` | 19.859 | 12,99 | 11,41 | −1,6 (−2,4; −0,8) | −0,9 |
| `(0,50; 0,75]` | 13.877 | 10,83 | 11,40 | +0,6 (−0,1; +1,3) | +1,3 |
| `(0,75; 1)` | 9.300 | 8,28 | 10,91 | +2,6 (+2,0; +3,3) | +1,7 |
| `= 1` | 25.222 | 6,79 | 8,69 | +1,9 (+1,6; +2,2) | +1,8 |
| `> 1` (baixo suporte) | 171 | 9,42 | 11,31 | +1,9 (sem IC) | +8,5 |

O padrão da validação se repete: o sinal coincide nas 8 faixas, e a diferença de validação cai dentro do intervalo de teste em 5; `(0,08; 0,25]` e `(0,75; 1)` ficam fora (a faixa `> 1`, de baixo suporte, não tem intervalo). A subestimação na faixa mais curta persiste e é o desvio dominante.

Idade do veículo: os intervalos distinguem de zero `6-10` (−0,48), `11-20` (+0,47) e `21+` (+2,88); `0-1` e `2-5` incluem zero. Região: só `R82` (+0,85) e `R91` (+2,01) se distinguem de zero; `R21`, `R42` e `R43` têm baixo suporte e ficam sem intervalo. Com tantas comparações sem ajuste de multiplicidade, esses desvios são descritivos.

## 5. Reconciliação (`test_reconciliation.csv`)

24 verificações (2 candidatos × 3 segmentações × 4 indicadores): a soma dos segmentos coincide, para cada candidato, com 101.859 registros, 53.839,09 apólices-ano, 5.415 sinistros observados e os previstos (Poisson 5.442,15; taxa global 5.429,36). Todas `ok`.

## 6. Conclusão

O teste **sustenta o ganho** do Poisson com offset fixo sobre a taxa global em desvio médio de Poisson; portanto, a escolha congelada é preservada. Esse ganho não elimina o desvio sistemático nas exposições curtas, que reaparece com o mesmo comportamento observado em validação.

Na calibração total, o erro absoluto pontual do Poisson é maior que o da referência. Como os intervalos individuais incluem zero, eles não permitem concluir equivalência nem diferença entre os candidatos. A reescala por faixa da etapa 4 permaneceu como diagnóstico de validação e não foi aplicada. As conclusões se restringem a este benchmark.

## 7. Notebooks, documentação e verificação

- Notebook `04_avaliacao_final.ipynb` nas duas línguas: a primeira célula apresenta o problema, a importância da calibração por segmento e a pergunta; lógica e gráficos em `src/freq_sinistros/` (`stage5.py`, `artifacts.py`, `plots.py`); consome os mesmos agregados e verifica a origem (`verify_stage5`) sem ler o ARFF. Os dois notebooks finais foram regenerados e executados também em uma cópia **sem** o ARFF e as atribuições (`python scripts/run_notebooks.py --final`), com 7/7 células por idioma e código de saída 0.
- READMEs: `README.pt-BR.md` e `README.en-US.md` (mesma estrutura, mesmas conclusões e mesmos comandos, com teste de paridade), e `README.md` como índice. Os agregados em `outputs/` são versionáveis; o ARFF e as atribuições ficam em `data/` e não são versionados.
- Execução dos oito notebooks (quatro pares) em kernel limpo, com código de saída 0: `scripts/run_notebooks.py`, registros em `outputs/etapa_04/notebook_execution.json` (03) e `outputs/notebook_execution_etapa05.json` (04); os de 01 a 03 em `outputs/etapa_03/notebook_execution.json`.
- Auditoria de 2026-11-23 (sem novas leituras do teste e sem mudar números de resultados): `check-stage5` agora recusa com código 5, sem gravar nada, quando `official_run.lock` existe (o portão regravaria artefatos selados); `verify-stage5` (CLI) combina `stage5.verify_official` e `artifacts.verify_stage5`, de modo que CLI e notebook usam a mesma conferência completa; os registros de execução dos notebooks ficam em `outputs/notebook_execution_etapa0{3,4,5}.json`, fora das pastas seladas.
- Ruff sem erros; pytest: 107 testes na entrega; 200 após as correções da auditoria de 2026-11-23 (cobertura 98%); CI **configurado** (`.github/workflows/ci.yml` corrigido: o nome do passo continha `: ` sem aspas e o parser o rejeitava; agora o arquivo é validado por `yaml.safe_load` e por `tests/test_ci_config.py`), com Ruff e pytest sobre dados sintéticos; estado de execução: **não executado** nesta etapa. Atualização de 2026-11-23: na publicação como pasta `Claim_Frequency_per_Policy` do repositório de portfólio, o `ci.yml` próprio foi substituído pela suíte do projeto no workflow raiz `Tests` (`.github/ci/test-projects.json`), conferida por `tests/test_ci_config.py`.

## 8. Limitações

O offset fixo subestima as exposições muito curtas e superestima as longas. Resultados internos ao benchmark, sem evidência temporal, sem `BonusMalus` e sem ZIP/ZINB. Intervalos condicionados aos modelos ajustados. O estado do CI não foi verificado.
