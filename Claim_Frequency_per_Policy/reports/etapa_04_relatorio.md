# Etapa 4. Diagnóstico de zeros, decisão de modelo e congelamento

`production_eligible = false`. Esta etapa transforma os resultados provisórios da validação em uma decisão congelada, sem abrir o teste. Só treino e validação foram usados: o leitor interpreta o ARFF integral em memória, `load_partitions` devolve apenas as partições pedidas e recusa `test`, e a verificação do congelamento registra `test_used = false`. Nenhum ZIP ou ZINB foi ajustado. As saídas estão em `outputs/etapa_04/`; o notebook é `03_diagnostico_zeros.ipynb` (PT-BR e EN-US).

## 1. Reprodução e origem

O reajuste determinístico no treino inteiro (474.581 registros, mesmo código e mesma amostra da etapa 3) reproduziu os resultados aprovados: diferença relativa máxima das métricas 2,0e-12; coeficientes do Poisson iguais aos de `coefficients_ntrain_full.csv` a 2,1e-12; coeficiente livre de `log(Exposure)` 0,38110 (diferença 3,5e-13) (`reproduction_check.json`). O hash da amostra de treino coincide com o degrau `train_full` da etapa 3. As saídas determinísticas de `stage4` são idênticas em duas execuções.

## 2. Decisão de modelo (`model_decision.csv`)

Regra aprovada, partição principal por `IDpol`, maior degrau viável (`train_full`):

| Degrau | Desvio Poisson − global (IC 95%) | Erro abs. de taxa: global / Poisson / NB2 | Poisson > global | NB2 > Poisson | Preferido |
| --- | --- | --- | :---: | :---: | --- |
| 100 mil | −0,00888 (−0,01044; −0,00730) | 0,131 / 0,142 / 0,063 | não | não | taxa global |
| 200 mil | −0,00942 (−0,01091; −0,00792) | 0,010 / 0,020 / 0,177 | não | não | taxa global |
| **Treino inteiro** | **−0,00964 (−0,01111; −0,00815)** | **0,068 / 0,054 / 0,258** | **sim** | **não** | **Poisson** |
| Agrupada (sensibilidade, treino inteiro) | −0,01135 (−0,01291; −0,00994) | 0,257 / 0,244 / 0,055 | sim | sim | NB2 |

O Poisson com offset fixo melhora o desvio médio frente à taxa global (0,31899 contra 0,32863), com IC excluindo zero, e cumpre o critério pontual de erro absoluto no treino inteiro; o sinal do ganho é o mesmo nos três degraus, mas a condição de calibração global não se cumpre nos degraus de 100 e 200 mil (erro do Poisson maior que o da taxa global). O NB2 melhora a log-verossimilhança em +0,00075 (+0,00036; +0,00130), mas seu desvio médio é maior (+0,00011; +0,00003 a +0,00019) e seu erro absoluto de taxa global (0,258) supera o do Poisson (0,054), de modo que suas condições não são cumpridas e o ganho isolado de verossimilhança não basta. Na partição agrupada (sensibilidade com população própria de 101.676 registros voltada à generalização a perfis distintos), o NB2 cumpre as condições (diferença de desvio −0,00001, IC −0,00009 a +0,00007; log-verossimilhança +0,00140; erro 0,055 contra 0,244), sem alterar a decisão da partição principal.

## 3. Exposição (`exposure_band_rates.csv`)

Taxas por 100 apólices-ano das previsões diretas e da exposição total de cada faixa (validação, 101.573 registros, 53.590,05 anos):

| Faixa | Registros | Apólices-ano | Observada | Poisson, offset fixo (IC da diferença) | Poisson, coef. livre (diagnóstico) |
| --- | ---: | ---: | ---: | --- | --- |
| `(0; 0,02]` | 2.873 | 32,2 | 238,99 | 14,24 (−281,5; −169,6) | 134,56 (−160,9; −50,0) |
| `(0,02; 0,08]` | 14.314 | 916,8 | 36,21 | 12,44 (−28,3; −19,4) | 43,51 (+2,8; +11,6) |
| `(0,08; 0,25]` | 16.322 | 2.815,9 | 23,69 | 12,07 (−13,5; −9,9) | 23,39 (−2,1; +1,4) |
| `(0,25; 0,50]` | 19.693 | 7.714,7 | 12,37 | 11,44 (−1,8; −0,1) | 13,58 (+0,4; +2,0) |
| `(0,50; 0,75]` | 13.882 | 8.784,5 | 9,97 | 11,30 (+0,7; +2,0) | 10,04 (−0,6; +0,7) |
| `(0,75; 1)` | 9.263 | 8.079,5 | 9,00 | 10,66 (+1,0; +2,3) | 8,00 (−1,7; −0,4) |
| `= 1` | 25.042 | 25.042,0 | 6,90 | 8,70 (+1,5; +2,1) | 6,87 (−0,4; +0,3) |
| `> 1` (baixo suporte) | 184 | 204,5 | 3,42 | 11,89 (sem IC) | 7,32 (sem IC) |
| Total | 101.573 | 53.590,0 | 10,02 | 10,07 (−0,22; +0,30) | 10,10 (−0,18; +0,34) |

Na faixa mais curta, o offset fixo prevê 14,24 contra 238,99 observados. O coeficiente livre aproxima a previsão para 134,56, mas ainda deixa um desvio expressivo e eleva o erro absoluto da diferença de taxa global de 0,054 para 0,084. Assim, a melhora local não basta para promovê-lo a candidato. O ajuste livre permanece como **diagnóstico da associação com a duração observada**. Essa relação não sustenta extrapolação causal nem prospectiva da frequência para outra duração.

## 4. Zeros (`zero_diagnostics.csv`, `zero_reconciliation.csv`, `zero_gate.json`)

As faixas reconciliam com o total para os quatro candidatos (registros, exposição, zeros observados e previstos). `P(Y_i = 0) = exp(−μ_i)` para a taxa global, o Poisson e o diagnóstico livre; a forma NB2 com `α = 1,249` para o NB2. Total: 101.573 registros, 96.496 zeros observados (95,0%).

| Candidato | Zeros previstos | Previstos − observados (IC 95%) | IC abaixo de zero? | Faixas com suporte e IC abaixo de zero |
| --- | ---: | --- | :---: | --- |
| Taxa global | 96.374,9 | −121,1 (−247,7; +4,6) | não | `(0,75; 1)`, `= 1` |
| Poisson, offset fixo | 96.406,6 | −89,4 (−210,6; +43,0) | não | `(0,50; 0,75]`, `(0,75; 1)`, `= 1` |
| NB2 | 96.565,5 | +69,5 (−52,3; +200,7) | não | `(0,50; 0,75]`, `(0,75; 1)`, `= 1` |
| Poisson, coef. livre (diagnóstico) | 96.335,8 | **−160,2 (−283,1; −30,0)** | **sim** | `(0,02; 0,08]`, `(0,25; 0,50]` |

Pelo critério da etapa 2 aplicado ao modelo de referência (Poisson, offset fixo), o IC global inclui zero, então a parte numérica não é cumprida, ao passo que o diagnóstico com coeficiente livre cumpre a parte numérica (total e duas faixas com suporte), ficando a diferença registrada separadamente enquanto o critério se aplica ao modelo de referência. Com offset fixo, o sinal varia com a duração: os zeros são superestimados nas faixas curtas e subestimados nas longas. Conforme `mean_vs_zero_by_band.csv`, os erros nas médias coexistem com as diferenças de zeros (onde contagens são subestimadas, zeros são superestimados, e vice-versa); em diagnóstico nunca usado em taxas ou previsões, ao reescalar `μ` por faixa para somar as contagens observadas, a diferença de zeros por faixa cai para algumas dezenas, com saldo de −96,3 no Poisson e −115,1 no diagnóstico livre (cerca de 0,1% dos zeros), sem intervalo associado.

Sobre um segundo processo, o dicionário define `ClaimNb` e `Exposure`, mas não descreve apólices sem risco, não há datas nem identificador de unidade superior, e o instante de medição de `BonusMalus` é desconhecido; nenhuma hipótese plausível está documentada, e o teste de Vuong não é usado como regra (Wilson, 2015). A decisão desta rodada é encerrar com os modelos de contagem já avaliados, **sem ZIP nem ZINB**, pois falham as duas condições predefinidas (o modelo de referência não cumpre o critério numérico de déficit persistente de zeros e o contrato dos dados não oferece hipótese plausível para um segundo processo).

## 5. Congelamento (`frozen_model.json`)

O artefato congela o Poisson com offset fixo `log(Exposure)` (coeficiente 1), ajustado no treino inteiro (474.581 registros) sem `BonusMalus` com 68 colunas, além da taxa global de referência do mesmo treino (10,0844 por 100 apólices-ano, taxa `Σ ClaimNb / Σ Exposure` em valor exato no artefato). O arquivo armazena o pré-processador já ajustado (níveis, referências, cortes, medianas e padronização de `log(Density)`), ordem das colunas, coeficientes (17 dígitos, exatos na ida e na volta), tratamento da exposição (dado do cenário; `μ = Exposure · exp(x′β)`), política de níveis desconhecidos e ausentes, hashes do ARFF, das atribuições, da amostra de treino e dos arquivos de origem, protocolo e hashes dos módulos de código, além das versões do ambiente.

A verificação (`python -I -m freq_sinistros --root . verify-freeze`) restaura o estado persistido sem reajustar o estimador nem reaprender o pré-processamento (um teste sintético proíbe `fit`) e reproduz 77 agregados de validação com diferença absoluta máxima 0 e as métricas aprovadas da etapa 3, sob tolerância declarada de `rtol = 1e-9` e `atol = 1e-12`, com testes sintéticos cobrindo pré-processamento, ordem das colunas, coeficientes, offset (dobrar a exposição dobra `μ`) e adulteração (qualquer coeficiente alterado é detectado). Como limitação do modelo selecionado, o offset fixo subestima as exposições muito curtas (14,24 contra 238,99 em `(0; 0,02]`) e superestima as longas.

## 6. Roteiro da etapa 5 (`stage5_script.json`)

Uma única execução sobre o teste, após o aceite do Codex, com o Poisson congelado e a taxa global de referência: desvio médio de Poisson, diferença pareada contra a taxa global (IC 95% por bootstrap pareado por `IDpol`, 1.000 réplicas, semente 20261121; sensibilidade por perfil), calibração das previsões diretas por segmento (faixa de exposição, idade do veículo, região) com a faixa `(0; 0,02]` visível, limites de baixo suporte (500 registros, 25 apólices-ano), reconciliações com o total do teste (101.859 registros), painel e verificação dos hashes antes da execução. Limitações a declarar: viés por duração, benchmark, `BonusMalus` fora, sem ZIP/ZINB.

## 7. Verificação de código

`ruff check .` sem erros; `pytest` 85 testes, cobertura 94%; os seis notebooks executam em kernel limpo com paridade PT-BR/EN-US (`scripts/run_notebooks.py`, `outputs/etapa_04/notebook_execution.json`).

## 8. Limitações

Intervalos condicionados aos modelos ajustados. A reescala por faixa é apenas diagnóstica. O diagnóstico livre não é candidato e sua relação com a duração é associativa. Resultados internos ao benchmark.

## Errata (2026-11-23, auditoria)

O registro de execução dos notebooks, citado na seção 7 como `outputs/etapa_04/notebook_execution.json`, foi movido para fora da pasta selada como `outputs/notebook_execution_etapa04.json` (regravado por `scripts/run_notebooks.py`), porque o executor regravava o arquivo depois que a lista era gerada e a checagem deixava de conferir. A linha de `notebook_execution.json` foi removida de `outputs/etapa_04/SHA256SUMS_etapa04.txt` (hash antigo `773d5ceb6e09469fd039da8c692172e937f6ac023f306c49fbeb1e98f9b55cd4`; novo `caa52117c5962e8b10e7e5694766082bfe39c133a1b17236295178ee587095dc`), sem efeito sobre os resultados nem sobre os hashes congelados (`frozen_model.json` e `stage5_script.json` não mudaram).
