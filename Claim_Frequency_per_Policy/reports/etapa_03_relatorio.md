# Etapa 3 — EDA de treino e comparação em validação

`production_eligible = false`. Nesta etapa, a EDA do treino é ligada à comparação dos modelos em validação, sem abrir a partição de teste.

**Reserva do teste.** O leitor interpreta o ARFF integral em memória (as 678.013 linhas), mas `load_partitions` devolve apenas as partições pedidas — treino e validação —, recusa `test` e descarta o restante logo após a leitura. Nenhuma linha do teste foi usada na EDA, no ajuste ou na avaliação. `comparison_config.json` registra somente os tamanhos de treino e validação. As saídas estão em `outputs/etapa_03/`. O protocolo emendado foi gravado **antes** do primeiro ajuste em validação, em `reports/etapa_02_protocolo.md` (seção 11) e `outputs/etapa_03/protocol_amendment.json`.

## 1. Origem e reprodução

| Item | Valor |
| --- | --- |
| ARFF (SHA-256) | `a45363e056e2ea56408b38eeb9d4d04d7f6c6982eb7a14ed5e807c7c71807cdd` (confere com o manifesto da etapa 1) |
| Atribuições (SHA-256) | `f14f53509d63ff468f025fd43096a5c2d12ba3fdd48529e7644f003c73396ce6` (confere com a etapa 2) |
| Validação | 101.573 registros, 53.590,05 anos (iguais à etapa 2); 5.368 sinistros; 96.496 zeros |
| Treino | 474.581 registros |
| Ambiente | conda `freq-sinistros`, Python 3.12.16, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2, nbclient 0.11.0 |

```powershell
$env:PYTHONNOUSERSITE = "1"     # evita pacotes do perfil do usuário sobrepondo o ambiente
python -I -m freq_sinistros --root . eda        # EDA só com o treino
python -I -m freq_sinistros --root . compare    # ajustes, validação, sensibilidades (≈ 4 min)
python scripts/build_01_eda.py ; python scripts/build_02_compare.py   # gera os .ipynb
python scripts/run_notebooks.py                 # executa os quatro notebooks em kernel limpo
python -m pytest ; ruff check .
```

Dois problemas de ambiente foram resolvidos e documentados em `environment.yml`. O MKL 2026.1 do conda travava `numpy.linalg` nesta máquina (código de saída −1066598273), por isso a biblioteca BLAS foi trocada por OpenBLAS (`libblas=*=*openblas`). Além disso, o `platformdirs` do perfil do usuário (`AppData\Roaming\Python`) sobrepunha o do ambiente e quebrava o pytest. `PYTHONNOUSERSITE=1` evita essa sobreposição.

## 2. EDA do treino (somente treino)

Treino: 474.581 registros, 251.070,30 anos, 25.319 sinistros, taxa 10,084 por 100 apólices-ano, 94,97% de zeros. Tabelas em `eda_*.csv`, resumo em `eda_summary.json`, narrativa em `01_eda_treino.ipynb`.

- **Exposição curta persiste dentro de perfis.** Em 24 de 24 estratos (idade do condutor × idade do veículo) com suporte nos dois grupos, a taxa das linhas com `Exposure ≤ 0,08` supera a de referência (`Exposure ≥ 0,75`): razão mediana 4,2, mínima 2,7. Razão observado/esperado por faixa: 30,9 em `(0; 0,02]`, 4,8, 2,9, 1,6, 1,3, 1,1 e 0,96 em `= 1`.
- **Pressuposto do offset.** Coeficiente livre de `log(Exposure)` no treino: 0,39 (erro padrão ingênuo 0,016), contra 1 do offset.
- **Casos extremos.** 8 registros com `ClaimNb > 4` (71 sinistros), 5 deles com `DrivAge = 52` e `VehAge = 12` (44 sinistros). 869 registros com `Exposure > 1`. 56,7% dos registros com `BonusMalus = 50` (auditoria).
- **Cobertura de categorias.** Em 374.581 linhas de treino fora da amostra de 100 mil: 3.618 linhas (`Region`: R21, R42, R43) e 4.700 (`VehPower`: 13, 14, 15) mapeadas para `OTHER`; matriz finita.

## 3. Estimação

- **Pré-processamento** (`prep.py`, ajustado só nas linhas de ajuste de cada comparação): níveis com `< 500` linhas em `OTHER`; `VehPower` como categoria; `VehAge` e `DrivAge` em 10 faixas por quantis do ajuste; `log(Density)` padronizado; `IDpol` excluído; medianas do ajuste para numéricos ausentes. Matriz de desenho: 64 a 68 colunas, posto completo, número de condição escalado ≈ 331 (treino inteiro). Níveis, cortes e referências por degrau em `preprocessing_n*.json`.
- **Estimadores** (`models.py`): taxa global `Σ ClaimNb / Σ Exposure` na mesma amostra; Poisson por IRLS com offset `log(Exposure)` (coeficientes iguais aos do `statsmodels.GLM` a 1,8e-13; log-verossimilhança igual); NB2 por alternância entre IRLS em `β` e máxima verossimilhança unidimensional em `log α`, com verificação do escore. `NB2 → Poisson` quando `α → 0`: a log-verossimilhança usa uma forma estável (soma de logaritmos e `log1p`), testada contra o Poisson com `α = 1e-9`. O teste sintético confirma que dobrar a exposição dobra a contagem prevista para covariáveis fixas.
- **Amostras aninhadas** (SHA-256 dos `IDpol` ordenados): 100 mil `673862…b1a2`; 200 mil `573f0f…74fa`; treino inteiro `2a1dc6…e120`.

## 4. Convergência e custo (`fits.csv`, `cost.json`)

Todos os 19 ajustes convergiram e ficaram dentro de 600 s e 8 GiB. Maior degrau viável: **treino inteiro**; estabilidade verificada nos três degraus.

| Degrau | Poisson (s) | NB2 (s) | `α` (NB2) | Pico RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 100 mil | 0,31 | 1,13 | 1,287 | 266 |
| 200 mil | 0,71 | 2,75 | 1,246 | 409 |
| Treino inteiro (474.581) | 1,75 | 7,59 | 1,249 | 703 |

`α` está longe do limite de Poisson. Escore no ótimo abaixo de 1e-7 por linha (máximo 6,2e-8). A execução completa de `compare` levou 236 s (inclui 1.000 réplicas bootstrap por análise; as réplicas por perfil levaram 79 s); o pico do processo foi de 1.019 MiB (ajuste do diagnóstico com `log(Exposure)` livre). Duas execuções completas geraram saídas idênticas, exceto os tempos (`fits.csv`, `cost.json`). Os tempos acima são da última execução; a primeira registrou valores próximos.

**Inicialização e avisos.** O Poisson inicia com só o intercepto igual a `log(Σ ClaimNb / Σ Exposure)`; o NB2 inicia em `β` do Poisson, com `α` de máxima verossimilhança dado o `μ` do Poisson. Nenhum ajuste emitiu aviso (`warnings` vazio em `fits.csv`). Um `α` próximo de zero significaria variância igual à média (limite de Poisson) e seria sinalizado em `alpha_at_poisson_limit`; não ocorreu em nenhum degrau.

**Dispersão condicional** (Poisson ajustado, `dispersion_diagnostics.csv`; população = linhas do degrau, graus de liberdade = `n − p`): no treino inteiro, 474.581 linhas e 474.513 graus de liberdade, Pearson/gl = 2,40; estimador de Cameron e Trivedi (1990) `α̂` = 0,978 (t robusta = 7,1), contra `α` = 1,249 do NB2.

## 5. Validação (população: 101.573 linhas, valores originais)

Treino inteiro (`validation_metrics.csv`, `intervals.csv`):

| Candidato | Desvio médio | Log-verossim. média | Taxa prevista | Prevista − observada |
| --- | ---: | ---: | ---: | ---: |
| Taxa global | 0,32863 | −0,21514 | 10,084 | +0,068 |
| Poisson | 0,31899 | −0,21032 | 10,071 | +0,054 |
| NB2 | 0,31910 | −0,20957 | 10,275 | +0,258 |

Taxa observada: 10,017 por 100 apólices-ano. Diferenças pareadas (IC 95%, 1.000 réplicas): desvio Poisson − global **−0,00964 (−0,01111; −0,00815)**; desvio NB2 − Poisson **+0,00011 (+0,00003; +0,00019)**; log-verossimilhança NB2 − Poisson **+0,00075 (+0,00036; +0,00130)**. Diferença de taxa global do Poisson: +0,054 (−0,217; +0,304); do NB2: +0,258 (−0,015; +0,508). As taxas vêm direto das previsões; não houve recalibração.

**Regras do parecer (provisórias; a decisão é da etapa 4)** (`rules_table`):

| Degrau / população | Poisson > global | NB2 > Poisson | Preferido pelas regras |
| --- | :---: | :---: | --- |
| 100 mil | não | não | taxa global |
| 200 mil | não | não | taxa global |
| Treino inteiro | sim | não | Poisson |
| Agrupada, treino inteiro | — | sim | NB2 |

O ganho de desvio do Poisson sobre a taxa global mantém o mesmo sinal e IC favorável nos três degraus (−0,0089, −0,0094, −0,0096). A condição de erro absoluto da taxa global, porém, só é satisfeita no treino inteiro: nos degraus de 100 e 200 mil, o erro do Poisson (0,142 e 0,020) supera o da taxa global (0,131 e 0,010).

O NB2 melhora a log-verossimilhança, mas piora o desvio e o erro de taxa global. Por isso, não supera o Poisson na partição principal. Na partição agrupada, ele satisfaz as condições (diferença de desvio −0,00001, IC −0,00009 a +0,00007; log-verossimilhança +0,00140; erros de taxa +0,055 contra +0,244 do Poisson). A preferência, portanto, depende da população de avaliação e ainda requer a decisão da etapa 4.

## 6. Calibração (`calibration_by_segment.csv`)

Por faixa de exposição, diferença prevista − observada por 100 apólices-ano (Poisson; IC 95%): `(0; 0,02]` −224,8 (−281,5; −169,6); `(0,02; 0,08]` −23,8; `(0,08; 0,25]` −11,6; `(0,25; 0,50]` −0,9; `(0,50; 0,75]` +1,3; `(0,75; 1)` +1,7; `= 1` +1,8 (+1,5; +2,1); `> 1` +8,5 (baixo suporte, sem IC). Os três candidatos subestimam as faixas curtas e superestimam as longas, como previsto na emenda. Por idade do veículo e região, os desvios são pequenos e os intervalos em geral incluem zero (exceção: R26, −3,3 com IC −6,1 a −0,7). Segmentos de baixo suporte (`> 1`, R21, R42, R43) ficam visíveis e sem IC.

**Zeros (preliminar, sem decisão):** zeros previstos − observados no total: Poisson −89,4 (−210,6; +43,0), NB2 +69,5 (−52,3; +200,7), taxa global −121,1 (−247,7; +4,6). Nas faixas curtas os zeros são superestimados (por exemplo, +200,6 em `(0,02; 0,08]`) e nas longas subestimados (−433,7 em `= 1`), padrão que acompanha a subestimação das contagens curtas. Essa leitura vale para a taxa global, o Poisson e o NB2 com exposição proporcional, cujos IC globais cruzam zero.

**Zeros do diagnóstico com `log(Exposure)` livre (apresentados à parte; não é candidato):** zeros previstos − observados no total **−160,2 (−283,1; −30,0)**, com IC inteiramente abaixo de zero em `(0,02; 0,08]` (−78,0; −110,1 a −42,9) e em `(0,25; 0,50]` (−110,5; −166,3 a −51,9), ambas com suporte; `(0; 0,02]` e `(0,75; 1)` têm sinal positivo. O resultado mostra déficit de zeros previstos nesse diagnóstico, mas a hipótese plausível de um segundo processo é um requisito separado do portão de ZIP/ZINB e não está atendida por esse número.

## 7. Reconciliação (`reconciliation.csv`)

Cada população tem seus denominadores (`population_totals` em `artifacts.py`; verificações em `reconciliation.csv`, uma por candidato, segmentação e indicador):

| População | Registros | Apólices-ano | Sinistros observados | Candidatos | Verificações | Resultado |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Validação principal | 101.573 | 53.590,05 | 5.368 | 3 | 108 | todas `ok` |
| Sensibilidade `BonusMalus` (mesma validação) | 101.573 | 53.590,05 | 5.368 | 4 | 48 | todas `ok` |
| Sensibilidade limites no ajuste (mesma validação) | 101.573 | 53.590,05 | 5.368 | 5 | 60 | todas `ok` |
| Diagnóstico `log(Exposure)` livre (mesma validação) | 101.573 | 53.590,05 | 5.368 | 3 | 36 | todas `ok` |
| Validação agrupada | 101.676 | 53.674,22 | 5.520 | 3 | 36 | todas `ok` |
| Recorte descritivo sem `(0; 0,02]` | 98.700 | 53.557,83 | 5.291 | 3 | — | sem reconciliação por segmento; só resumo da métrica geral |

Os totais de registros e exposição da validação principal coincidem com `overview.json` da etapa 2; a reconciliação de **sinistros** da validação é feita aqui pela primeira vez, conforme corrigido no protocolo.

## 8. Sensibilidades (`sensitivity` em `validation_metrics.csv` e `intervals.csv`)

| Sensibilidade | Resultado (treino inteiro) |
| --- | --- |
| `BonusMalus` | desvio do Poisson −0,00539 (−0,00637; −0,00448); NB2 −0,00543. Disponibilidade indeterminada: serve só de sensibilidade. |
| Limites didáticos só no ajuste | desvio do Poisson +0,00003 (−0,00000; +0,00006): sem efeito relevante. |
| Bootstrap por perfil | mesmos sinais e ICs ligeiramente mais largos (por exemplo, Poisson − global −0,00964; −0,01112 a −0,00813). |
| Partição agrupada | 474.478 de ajuste, 101.676 de avaliação, 0 chaves em comum. Poisson − global −0,01135 (−0,01291; −0,00994); NB2 − Poisson em log-verossimilhança +0,00140. |
| Sem a faixa `(0; 0,02]` (descritivo) | 98.700 registros e 5.291 sinistros. O desvio **aumenta**: Poisson de 0,31899 para 0,31979 e taxa global de 0,32863 para 0,32876. A métrica primária segue sobre todas as linhas. |
| `log(Exposure)` livre (diagnóstico, não candidato) | coeficiente 0,38; diferença de desvio −0,01116 (−0,01290; −0,00938) frente ao Poisson com offset, maior que o ganho das covariáveis. Na faixa `(0; 0,02]` prevê 134,56 por 100 apólices-ano contra 238,99 observados (o Poisson com offset previu 14,24): melhora relevante, ainda com desvio expressivo. O erro absoluto da diferença de taxa global sobe de 0,054 para 0,084; a melhora por faixa e a variação do erro global devem ser lidas em conjunto. |

## 9. Notebooks e verificação

Quatro arquivos: `notebooks/{pt_br,en_us}/{01_eda_treino,02_comparacao_modelos}.ipynb`, gerados por `scripts/build_*.py` (narrativa bilíngue, código compartilhado e `LANG` por parâmetro). A lógica reside em `src/freq_sinistros/`. As interpretações têm voz passiva, citações `(Autor, ano)` e referências conferidas no Crossref e em catálogos: Cameron e Trivedi (1990, 2013); Dutang e Charpentier (2018, página CASdatasets); Fávero e Belfiore (**2017**, edição original com Excel, SPSS e Stata, GEN LTC, ISBN 978-85-352-7087-7, a mesma obra que o notebook 08.01 do MBA indica, capítulo 14); Noll, Salzmann e Wüthrich (**2020**). Para Noll et al., o DOI registra a primeira postagem em 2018 e a revisão de 2020; usa-se 2020, versão listada no dicionário CASdatasets, e o estudo é citado apenas como outro trabalho com o mesmo conjunto de dados (seu conteúdo não foi parafraseado). A execução e a paridade estão em `outputs/etapa_03/notebook_execution.json` e na devolutiva.

## 10. Decisões que dependem da etapa 4

1. **A exposição é a decisão central.** Na faixa `(0; 0,02]` foram observados 238,99 sinistros por 100 apólices-ano; o Poisson com offset previu 14,24 e o com coeficiente livre, 134,56. Decidir **o que será previsto**: o offset representa contagem proporcional à duração; o coeficiente livre descreve a contagem condicionada à duração observada no benchmark, cuja origem e disponibilidade antecipada precisam ser consideradas antes de lê-lo como previsão para um ano futuro. A melhora é relevante e deixa desvio expressivo, e o erro de taxa global sobe (0,054 → 0,084). Qualquer mudança de especificação é ajustada no treino, com previsões diretas e o teste reservado.
2. Escolha entre taxa global, Poisson e NB2, sensível ao degrau (condição de erro de taxa global) e à população (principal contra agrupada).
3. Tratar `BonusMalus` somente como sensibilidade, enquanto o instante de medição for indeterminado.
4. **Zeros.** Para a taxa global, o Poisson e o NB2 com offset fixo, os IC globais de zeros previstos − observados cruzam zero. No diagnóstico com `log(Exposure)` livre, o total é −160,2 (−283,1; −30,0), com déficits com suporte em `(0,02; 0,08]` e `(0,25; 0,50]`. Os dois resultados ficam separados, e a hipótese de um segundo processo segue como requisito à parte.

## 11. Limitações

Intervalos condicionados aos modelos ajustados, sem a variabilidade do ajuste. Perfis coincidentes representam atributos observados, sem identidade de apólice ou condutor. Pearson/gl é sensível a médias previstas muito pequenas. A comparação é interna ao benchmark.

## Errata (2026-11-23, auditoria)

- O registro de execução dos notebooks, citado na seção 9 como `outputs/etapa_03/notebook_execution.json`, foi movido para fora da pasta selada: `outputs/notebook_execution_etapa03.json` (regravado por `scripts/run_notebooks.py`).
- A linha de `notebook_execution.json` foi removida de `outputs/etapa_03/SHA256SUMS_etapa03.txt`. Hash da lista: antigo `25213ade1fff5bbb1cbad5ff42720482ef697506b8e60f537b69f481ba5eacf3`; novo `8e72eb300d9e3a1b92e86982d35db1389689c7126a04dd9810b639a0ab7ef20b`.
- Motivo: o executor de notebooks regravava o arquivo depois de a lista ter sido gerada, e a lista deixava de conferir.
- Sem efeito sobre os resultados nem sobre os hashes congelados (o arquivo não fazia parte deles).
