# Etapa 1. Aquisição e auditoria do `freMTPL2freq`

`production_eligible = false`. Esta etapa estabelece a procedência e os limites da base antes de qualquer escolha de modelagem. Nenhuma correção, exclusão, deduplicação, imputação ou partição foi aplicada. O arquivo bruto foi preservado byte a byte.

## 1. Procedência e reprodução

| Item | Valor |
| --- | --- |
| Dataset / versão / ID / file_id | `freMTPL2freq` / 1 / 41214 / 20649148 |
| Licença catalogada | CC0 |
| Representação | ARFF, a rota `url` do metadado OpenML (`parquet_url` existe e não foi usada) |
| Metadados | <https://www.openml.org/api/v1/json/data/41214> (+ `/features/` e `/qualities/`), cópia em `outputs/etapa_01/openml_metadata.json` |
| Download | `https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff` (o declarado, `openml.org/...`, redireciona para o `www`) |
| Adquirido em | 2026-11-21 14:03 UTC |
| **Arquivo bruto local** | `D:\Projetos\Frequência_de_Sinistros_por_Apólice\data\raw\freMTPL2freq_v1.arff` (fora do versionamento, ver `.gitignore`) |
| Tamanho | 36.288.567 bytes |
| SHA-256 (bytes baixados) | `a45363e056e2ea56408b38eeb9d4d04d7f6c6982eb7a14ed5e807c7c71807cdd` |
| MD5 (bytes baixados) | `f8875568bf0ca622929105197e2db613` |
| MD5 declarado pelo OpenML | `f8875568bf0ca622929105197e2db613` → **confere** (o OpenML só publica MD5, referente ao ARFF; o SHA-256 é local) |

O manifesto está em `outputs/etapa_01/manifest.json`. Antes de ler os dados, a auditoria confere tamanho e SHA-256 contra esse registro. Qualquer divergência interrompe a execução com código 2, para que um arquivo diferente não seja auditado como se fosse o original.

**Comandos** (raiz do projeto, PowerShell; ambiente conda `freq-sinistros`, Python 3.12.16, numpy 2.5.5, pandas 3.1.1, psutil 7.2.2; `environment.yml`):

```powershell
$env:CONDA_SSL_VERIFY = "truststore"   # preparação: necessária nesta máquina (SSL do conda)
conda env create -f environment.yml
conda activate freq-sinistros
python -I -m freq_sinistros --root . acquire   # rede; não sobrescreve o bruto existente
python -I -m freq_sinistros --root . audit     # sem rede; regenera outputs/etapa_01/*
python -m pytest                                # sintético; sem dados, rede ou GUI
ruff check .
```

A aquisição desta entrega rodou antes da criação do ambiente conda, com Python 3.15.1 e só biblioteca padrão (venv provisória, removida). Isso está registrado no manifesto. O código de aquisição é o mesmo.

## 2. Contagens entre fontes

| Fonte | Contagem | Diferença p/ linhas observadas |
| --- | ---: | ---: |
| Arquivo baixado, linhas de dados | **678.013** | 0 |
| Arquivo baixado, `IDpol` distintos | **678.013** | 0 |
| OpenML `NumberOfInstances` / docstring do scikit-learn | 678.013 | 0 |
| Descrição OpenML / CASdatasets (descrição e `dim()`) | 677.991 | −22 |

O arquivo coincide com o OpenML e com o scikit-learn. Já a descrição e o CASdatasets informam 22 linhas a menos, diferença cuja origem **não pôde ser determinada**. IDs repetidos, linhas duplicadas ou rejeições na leitura não explicam a discrepância, pois as 678.013 linhas correspondem a 678.013 IDs distintos. Como a fonte original em R do CASdatasets não foi baixada, os IDs não puderam ser comparados diretamente.

Por isso, os 678.013 registros do artefato OpenML são adotados como população de trabalho, e a diferença permanece registrada como limitação. Foram consultados a API do OpenML, a página `freMTPL.html` do CASdatasets e o exemplo do scikit-learn.

## 3. Esquema e integridade

12 colunas, 678.013 linhas, 0 linhas rejeitadas, 0 faltantes, 0 não numéricos, 0 valores fora do domínio nominal (`schema_summary.csv`, `categorical_levels.csv`, `numeric_quantiles.csv`).

Divergências e achados registrados na auditoria mostram que o OpenML lista em duplicata os níveis nominais de `Area`, `VehBrand` e `Region`, embora os conjuntos únicos coincidam com o ARFF, e `VehGas` vem como `string` no ARFF e no OpenML e é lida como categórica (2 níveis). `VehBrand` apresenta 11 níveis (B1–B6, B10–B14), descritos no dicionário CASdatasets como "unknown categories", enquanto `BonusMalus` varia de 50 a 230 (contra 50–350 no dicionário), com 384.156 linhas (56,7%) no piso 50. Os domínios extremos descritivos (não corrigidos) atingem `VehAge` até 100 (25 linhas em 100 e 1.116 acima de 30), `DrivAge` de 18 a 100 (3 linhas em 100) e `Density` até 27.000, sem colunas de data nem identificador de unidade superior (cliente, condutor ou contrato).

## 4. Identificador e unidade

Os 678.013 IDs presentes são distintos, com 0 IDs repetidos, 0 linhas duplicadas e 0 grupos conflitantes (`id_repeats.csv` contém apenas o cabeçalho). Um ID está gravado em notação científica (`1e+05`, linha 49.345 do arquivo); a leitura como texto seguida de conversão `float64` para `int64` recupera exatamente 100.000, sem perda nem colisão, artefato de serialização registrado em `read_noncanonical_id_tokens` com identificador tipado como `int64`.

Entre os 9 campos de covariáveis (todos exceto `IDpol`, `ClaimNb` e `Exposure`), observam-se 528.765 conjuntos distintos de valores; 108.663 conjuntos aparecem em mais de uma linha, num total de 257.911 linhas (38,0%). O maior grupo tem 22 linhas, e em 18.703 dessas linhas em grupos coincidentes a soma de exposição do grupo passa de 1 ano. Como a fonte omite identificador de unidade superior e datas, esse padrão descreve apenas coincidência de atributos observados, mantendo indeterminada a identidade de apólice, contrato ou condutor. Cada linha, identificada por `IDpol`, é um registro com sua exposição e é a unidade de rastreabilidade (embora a descrição do OpenML fale em "policies", o arquivo não permite afirmar que `IDpol` corresponde a uma apólice completa), ficando o desenho da partição para a etapa 2, que poderá usar o agrupamento por conjunto de atributos para testar a generalização a perfis distintos.

## 5. `ClaimNb` (alvo)

Faltantes, não numéricos, não finitos, negativos e fracionários: 0 em todos. Todas as 678.013 linhas são válidas (denominador).

| Valor | Linhas | % das linhas | Sinistros | % dos sinistros |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 643.953 | 94,976% | 0 | 0 |
| 1 | 32.178 | 4,746% | 32.178 | 89,13% |
| 2 | 1.784 | 0,263% | 3.568 | 9,88% |
| 3 | 82 | 0,012% | 246 | 0,68% |
| 4 | 7 | 0,001% | 28 | 0,08% |
| >4 (5×2, 6, 8, 9, 11×3, 16) | 9 | 0,001% | 82 | 0,23% |

Total 36.102 sinistros. Média = total / linhas válidas = 0,053247. Variância 0,057656 (amostral, ddof=1; ddof=0: 0,057656). Razão variância/média 1,083, descritiva (não é diagnóstico de superdispersão condicional). Máximo 16. Zeros: 643.953 (94,98%); positivos: 34.060 (5,02%).

## 6. `Exposure`

Unidade: anos (dicionário). Nulos, não numéricos, não finitos, negativos e zero: 0. Todas as linhas são válidas. Mínimo positivo 0,002732 (1/366). Percentis: p1 0,0082, p5 0,04, p25 0,18, p50 0,49, p75 0,99, p95 1, p99 1, p99,9 1,03. Máximo 2,01. Soma da exposição positiva finita: **358.499,45 anos**.

Limites descritivos declarados (não são regras de exclusão):

| Limite | Linhas | Exposição | Sinistros | Zeros |
| --- | ---: | ---: | ---: | ---: |
| Exposure ≤ 0,02 (~7 dias) | 19.259 | 214,37 | 535 | 18.749 |
| Exposure ≤ 0,08 (~1 mês) | 114.793 | 6.317,44 | 2.815 | 112.130 |
| Exposure > 1 | 1.224 | 1.363,34 | 54 | 1.173 |
| Exposure = 1 | 168.125 | 168.125 | 11.776 | 156.914 |

## 7. Zeros, caudas e totais (`exposure_bands.csv`, `exposure_x_claimnb.csv`)

Faixas não sobrepostas. Taxa = 100 × sinistros / exposição. O subconjunto de taxa (ClaimNb e Exposure válidos) tem as 678.013 linhas, então nenhuma fica fora do denominador. A linha `exposicao_invalida` existe e está vazia.

| Faixa | Linhas | Exposição | Sinistros | Zeros | % zeros | Taxa /100 ap.-ano |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0; 0,02] | 19.259 | 214,37 | 535 | 18.749 | 97,35 | 249,57 |
| (0,02; 0,08] | 95.534 | 6.103,07 | 2.280 | 93.381 | 97,75 | 37,36 |
| (0,08; 0,25] | 108.019 | 18.647,33 | 4.316 | 103.925 | 96,21 | 23,15 |
| (0,25; 0,50] | 131.302 | 51.388,74 | 6.481 | 125.231 | 95,38 | 12,61 |
| (0,50; 0,75] | 92.494 | 58.574,59 | 5.968 | 86.892 | 93,94 | 10,19 |
| (0,75; 1) | 62.056 | 54.083,01 | 4.692 | 57.688 | 92,96 | 8,68 |
| = 1 | 168.125 | 168.125,00 | 11.776 | 156.914 | 93,33 | 7,00 |
| > 1 | 1.224 | 1.363,34 | 54 | 1.173 | 95,83 | 3,96 |
| inválida | 0 | 0 | 0 | 0 | n/d | n/d |
| **Total** | **678.013** | **358.499,45** | **36.102** | **643.953** | **94,98** | **10,07** |

`reconciliation.csv`: 8 verificações (linhas, sinistros, exposição, subconjunto) entre o bruto e a tabela de faixas, todas `ok`.

A taxa cai de 249,6 para 7,0 por 100 ap.-ano da faixa mais curta à de ano completo; trata-se de leitura descritiva, não causal, pois a interpretação do offset depende da relação entre a duração observada e a contagem, e a faixa curta deve ficar explícita na avaliação de calibração. Na cauda de `ClaimNb`, das 9 linhas com `ClaimNb > 4`, 4 estão em (0,02; 0,08] (36 sinistros em 0,31 ano), 3 em (0,25; 0,5] e 2 em `= 1`. Já em `Exposure > 1` há 1.224 linhas (máximo 2,01), tratadas como cauda observada porque a fonte descreve as apólices como observadas "mostly on one year" (predominantemente por um ano), o que não exclui períodos maiores, mantendo preservados os valores válidos.

**Sensibilidade agregada dos limites do scikit-learn** (`clip_sensitivity.csv`). Regra: `ClaimNb' = min(ClaimNb, 4)` e `Exposure' = min(Exposure, 1)`, sem remover linhas. Totais ajustados = brutos menos as excedências. Escopo: as 678.013 linhas.

| Cenário | Linhas afetadas | Sinistros | Exposição | Taxa /100 | Δ sinistros | Δ exposição | Δ taxa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bruto | 0 | 36.102 | 358.499,45 | 10,0703 | | | |
| ClaimNb ≤ 4 | 9 | 36.056 | 358.499,45 | 10,0575 | −46 (−0,127%) | 0 | −0,127% |
| Exposure ≤ 1 | 1.224 | 36.102 | 358.360,11 | 10,0742 | 0 | −139,34 (−0,039%) | +0,039% |
| Ambos | 1.233 | 36.056 | 358.360,11 | 10,0614 | −0,127% | −0,039% | −0,089% |

No agregado, os limites alteram pouco a taxa. Esta etapa, porém, não mediu o efeito por segmento; em especial, o resultado não permite concluir que as faixas de exposição curta sejam igualmente pouco afetadas.

## 8. Disponibilidade das covariáveis

Detalhe, papel e unidade em `covariate_contract.csv` (PT-BR e EN-US). Fonte: dicionário CASdatasets e metadados OpenML. `ClaimNb` é desfecho e `Exposure` é dado do cenário. Para `Area`, `VehPower`, `VehAge`, `DrivAge`, `VehBrand`, `VehGas`, `Density` e `Region`, a fonte descreve o atributo e **omite a data de referência**. Por isso, a disponibilidade na data de estimação da taxa é uma **hipótese**, registrada assim em `covariate_contract.csv` (PT-BR e EN-US).

**`BonusMalus`: disponibilidade indeterminada.** O dicionário diz apenas "between 50 and 350: <100 means bonus, >100 means malus". Nenhuma fonte consultada informa quando o índice foi medido em relação ao período dos sinistros. O esquema não tem datas. Pelo plano de supervisão, o modelo principal deve ficar sem `BonusMalus`, com sensibilidade com e sem a variável. A relação com o alvo não foi explorada na base inteira, porque o plano reserva isso ao treino, depois da partição.

## 9. Custo local (CPU, Windows 11, 16 threads, 15,7 GiB de RAM)

| Estágio | Tempo de parede | Pico de memória acumulado |
| --- | ---: | ---: |
| Download (36,3 MB) | 114,2 s | n/d (não medido) |
| Verificação SHA-256 | 0,07 s | 76 MiB |
| Leitura ARFF em blocos de 100 mil linhas | 3,12 s | 196 MiB |
| Cálculo da auditoria | 2,27 s | 301 MiB |
| Escrita | 0,04 s | 301 MiB |
| **Total da auditoria** | **5,5 s** | **≈301 MiB** (`peak_wset`, psutil) |

A base inteira foi processada com folga nos 16 GB disponíveis. Em duas execuções consecutivas, as saídas CSV e JSON foram idênticas bit a bit (`SHA256SUMS_audit.txt`). Apenas `cost.json`, que registra medidas de execução, variou.

## 10. Decisões encaminhadas pelo parecer do Codex

| Tema | Encaminhamento |
| --- | --- |
| População | Adotar os 678.013 registros do OpenML v1, mantendo a diferença de 22 frente ao CASdatasets como limitação de origem indeterminada. |
| Unidade | `IDpol` identifica cada registro, e o desenho da partição será fechado após a aprovação. |
| Caudas | Preservar observações válidas na especificação principal (exposições curtas e acima de 1 ano, contagens altas e idades extremas), mantendo a faixa curta explícita na calibração e com qualquer sensibilidade predefinida após a partição. |
| `BonusMalus` | Manter fora da especificação principal, com análise de sensibilidade com e sem a variável, dado que o instante de medição segue indeterminado. |

## 11. Limitações

Notebooks PT-BR e EN-US e a validação de paridade dos notebooks não fazem parte desta entrega; o CI (`.github/workflows/ci.yml`) cobre Ruff e pytest, e não foi executado (o projeto não era repositório git nesta etapa). A aquisição foi feita com o Python 3.15.1 da venv provisória, e o comando de aquisição não foi reexecutado no ambiente conda porque não sobrescreve o bruto existente.

## 12. Correções após o parecer do Codex

| Item | Mudança | Evidência |
| --- | --- | --- |
| Denominador zero | `safe_ratio` em `audit.py`: taxa zero quando há exposição e nenhum sinistro; taxas e variações percentuais com denominador zero ou ausente viram valor ausente (`NaN`; `null` no JSON; vazio no CSV). Vale para `clip_sensitivity`, `band_table` e o resumo. | `test_clip_sensitivity_all_zero_claims_keeps_zero_rate`, `test_clip_sensitivity_without_valid_rows_is_missing_not_error`, `test_safe_ratio`, `test_audit_terminates_when_all_claims_are_zero`, `test_audit_terminates_without_valid_rate_rows` (a auditoria termina e reconcilia nos dois casos) |
| Checksum do download | `download_file` só move o arquivo ao caminho final se o MD5 declarado conferir; senão apaga o temporário e levanta `ChecksumMismatchError`. `acquire` exige o MD5 declarado, não grava manifesto em caso de falha e recusa sobrescrever um bruto existente antes de consultar a rede. A CLI devolve código 3. | `test_acquire_rejects_md5_mismatch_then_accepts_retry` (falha sem arquivo, sem `.part`, sem manifesto, código 3; nova tentativa na mesma pasta é aceita), `test_acquire_preserves_existing_verified_raw`, `test_acquire_requires_declared_md5` |
| Contrato e redação | Cauda de `ClaimNb` corrigida (`5×2`); `Exposure > 1` tratada como cauda observada; perfis coincidentes tratados como grupos de atributos com identidade indeterminada (relatório e docstring de `profile_groups`); disponibilidade das covariáveis cadastrais registrada como hipótese. | `test_covariate_availability_is_a_hypothesis_when_date_is_omitted` |

Nova execução da auditoria no mesmo ARFF: SHA-256 do bruto inalterado (`a45363e0…807cdd`) e conferido contra o manifesto. Reconciliação: 678.013 linhas, 36.102 sinistros, 643.953 zeros e 358.499,445463 anos, todas as 8 verificações `ok`. Das 12 saídas, **só `covariate_contract.csv` mudou** (novo texto de disponibilidade); as demais têm hash idêntico ao da entrega anterior. `cost.json` varia por natureza (5,5 s; pico de 301 MiB). Ruff: sem erros. Pytest: 24 testes, 93% de cobertura. O comando de aquisição não foi reexecutado contra a rede, porque não sobrescreve o bruto verificado; o novo caminho de erro foi demonstrado nos testes sintéticos.
