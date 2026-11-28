# Etapa 2. Protocolo de avaliação

`production_eligible = false`. Esta etapa fixa as regras antes de observar os resultados de validação. Nenhum modelo foi ajustado e nenhuma previsão foi feita. Os parâmetros vêm de `src/freq_sinistros/protocol.py`, estão registrados em `outputs/etapa_02/protocol_config.json` e não dependem de desfechos de validação ou de teste.

## 1. População e partição

A população reúne os 678.013 registros do ARFF OpenML 41214 v1 (SHA-256 `a45363e0…807cdd`, conferido contra o manifesto da etapa 1 antes de particionar), com `IDpol` único (678.013 valores) lido de forma canônica, inclusive o token `1e+05` = 100.000. Todas as observações válidas são preservadas (zeros, exposições curtas e acima de 1 ano, contagens altas e idades extremas), mantendo a diferença de 22 registros frente ao CASdatasets como limitação de origem indeterminada. A regra de partição em única alocação, sem estado nem semente, segue `bucket = int.from_bytes(SHA-256(utf8("freMTPL2freq-v1-etapa2|" + IDpol))[:8], "big") mod 10000`, destinando `[0, 7000)` ao treino, `[7000, 8500)` à validação e `[8500, 10000)` ao teste, regra sugerida na tarefa e adotada sem ajustes.

| Partição | IDs | Proporção |
| --- | ---: | ---: |
| Treino | 474.581 | 69,996% |
| Validação | 101.573 | 14,981% |
| Teste | 101.859 | 15,023% |

As verificações (`partition_config.json`) confirmam cobertura integral, 0 IDs em mais de uma partição, tamanhos somando 678.013 e o mesmo hash com linhas embaralhadas e em nova execução, com SHA-256 das atribuições ordenadas por `IDpol` (`IDpol,partition`, LF) igual a `f14f53509d63ff468f025fd43096a5c2d12ba3fdd48529e7644f003c73396ce6`; o arquivo de atribuições (`data/splits/partition_assignments.csv`, apenas `IDpol` e partição) é local, fica fora do versionamento, e `verify-partition` o recria do ARFF e compara. Do teste reservado utiliza-se aqui apenas a contagem de IDs (um teste automático altera `ClaimNb` e `Exposure` das linhas de teste e confirma que nenhuma saída muda), permanecendo sua avaliação final para uma única execução após o congelamento do modelo e do roteiro na etapa 4.

## 2. Treino: verificação (`overview.json`, `segment_support.csv`)

O treino reúne 474.581 registros, 251.070,30 anos de exposição, 25.319 sinistros e 450.731 zeros (94,97%), com taxa de 10,084 por 100 apólices-ano. Também preserva 8 registros com `ClaimNb > 4`, 869 com `Exposure > 1` e 13.465 com `Exposure ≤ 0,02`.

A validação tem 101.573 registros e 53.590,05 anos. Seus desfechos não são usados nesta verificação. As taxas observadas por segmento no treino estão em `segment_support.csv`: 250,58 na faixa `(0; 0,02]` e 7,07 em `= 1`, por 100 apólices-ano. Essa diferença será acompanhada na avaliação conforme a regra já definida de manter a exposição curta visível.

## 3. Perfis coincidentes e partição agrupada (somente desenvolvimento)

Chave de perfil: as oito covariáveis candidatas (`Area`, `VehPower`, `VehAge`, `DrivAge`, `VehBrand`, `VehGas`, `Density`, `Region`) em texto separado por `|`. Numéricos integrais viram inteiro, os demais usam `repr`, categorias usam o nível, e ausente vira `__MISSING__`. A chave representa **atributos observados**; a identidade de apólice ou condutor é indeterminada.

| Indicador (treino × validação) | Valor |
| --- | ---: |
| Registros de validação cujo perfil existe no treino | 40.126 (39,5%) |
| Exposição de validação nesses registros | 25,5% |
| Perfis distintos de validação presentes no treino | 36.673 de 95.031 (38,6%) |
| Registros de treino em perfis compartilhados dentro do treino | 189.628 (40,0%) |

A partição principal por `IDpol` mede, portanto, um cenário em que cerca de 40% dos registros de validação têm perfil já visto no treino. Para avaliar o desempenho em perfis que não aparecem no treino, foi predefinida uma **análise secundária, a executar na etapa 3**.

Nessa análise, o desenvolvimento (treino + validação, 576.154 registros) é dividido por chave: `bucket(SHA-256("freMTPL2freq-v1-etapa2-grupo|" + chave)[:8]) mod 10000 < 8235` vai para `group_train` (razão 70:15). O resultado contém 474.478 registros em `group_train` (82,35%), 101.676 em `group_validation`, 424.986 chaves distintas e **0 chaves em ambos** (`grouped_partition_summary.json`; atribuições em `data/splits/grouped_dev_assignments.csv`, hash `65ec330e…fd09f`). O teste não participa dessa partição.

## 4. Preparação e exposição

O alvo é `ClaimNb` (escala de contagem), `IDpol` fica restrito à rastreabilidade, e `Exposure` entra como `log(Exposure)` em offset de coeficiente fixo 1 (Poisson e NB2), sendo a duração dado do cenário de previsão e a avaliação interna ao benchmark; as oito covariáveis do modelo principal excluem `BonusMalus` (mantido apenas em sensibilidade).

Nos passos ajustados exclusivamente no treino de cada comparação, codificam-se `Area`, `VehBrand`, `VehGas`, `Region` e `VehPower` (tratada como categoria), agrupando em `OTHER` níveis com menos de 500 registros no treino; `VehAge` e `DrivAge` entram em faixas por quantis do treino, e `Density` entra em `log` padronizado com média e desvio do treino. Níveis futuros desconhecidos vão para `OTHER` (ou para o nível de referência se `OTHER` não existir), com contagem registrada; valores categóricos ausentes futuros recebem `__MISSING__` (mapeado como desconhecido) e numéricos recebem a mediana do treino, com contagem registrada, enquanto `ClaimNb`, `Exposure` ou média prevista ausentes ou inválidos **interrompem** a avaliação sem imputação ou descarte silencioso. A taxa global de referência é `Σ ClaimNb / Σ Exposure` do treino (`μ_i = taxa × Exposure_i`), com taxas agregadas multiplicadas por 100.

## 5. Comparações e amostra de treino

Os candidatos são taxa global, Poisson e NB2 (`Var = μ + αμ²`), avaliados sobre as mesmas linhas e covariáveis equivalentes, restringindo ZIP e ZINB à regra da seção 8; todas as comparações na validação tomam como referência principal as contagens e exposições originais.

| Sensibilidade predefinida | Descrição |
| --- | --- |
| Com `BonusMalus` | Oito covariáveis + `BonusMalus` |
| Limites do exemplo didático | Ajuste com `ClaimNb' = min(ClaimNb, 4)` e `Exposure' = min(Exposure, 1)` no treino, avaliado nas mesmas linhas de validação com contagens e exposições originais (previsão com a exposição original no offset); a auditoria mostrou que o efeito agregado é pequeno (taxa −0,09%) |
| Partição agrupada por perfil | Divisão do desenvolvimento descrita na seção 3 |
| Bootstrap por chave de perfil | Reamostragem em grupo descrita na seção 7 |
| Descritiva sem `(0; 0,02]` | Métricas gerais com e sem a faixa `(0; 0,02]`, **sem** efeito na escolha, mantendo a faixa curta visível no painel |

Para escalonamento amostral, a amostra inicial toma os `n` menores valores de `SHA-256("freMTPL2freq-v1-etapa2-amostra|" + IDpol)` entre os IDs de treino (independente da ordem e aninhada), nos degraus de 100.000, 200.000 e treino inteiro (474.581), com a mesma amostra aplicada a taxa global, Poisson e NB2 em cada comparação direta. Pela regra de ampliação, passa-se ao degrau seguinte se o ajuste atual convergir, levar até 600 s e usar até 8 GiB de pico (metade dos 16 GB) em CPU; caso contrário fixa-se o degrau atual, registrando sempre o tempo e o pico de RAM de cada ajuste.

## 6. Métricas

A métrica primária é o desvio médio de Poisson por registro, `D = (2/n) Σ [ y·log(y/μ) − (y − μ) ]`, com `y·log(y/μ) = 0` quando `y = 0` (contribuição `2μ`), exigindo `μ` finita e `> 0` (valor inválido interrompe a avaliação) sobre a mesma população para todos os candidatos. A calibração mede totais previstos e observados e a diferença **prevista − observada** entre taxas por 100 apólices-ano (`100 · (Σμ − Σy) / Σ Exposure`, usando a exposição total do grupo), global e por segmento. Para distribuição, calculam-se a log-verossimilhança preditiva média por registro (Poisson e NB2 com `lgamma`) e os zeros previstos `Σ P(Y_i = 0)` contra zeros observados, global e por faixa de exposição, entrando a taxa global como Poisson com `μ_i = taxa · Exposure_i`.

## 7. Segmentos e suporte (`segment_support.csv`, `segment_reconciliation.csv`)

As faixas de exposição seguem a auditoria: `(0; 0,02]`, `(0,02; 0,08]`, `(0,08; 0,25]`, `(0,25; 0,50]`, `(0,50; 0,75]`, `(0,75; 1)`, `= 1` e `> 1`, mais a linha de exposição inválida. A idade do veículo usa `0–1`, `2–5`, `6–10`, `11–20` e `21+` com limite inferior inclusivo e superior exclusivo (`[0,2)`, `[2,6)`, `[6,11)`, `[11,21)`, `[21,∞)`), indo ausente, não finita ou negativa para `idade_ausente_ou_invalida`; a região usa o código cadastral (22 regiões), com ausente ou desconhecido em `__MISSING__`.

Pelo escopo corrigido na etapa 3, cada segmentação reconcilia registros e exposição com o total da partição em **treino e validação**, e sinistros **apenas no treino** (`segment_reconciliation.csv`), sem o teste (sinistros da validação passam a ser reconciliados na etapa 3). Mantém-se como proposto o critério de baixo suporte (`< 500` registros **ou** `< 25` apólices-ano): nenhum segmento populado é sinalizado no treino, e na validação ficam sinalizados `> 1` (184 registros), `R21`, `R42` e `R43`, além dos segmentos sem registros; os limites são mantidos porque a sinalização ocorre na partição avaliada, os segmentos continuam visíveis e ficam fixos antes da etapa 3 (como observação, a faixa `(0; 0,02]` soma 32,2 apólices-ano na validação, pouco acima do limite de 25).

## 8. Incerteza e regra de decisão

O bootstrap percentil de 95% mantém os modelos fixos e gera 1.000 réplicas com semente 20261121 por reamostragem de registros (`IDpol`) com reposição e **mesmos índices** para todos os modelos (diferenças pareadas), fornecendo intervalos para desvio médio e diferenças pareadas, diferença prevista − observada de taxas global e por segmento sem baixo suporte, log-verossimilhança média e diferenças pareadas, e zeros previstos − observados (global e por faixa de exposição). Réplicas com denominador zero deixam o indicador indefinido nessa réplica e são excluídas e contadas (se superarem 1% das réplicas, o intervalo não é informado), e segmentos de baixo suporte são exibidos sem intervalo; como sensibilidade, usa-se o bootstrap por chave de perfil (reamostrando grupos inteiros, porque 40% dos registros de treino compartilham perfil e o bootstrap por `IDpol` pode subestimar a incerteza), sendo reportadas as duas versões.

Na ordem de simplicidade (taxa global < Poisson < NB2 < ZIP/ZINB), aplicam-se cinco critérios de decisão:

| Regra | Critério |
| --- | --- |
| 1 | Um candidato mais complexo só vence se o intervalo pareado da melhora excluir zero a seu favor: no desvio médio para Poisson contra a taxa global, e na log-verossimilhança preditiva média para NB2 contra Poisson (porque as médias quase coincidem). |
| 2 | O ajuste deve convergir sem erro, a calibração global (diferença de taxas) não pode piorar, e o sentido do ganho não pode se reverter ao ampliar a amostra de treino ao degrau seguinte (estabilidade). |
| 3 | Diante de evidência inconclusiva, permanece o modelo mais simples. |
| 4 | ZIP ou ZINB só entram com **desajuste persistente dos zeros** em validação (zeros previstos − observados com intervalo excluindo zero no total e em ao menos duas faixas de exposição sem baixo suporte, após o melhor entre Poisson e NB2) **e** com hipótese plausível para o segundo processo escrita antes do ajuste. |
| 5 | A escolha é congelada com modelo, covariáveis, amostra e roteiro antes de abrir o teste; o teste é avaliado uma única vez e os resultados valem dentro do benchmark. |

## 9. Base metodológica

O protocolo adapta três referências metodológicas do MBA: o notebook 08 orienta o contrato de contagem, dispersão e zeros; o 08.01, a comunicação da incidência anual; e o 08.02, a comparação entre distribuições observadas e previstas. As adaptações tratam a exposição variável e a reserva das partições. Até esta etapa, as funções de métricas e bootstrap (`src/freq_sinistros/metrics.py`) foram verificadas apenas com dados sintéticos.

## 10. Reprodução e custo

```powershell
python -I -m freq_sinistros --root . partition          # materializa e verifica; sem rede
python -I -m freq_sinistros --root . verify-partition   # recria e compara o hash
python -m pytest ; ruff check .
```

Execução real: 22,6 s de parede, pico de 485 MiB (ambiente conda `freq-sinistros`, CPU). Duas execuções geraram saídas idênticas (`SHA256SUMS_etapa02.txt`; só `cost.json` varia).

## 11. Emenda da etapa 3 (registrada antes do primeiro ajuste em validação)

Fonte legível por máquina: `outputs/etapa_03/protocol_amendment.json`. A seção 7 foi corrigida acima quanto ao escopo das reconciliações.

**Decisões do parecer da etapa 2**

| Tema | Regra |
| --- | --- |
| Amostra e custo | Degraus aninhados 100 mil, 200 mil e treino integral, mesmas linhas para os três candidatos. Limites 600 s por ajuste e 8 GiB de pico. Amplia-se se os três convergirem e couberem; se um degrau falhar, usa-se o último degrau comum elegível; se o primeiro falhar, devolve-se o diagnóstico. |
| Baixo suporte | `< 500` registros ou `< 25` apólices-ano; segmentos visíveis; intervalos só para segmentos sem baixo suporte. |
| Bootstrap | 1.000 réplicas pareadas por `IDpol` (principal) e por perfil (sensibilidade); intervalos condicionados aos modelos ajustados; perfis coincidentes são atributos observados. |
| Poisson contra a taxa global | IC 95% pareado da melhora no desvio médio exclui zero **e** o erro absoluto da diferença de taxa global do Poisson não supera o da taxa global (estimativas pontuais). |
| NB2 contra Poisson | IC 95% pareado da melhora na log-verossimilhança média exclui zero **e** desvio médio e erro absoluto da taxa global do NB2 são iguais ou menores (estimativas pontuais). Se a taxa global seguir como referência, o NB2 precisa de ganho pareado no desvio também contra ela. |
| Estabilidade | Convergência exigida. Ganho com IC excluindo zero no maior degrau viável e estimativa pontual com o mesmo sinal em todos os degraus viáveis. Com um único degrau, estabilidade por tamanho "não verificada" e comparação provisória. |
| Inflação de zeros (etapa 4) | Só se o modelo de contagem de referência **subestimar** zeros: IC 95% de `zeros previstos − observados` inteiramente abaixo de zero no total e em pelo menos duas faixas de exposição com suporte, e hipótese plausível documentada. |

**Verificações derivadas da EDA (treino, `eda_summary.json`)**

| Verificação | Achado |
| --- | --- |
| 1. Exposição curta em perfis | Em estratos de faixa de idade do condutor × faixa de idade do veículo (referência com taxa das linhas de `Exposure ≥ 0,75`), a razão entre a taxa das linhas com `Exposure ≤ 0,08` e a de referência superou 1 em 24 de 24 estratos com suporte nos dois grupos (mediana 4,2; mínimo 2,7). A razão observado/esperado de sinistros nas faixas curtas foi 30,9 (`(0; 0,02]`), 4,8 (`(0,02; 0,08]`), 2,9, 1,6, 1,3 e 1,1, e 0,96 em `= 1`; o padrão persiste dentro dos estratos de idade com suporte em comparação descritiva que controla apenas a idade do condutor e a do veículo e não exclui outras explicações. |
| 2. Pressuposto do offset | `E[ClaimNb]` proporcional à exposição (coeficiente 1 em `log(Exposure)`). No diagnóstico do treino (100 mil linhas, oito covariáveis), o coeficiente livre de `log(Exposure)` foi 0,39 (erro padrão ingênuo 0,016), longe de 1, indicando que os três candidatos subestimarão as faixas curtas e superestimarão as longas se o padrão se mantiver na validação. |
| 3. Conferência na validação | Calibração por faixa de exposição (prevista − observada, por 100 apólices-ano, com IC) para os três candidatos sem recalibração pós-hoc nem fatores por segmento; o Poisson com `log(Exposure)` livre entra apenas como **diagnóstico exploratório, não candidato** (seu uso exigiria nova decisão do Codex na etapa 4), mantendo inalteradas a métrica primária e as regras de escolha. |
| 4. Casos extremos | Oito registros de treino com `ClaimNb > 4` (71 sinistros), dos quais cinco compartilham `DrivAge = 52` e `VehAge = 12` (44 sinistros), indício de concentração em poucos perfis com a identidade indeterminada; as caudas válidas são preservadas e têm o bootstrap por perfil como sensibilidade correspondente. |
| 5. Cobertura de categorias | Na amostra de 100 mil linhas, `Region` R21, R42 e R43 e `VehPower` 13, 14 e 15 ficam abaixo de 500 linhas e vão para `OTHER`. Em 374.581 linhas de treino fora da amostra, 3.618 (`Region`) e 4.700 (`VehPower`) linhas com nível agrupado ou ausente do ajuste foram mapeadas para `OTHER`, mantendo finita a matriz de desenho. |

**Definições operacionais.** População de avaliação: as 101.573 linhas de validação, com contagens e exposições originais. Diferença de taxa: `100 · (Σμ − Σ ClaimNb) / Σ Exposure`. Sensibilidades no maior degrau viável: `BonusMalus` em faixas por quantis; limites didáticos só no ajuste; bootstrap por perfil; partição agrupada (pré-processamento, ajustes e avaliação refeitos em `group_train` e `group_validation`, com resultados à parte); descrição da métrica sem a faixa `(0; 0,02]`; `log(Exposure)` livre (diagnóstico).

**Divulgação.** Antes desta emenda foi executado um teste de fumaça do estimador em 100 mil linhas de treino (tempo e equivalência com o `statsmodels`), sem uso de validação.
