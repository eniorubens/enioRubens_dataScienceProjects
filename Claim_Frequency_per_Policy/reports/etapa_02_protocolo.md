# Etapa 2 — Protocolo de avaliação

`production_eligible = false`. Esta etapa fixa as regras antes de observar os resultados de validação. Nenhum modelo foi ajustado e nenhuma previsão foi feita. Os parâmetros vêm de `src/freq_sinistros/protocol.py`, estão registrados em `outputs/etapa_02/protocol_config.json` e não dependem de desfechos de validação ou de teste.

## 1. População e partição

- **População:** os 678.013 registros do ARFF OpenML 41214 v1 (SHA-256 `a45363e0…807cdd`, conferido contra o manifesto da etapa 1 antes de particionar). `IDpol` é único (678.013 valores) e é lido de forma canônica, inclusive o token `1e+05` = 100.000. Todas as observações válidas são preservadas: zeros, exposições curtas e acima de 1 ano, contagens altas e idades extremas. A diferença de 22 registros frente ao CASdatasets permanece como limitação de origem indeterminada.
- **Regra (única alocação, sem estado nem semente):** `bucket = int.from_bytes(SHA-256(utf8("freMTPL2freq-v1-etapa2|" + IDpol))[:8], "big") mod 10000`. Treino `[0, 7000)`, validação `[7000, 8500)`, teste `[8500, 10000)`. É a regra sugerida na tarefa, adotada sem ajuste.
- **Resultado:**

| Partição | IDs | Proporção |
| --- | ---: | ---: |
| Treino | 474.581 | 69,996% |
| Validação | 101.573 | 14,981% |
| Teste | 101.859 | 15,023% |

- **Verificações** (`partition_config.json`): cobertura integral, 0 IDs em mais de uma partição, tamanhos somam 678.013, mesmo hash com as linhas embaralhadas e em nova execução. Hash SHA-256 das atribuições ordenadas por `IDpol` (`IDpol,partition`, LF): `f14f53509d63ff468f025fd43096a5c2d12ba3fdd48529e7644f003c73396ce6`. O arquivo de atribuições (apenas `IDpol` e partição) está em `data/splits/partition_assignments.csv`, local e fora do versionamento. `verify-partition` o recria do ARFF e compara.
- **Teste reservado.** Esta etapa usa do teste apenas a contagem de IDs. Um teste automático altera `ClaimNb` e `Exposure` das linhas de teste e confirma que nenhuma saída muda. O teste final é avaliado uma única vez, após o congelamento do modelo e do roteiro (etapa 4).

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

- **Colunas:** alvo `ClaimNb` (escala de contagem); `IDpol` só para rastreabilidade; `Exposure` como `log(Exposure)` em offset de coeficiente fixo 1 (Poisson e NB2). A duração é dado do cenário de previsão, e a avaliação é interna ao benchmark. Covariáveis do modelo principal: as oito acima. `BonusMalus` fica fora (sensibilidade).
- **Passos (ajustados só no treino de cada comparação):** níveis de `Area`, `VehBrand`, `VehGas`, `Region` e `VehPower` (tratada como categoria); níveis com menos de 500 registros no treino são agrupados em `OTHER`; `VehAge` e `DrivAge` em faixas por quantis do treino; `Density` em `log` padronizado com média e desvio do treino.
- **Categorias desconhecidas:** nível fora do treino vai para `OTHER` (ou, se `OTHER` não existir, para o nível de referência), com contagem registrada.
- **Faltantes futuros:** categórico → `__MISSING__` (mapeado como desconhecido); numérico → mediana do treino, com contagem registrada. `ClaimNb`, `Exposure` ou média prevista ausentes ou inválidos **interrompem** a avaliação; nada é imputado nem descartado em silêncio.
- **Referência:** taxa global `Σ ClaimNb / Σ Exposure` do treino; `μ_i = taxa × Exposure_i`; taxas agregadas × 100.

## 5. Comparações e amostra de treino

- **Candidatos:** taxa global, Poisson e NB2 (`Var = μ + αμ²`), com as mesmas linhas e covariáveis equivalentes. ZIP e ZINB só pela regra da seção 8.
- **Valores originais como referência principal**, na validação, com contagens e exposições originais em todas as comparações.
- **Sensibilidades predefinidas:**
  1. Com `BonusMalus` (oito covariáveis + `BonusMalus`).
  2. Limites do exemplo didático: ajuste com `ClaimNb' = min(ClaimNb, 4)` e `Exposure' = min(Exposure, 1)` no treino; avaliação nas mesmas linhas de validação com contagens e exposições originais (previsão com a exposição original no offset). A auditoria mostrou que o efeito agregado é pequeno (taxa −0,09%).
  3. Partição agrupada por perfil (seção 3).
  4. Bootstrap por chave de perfil (seção 7).
  5. Descritiva: métricas gerais com e sem a faixa `(0; 0,02]`, **sem** efeito na escolha. A faixa curta permanece visível no painel.
- **Amostra inicial:** os `n` menores valores de `SHA-256("freMTPL2freq-v1-etapa2-amostra|" + IDpol)` entre os IDs de treino, independente da ordem e aninhada. Degraus: 100.000, 200.000 e treino inteiro (474.581). A mesma amostra vale para taxa global, Poisson e NB2 em cada comparação direta. **Regra de ampliação:** passar ao degrau seguinte se o ajuste atual convergir, levar até 600 s e usar até 8 GiB de pico (metade dos 16 GB), em CPU. Caso contrário, fixar o degrau atual. O tempo e o pico de RAM de cada ajuste são registrados.

## 6. Métricas

- **Primária:** desvio médio de Poisson por registro, `D = (2/n) Σ [ y·log(y/μ) − (y − μ) ]`, com `y·log(y/μ) = 0` quando `y = 0` (a linha contribui `2μ`). Exige `μ` finita e `> 0`; valor inválido interrompe a avaliação. Mesma população de avaliação para todos os candidatos.
- **Calibração:** totais previstos e observados e diferença **prevista − observada** entre taxas por 100 apólices-ano, `100 · (Σμ − Σy) / Σ Exposure` com a exposição total do grupo, global e por segmento.
- **Distribuição (modelos com distribuição):** log-verossimilhança preditiva média por registro (Poisson e NB2 com `lgamma`) e zeros previstos `Σ P(Y_i = 0)` contra zeros observados, global e por faixa de exposição. A taxa global entra como Poisson com `μ_i = taxa · Exposure_i`.

## 7. Segmentos e suporte (`segment_support.csv`, `segment_reconciliation.csv`)

- **Faixa de exposição** (da auditoria): `(0; 0,02]`, `(0,02; 0,08]`, `(0,08; 0,25]`, `(0,25; 0,50]`, `(0,50; 0,75]`, `(0,75; 1)`, `= 1`, `> 1`, mais a linha de exposição inválida.
- **Idade do veículo:** 0–1, 2–5, 6–10, 11–20 e 21+, com limite inferior inclusivo e superior exclusivo (`[0,2)`, `[2,6)`, `[6,11)`, `[11,21)`, `[21,∞)`); ausente, não finita ou negativa vai para `idade_ausente_ou_invalida`.
- **Região:** código cadastral (22 regiões); ausente ou desconhecido vai para `__MISSING__`.
- **Reconciliação (escopo corrigido na etapa 3):** cada segmentação reconcilia registros e exposição com o total da partição em **treino e validação**, e sinistros **apenas no treino** (`segment_reconciliation.csv`); o teste não participa. Sinistros da validação só passam a ser reconciliados na etapa 3.
- **Baixo suporte:** `< 500` registros **ou** `< 25` apólices-ano, mantido como proposto. No treino nenhum segmento populado é sinalizado. Na validação ficam sinalizados `> 1` (184 registros), `R21`, `R42` e `R43`, além dos segmentos sem registros. Os limites são mantidos porque a sinalização ocorre na partição avaliada e os segmentos continuam visíveis. Ficam fixos antes da etapa 3. Observação: a faixa `(0; 0,02]` tem 32,2 apólices-ano na validação, pouco acima do limite de 25.

## 8. Incerteza e regra de decisão

- **Bootstrap percentil de 95%:** modelos fixos, 1.000 réplicas, semente 20261121, reamostragem de registros (`IDpol`) com reposição, **mesmos índices** para todos os modelos (diferenças pareadas). Indicadores com intervalo: desvio médio de cada candidato e diferenças pareadas, diferença prevista − observada de taxas global e por segmento sem baixo suporte, log-verossimilhança média e diferenças pareadas, e zeros previstos − observados (global e por faixa de exposição).
- **Réplicas com denominador zero:** o indicador fica indefinido nessa réplica, que é excluída dele e contada. Se passar de 1% das réplicas, o intervalo não é informado. Segmentos de baixo suporte são mostrados sem intervalo.
- **Sensibilidade:** bootstrap por chave de perfil (reamostra grupos inteiros). Como 40% dos registros de treino compartilham perfil, o bootstrap por `IDpol` pode subestimar a incerteza. As duas versões são reportadas.
- **Regra de decisão** (ordem de simplicidade: taxa global < Poisson < NB2 < ZIP/ZINB):
  1. Um candidato mais complexo só vence se o intervalo pareado da melhora excluir zero a seu favor: no desvio médio para Poisson contra a taxa global; na log-verossimilhança preditiva média para NB2 contra Poisson, porque as médias quase coincidem.
  2. Além disso, o ajuste deve convergir sem erro, a calibração global (diferença de taxas) não pode piorar, e o sentido do ganho não pode se reverter ao ampliar a amostra de treino ao próximo degrau (estabilidade).
  3. Evidência inconclusiva → fica o modelo mais simples.
  4. ZIP ou ZINB só entram com **desajuste persistente dos zeros** em validação (zeros previstos − observados com intervalo excluindo zero no total e em ao menos duas faixas de exposição sem baixo suporte, depois do melhor entre Poisson e NB2) **e** com hipótese plausível para o segundo processo, escrita antes do ajuste.
  5. A escolha é congelada, com modelo, covariáveis, amostra e roteiro, antes de abrir o teste. O teste é avaliado uma única vez. Os resultados são válidos dentro do benchmark.

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

1. **A taxa alta das exposições curtas persiste dentro de perfis.** Em estratos de faixa de idade do condutor × faixa de idade do veículo, com taxa de referência das linhas de `Exposure ≥ 0,75`, a razão entre a taxa das linhas com `Exposure ≤ 0,08` e a de referência foi acima de 1 em 24 de 24 estratos com suporte nos dois grupos (mediana 4,2; mínimo 2,7). A razão observado/esperado de sinistros nas faixas curtas é 30,9 (`(0; 0,02]`), 4,8 (`(0,02; 0,08]`), 2,9, 1,6, 1,3 e 1,1, e 0,96 em `= 1`. O padrão persiste dentro dos estratos de idade com suporte; é uma comparação descritiva, que controla só a idade do condutor e a do veículo e não exclui outras explicações.
2. **O que o offset pressupõe.** `E[ClaimNb]` proporcional à exposição (coeficiente 1 em `log(Exposure)`). Como diagnóstico no treino (100 mil linhas, oito covariáveis), o coeficiente livre de `log(Exposure)` foi 0,39 (erro padrão ingênuo 0,016), longe de 1. Se o padrão se mantiver na validação, os três candidatos subestimarão as faixas curtas e superestimarão as longas.
3. **O que será conferido na validação.** Calibração por faixa de exposição (prevista − observada, por 100 apólices-ano, com IC) para os três candidatos, sem recalibração pós-hoc nem fatores por segmento. Um Poisson com `log(Exposure)` livre será avaliado como **diagnóstico exploratório, não candidato**; seu uso exigiria nova decisão do Codex na etapa 4. A métrica primária e as regras de escolha não mudam.
4. **Casos extremos.** Oito registros de treino com `ClaimNb > 4` (71 sinistros); cinco deles compartilham `DrivAge = 52` e `VehAge = 12` e somam 44 sinistros, um indício de concentração em poucos perfis (a identidade permanece indeterminada). As caudas válidas são preservadas; o bootstrap por perfil é a sensibilidade correspondente.
5. **Cobertura de categorias.** Na amostra de 100 mil linhas, `Region` R21, R42 e R43 e `VehPower` 13, 14 e 15 têm menos de 500 linhas e vão para `OTHER`. Em 374.581 linhas de treino fora da amostra, 3.618 (`Region`) e 4.700 (`VehPower`) linhas com nível agrupado ou ausente do ajuste foram mapeadas para `OTHER`, e a matriz de desenho ficou finita.

**Definições operacionais.** População de avaliação: as 101.573 linhas de validação, com contagens e exposições originais. Diferença de taxa: `100 · (Σμ − Σ ClaimNb) / Σ Exposure`. Sensibilidades no maior degrau viável: `BonusMalus` em faixas por quantis; limites didáticos só no ajuste; bootstrap por perfil; partição agrupada (pré-processamento, ajustes e avaliação refeitos em `group_train` e `group_validation`, com resultados à parte); descrição da métrica sem a faixa `(0; 0,02]`; `log(Exposure)` livre (diagnóstico).

**Divulgação.** Antes desta emenda foi executado um teste de fumaça do estimador em 100 mil linhas de treino (tempo e equivalência com o `statsmodels`), sem uso de validação.
