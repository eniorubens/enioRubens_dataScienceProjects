# Plano da avaliação F2 (pós-hoc)

**Estado:** **aprovado em 2026-11-14** (P-06 em `REGISTRO-DE-DECISOES.md`), com as três propostas da seção 7. Implementado (T-10) e **executado em 2026-11-14**; resultado e observações em `REGISTRO-DE-DECISOES.md`, seção "Abertura do teste", linha F2.
**Natureza:** **pós-hoc e exploratória.** As hipóteses abaixo nasceram das observações do F1 em jul–ago/2018. Por isso o F2 não substitui o F1, não confirma nem refuta nada sobre o modelo pré-registrado e não altera `production_eligible = false`. O resultado do F2 só serve para formular hipóteses a testar em dados futuros.

## 1. O que o F1 mostrou e por que isso limita o F2

O F1 foi inconclusivo: o CatBoost teve AP de 0,082 contra 0,084 da baseline promessa + UF. Três sinais aparecem no teste:

1. O prazo prometido sozinho foi o melhor preditor (AP 0,142).
2. O CatBoost superestimou o risco (prevalência de 7,5% no treino contra 4,8% no teste).
3. As taxas históricas da rede mudaram muito (PSI até 6,3).

Uma verificação feita só com rótulos do desenvolvimento mostrou que **esse regime não existe nos folds de desenvolvimento**:

- A AUC da regra "−prazo prometido" fica entre 0,40 e 0,63 em todos os meses de 2017-01 a 2018-06, ou seja, perto do acaso.
- A mediana do prazo prometido oscila entre 21 e 27 dias até junho de 2018 e cai para 19,3 dias em julho e 13,4 em agosto.

Consequência: nenhuma variante baseada no comportamento da promessa pode ser validada no desenvolvimento. Qualquer ganho no teste seria, em parte, ajuste ao próprio teste. O F2 é desenhado para conter esse risco, não para eliminá-lo.

## 2. Desenho: implantação simulada em 2018-08-01

Em vez de reutilizar jul–ago inteiro, o F2 simula o que uma operação real faria: o modelo entra em produção em 1º de julho e é **atualizado em 1º de agosto** com os rótulos que já seriam conhecidos naquela data.

| Item | Definição |
|---|---|
| Instante de atualização | 2018-08-01 |
| Rótulos disponíveis na atualização | Pedidos com `order_delivered_customer_date` < 2018-08-01. Isso inclui junho e parte de julho, que é exatamente a informação que uma operação teria. |
| Janela avaliada | Compras em agosto de 2018 (≈ 6.351 pedidos, 393 violações, já observadas no F1) |
| Blocos do bootstrap | **Diários** (≈ 29 blocos), 1.000 reamostragens, semente 42. Agosto tem só ~5 semanas, então blocos semanais dariam intervalos sem sentido. |
| Comparações | Diferença pareada de AP de cada variante contra (a) `baseline_logistic_promise_plus_state` retreinada em 2018-08-01 e (b) a regra `negative_promised_lead_days`. Também são reportados Brier, ECE, confiabilidade e top 5%/10%. |

## 3. Variantes (lista fechada; todas são reportadas, nenhuma é escolhida pelo teste)

| Código | Variante | Mecanismo testado |
|---|---|---|
| A | Modelo do F1 congelado (treino < 2018-06-01), aplicado a agosto | Referência: sem atualização |
| B | Mesma configuração do F1, **retreinada** em 2018-08-01 com os rótulos conhecidos | Retreino mensal |
| C | B + **pesos por recência** (decaimento exponencial, meia-vida de 90 dias) | Adaptação a mudança de regime |
| D | B + **recalibração recente**: o modelo é ajustado com rótulos até 30 dias antes da atualização, e a calibração isotônica usa só os rótulos dos últimos 30 dias | Correção de mudança de prevalência |
| E | B + feature **`promise_vs_recent_median`**: razão entre o prazo prometido do pedido e a mediana dos prazos prometidos nos 30 dias anteriores para a mesma UF de destino. É calculada as-of, só com compras anteriores, e não usa rótulo. | Captura de mudança na política de promessa |

As features de rede continuam as-of em todas as variantes. Nenhuma variante usa rótulos de agosto.

## 4. Gate no desenvolvimento antes de tocar no teste

Cada mecanismo de C, D e E é aplicado nos 9 folds de 2017-10 a 2018-06, com o mesmo protocolo do torneio, contra a configuração do F1. Uma variante só segue para a avaliação de agosto se for **não inferior** no desenvolvimento:

- AP média ≥ 95% da AP média da configuração do F1;
- ECE médio no máximo 0,01 acima do ECE médio do F1.

Variantes reprovadas no gate são reportadas como tal, com os números do desenvolvimento, e não são avaliadas em agosto. A e B não passam pelo gate: A é a referência e B só retreina com a configuração já aprovada.

## 5. Regras de leitura e de uso

- O F2 usa as versões `F2` em `run_final_evaluation`/`cached_run`, com `post_hoc: true`, semântica congelada e ledger (T-09). Uma linha é acrescentada ao ledger.
- Todos os resultados são apresentados como **exploratórios**. Com 4 variantes comparadas, nenhuma diferença é chamada de "confirmada"; o texto descreve tamanho de efeito e intervalo.
- Uma variante só pode ser recomendada para um **próximo ciclo**, a ser validada em dados posteriores a agosto de 2018 (inexistentes no Olist). Não pode ser recomendada para produção.
- O relatório final apresenta o F1 como resultado oficial e o F2 como análise complementar.

## 6. Implementação executada

- **`src/ecom_delivery_sla/posthoc.py`:**
  - função de rótulos conhecidos as-of de uma data;
  - pesos por recência;
  - recalibração em janela recente;
  - feature `promise_vs_recent_median` as-of;
  - gate de não inferioridade nos folds;
  - `run_posthoc_evaluation(..., allow_test=True, version="F2")`, reaproveitando o bootstrap e o mecanismo de congelamento do `final_evaluation.py`, com blocos diários.
- **Testes sintéticos:**
  - nenhum rótulo com entrega ≥ data de atualização entra no treino;
  - a feature de promessa não usa compras posteriores nem o próprio pedido;
  - os pesos decaem corretamente;
  - a recalibração usa só a janela recente;
  - o gate classifica corretamente;
  - a versão F2 é marcada como pós-hoc;
  - o F2 é congelado como o F1.
- **Notebooks `10_pos_hoc_f2.ipynb` (PT) e `10_post_hoc_f2.ipynb` (EN):** código idêntico, narrativa em voz passiva e sem bullets, com uma primeira seção que declara a natureza pós-hoc e as limitações.
- A execução foi realizada pelo responsável em 2026-11-14, com os artefatos F2 gravados e o resultado classificado como exploratório.

## 7. Decisões encerradas no P-06

1. Foi adotado o desenho de implantação simulada com atualização em 2018-08-01 e avaliação somente em agosto.
2. Foi adotada a lista fechada de variantes A–E.
3. Foi adotado o gate de não inferioridade: AP ≥ 95% e ECE com folga de 0,01.
