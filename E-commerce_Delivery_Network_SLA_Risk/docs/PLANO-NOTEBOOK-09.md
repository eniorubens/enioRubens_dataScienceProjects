# Plano do notebook 09 — avaliação final única

**Estado:** pré-registro **aprovado em 2026-11-14** (P-05 em `REGISTRO-DE-DECISOES.md`), com as três decisões da seção 6: configuração padrão do torneio, treino até 2018-05-31 e critério pelo IC 95% da diferença pareada de AP. O teste F1 foi aberto em 2026-11-14 e o resultado está congelado no cache correspondente.
**Base:** P-01 a P-04 aprovados. Candidato da classificação: CatBoost. Regressão: sem candidato.

## 1. Objetivo

O período de 2018-07-01 a 2018-08-31 deve ser usado uma única vez para responder a duas perguntas:

1. O CatBoost, treinado com todo o histórico disponível, mantém fora do período de desenvolvimento a vantagem sobre a melhor baseline observada nos folds?
2. As probabilidades continuam calibradas sob o drift conhecido do período? A mediana do prazo prometido cai de ~24 para ~16 dias.

O notebook não seleciona modelos nem ajusta nada. Ele mede e documenta.

## 2. Pré-registro: o que fica fixado antes de abrir o teste

| Item | Proposta | Justificativa |
|---|---|---|
| Modelo | CatBoost com a configuração padrão do torneio (`iterations=300`, `learning_rate=0.05`, `depth=6`, `random_seed=42`), sem pesos de classe, calibração isotônica `cv=3` no treino (T-05) | Foi a configuração que atendeu ao P-01. O Optuna não trouxe ganho relevante, e usar a configuração já validada evita escolher hiperparâmetros com base em mais buscas. |
| Features | Exatamente `NUMERIC_FEATURES` + `CATEGORICAL_FEATURES` atuais, incluindo a rede as-of e `distance_mean_km` (P-03) | É o contrato validado nos folds. |
| Janela de treino | Compras de 2016-09 a **2018-05-31** (embargo de 30 dias antes de 2018-07-01) | Mantém a mesma regra dos folds: no instante da implantação, os pedidos de junho ainda não teriam desfecho conhecido. |
| População de teste | Pedidos supervisionados com compra em [2018-07-01, 2018-09-01); pedidos `shipped` sem entrega excluídos (P-02) | O cenário `include_as_breach` é reportado só em contagens e prevalência. |
| Baselines no teste | As três de classificação (prevalência, `−promised_lead_days`, logística promessa + UF) e as três de regressão, ajustadas na mesma janela de treino | Comparação justa. A logística promessa + UF é a referência principal. |
| Features de rede no teste | As-of por pedido, usando desfechos conhecidos antes de cada compra, inclusive os de julho | Reproduz um sistema em produção, que atualiza o histórico continuamente. Isso é declarado como premissa. |
| Métricas | AP (principal), ROC AUC, Brier, ECE (10 bins), curva de confiabilidade, precisão e recall em top 5% e top 10% | São as mesmas do torneio. |
| Incerteza | Bootstrap por blocos semanais (1.000 reamostragens, semente fixa), com IC 95% para cada métrica e para a **diferença pareada** de AP entre CatBoost e baseline | Os pedidos da mesma semana são correlacionados. |
| Critério de leitura | O ganho é considerado **confirmado** se a diferença de AP CatBoost − baseline for positiva e seu IC 95% excluir zero. É **inconclusivo** se o IC incluir zero, e **não confirmado** se a diferença for negativa com IC abaixo de zero | É fixado antes de ver o resultado. O critério não muda depois. |
| Regressão | Apenas descritiva: MAE e RMSE da baseline por UF e do melhor modelo do torneio (CatBoost regressor padrão), sem declarar vencedor | Consistente com o P-04. |
| Drift | PSI treino × teste por feature, com destaque para `promised_lead_days` e as taxas da rede; prevalência mensal | Explica eventual queda de desempenho. |

## 3. Regra de uso único do teste

- A avaliação é implementada como função em `src/` (ex.: `run_final_evaluation(...)`) e é a única chamada com `allow_test=True`.
- O resultado é gravado com `cached_run("final_evaluation", "F1", ...)` no modo `"auto"`. Uma reexecução do notebook, inclusive no outro idioma, **carrega** o resultado em vez de reavaliar.
- O modelo final treinado é salvo (joblib) em `outputs/models/catboost_final_F1/`, com o `metadata.json` do cache. Só esse artefato vai para o notebook 09.
- Ao abrir o teste, uma linha é acrescentada ao `REGISTRO-DE-DECISOES.md` com a data, a versão `F1` e o fingerprint.
- Qualquer avaliação posterior com outra configuração recebe outra versão (`F2`...) e é rotulada como **pós-hoc** em todo texto que a cite.
- Deve ser declarado que versões antigas do projeto (outro alvo e outras features) já consultaram esse período. A avaliação F1 é a primeira feita com o contrato atual, sem seleção baseada nele.

## 4. Estrutura do notebook (PT e EN, código idêntico, narrativa em voz passiva e sem bullets)

1. Abertura: objetivo, pré-registro (tabela do item 2) e regra de uso único.
2. Dados e populações: tamanhos do treino e do teste, prevalência e cenário de censura.
3. Drift treino × teste: PSI e prevalência mensal.
4. Avaliação da classificação: tabela de métricas com IC, diferença pareada contra a baseline, curva de confiabilidade e top-k.
5. Regressão descritiva.
6. Leitura pelo critério pré-registrado, limitações (censura, drift da promessa, histórico com viés de atraso de rótulo, uso anterior do período) e implicação para `production_eligible`, que permanece `false`.

## 5. Implementação executada

- `src/ecom_delivery_sla/final_evaluation.py`: treino na janela fixada, predição no teste, métricas, bootstrap por blocos semanais, diferença pareada, PSI e persistência via `run_cache`.
- Testes sintéticos:
  - o bootstrap é reprodutível com a semente fixa;
  - os blocos semanais estão corretos;
  - o critério de leitura dá a classificação certa nos três casos;
  - a função recusa rodar sem `allow_test=True`;
  - o treino não contém compras ≥ 2018-06-01.
- Notebooks `09_avaliacao_final.ipynb` (PT) e `09_final_evaluation.ipynb` (EN), incluídos nos testes de paridade e de sequência.
- A avaliação real foi executada pelo responsável em 2026-11-14, com uso único do teste F1 e persistência do resultado no cache e no ledger.

## 6. Decisões encerradas no P-05

1. Foi adotada a configuração padrão do torneio, e não a seleção posterior pelo Optuna.
2. Foi adotada a janela de treino com embargo até 2018-05-31.
3. Foi adotado o critério de leitura pelo IC 95% da diferença pareada de AP.
