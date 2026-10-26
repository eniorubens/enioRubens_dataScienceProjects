# Model Card — UCI Adult Responsible ML

## Status

Este documento descreve um modelo exploratório de portfolio. O modelo não é elegível para produção (`production_eligible=false`) e não deve ser usado para decisões reais sobre pessoas.

## Resumo do modelo

O modelo selecionado é um `HistGradientBoostingClassifier` com pré-processamento encapsulado em pipeline. Os atributos `sex` e `race` não entram como preditores, mas atributos proxy permanecem no conjunto de features. `fnlwgt` é usado como preditor, não como peso amostral.

A configuração escolhida no fine-tuning foi:

| Hiperparâmetro | Valor |
|---|---:|
| `max_iter` | 250 |
| `learning_rate` | 0,05 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 30 |

O threshold final é `0,31`. Ele foi escolhido depois do ajuste do estimator e da calibração sigmoide, usando a exceção de portfolio documentada no contrato de seleção.

## Uso pretendido e usos proibidos

O uso pretendido é demonstrar um protocolo reproduzível de model risk, calibração, seleção de threshold e auditoria de fairness em um dataset histórico. O modelo não deve ser usado para crédito, emprego, benefícios, triagem ou qualquer decisão automatizada sobre pessoas reais.

## Dados

O dataset é o UCI Adult, derivado de dados censitários de 1994, com 48.842 linhas. O treino oficial contém 32.561 linhas e o teste oficial contém 16.281 linhas. O alvo é a classificação de renda `>50K` versus `<=50K`.

O conjunto possui valores ausentes em variáveis categóricas e duplicatas registradas no contrato de dados. Adotamos como identidade operacional a assinatura exata dos preditores, sem incluir o alvo, preservando as linhas originais e evitando deduplicação silenciosa. O diagnóstico final encontrou 26 assinaturas repetidas entre treino e teste, envolvendo 28 linhas de treino e 26 de teste; três assinaturas apresentam alvo discordante. Essa sobreposição limita a interpretação do teste como evidência completamente independente.

Fonte: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult), DOI `10.24432/C5XW20`.

## Treinamento e seleção

O fine-tuning avaliou seis configurações do HistGradientBoosting por previsões OOF group-aware dentro do bloco `fit`. A ordem de seleção foi PR-AUC, ROC-AUC e Brier. O ganho sobre a configuração baseline foi pequeno: PR-AUC de 0,818356 para 0,818842 no artefato de tuning.

Depois do tuning, o modelo foi ajustado no bloco `fit`, as probabilidades foram calibradas no bloco `calibration` e o threshold foi escolhido no bloco `selection`. O teste oficial não participou dessas decisões.

## Desempenho no teste oficial

O teste oficial foi usado uma única vez, com modelo, calibrador e threshold congelados.

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0,9233 |
| PR-AUC | 0,8170 |
| Brier | 0,0903 |
| Precisão | 0,6330 |
| Recall / TPR | 0,8138 |
| F1 | 0,7121 |
| Taxa de seleção | 0,3037 |

Esses valores descrevem o snapshot avaliado e não constituem garantia de desempenho futuro.

## Auditoria por grupo

Os intervalos apresentados são intervalos de Wilson de 95%. No teste, as principais diferenças marginais por sexo foram:

| Grupo | Taxa de seleção | TPR | FPR | PPV |
|---|---:|---:|---:|---:|
| Female | 0,134 | 0,753 | 0,059 | 0,611 |
| Male | 0,388 | 0,825 | 0,201 | 0,637 |

As categorias raciais menores têm denominadores reduzidos em TPR e PPV, com `low_support=true` e intervalos mais largos. As diferenças observadas são descritivas; não demonstram causalidade, intenção discriminatória ou ausência de viés.

## Limitações

O dataset representa uma população e um contexto históricos específicos. A exclusão direta de `sex` e `race` não elimina proxies, pois variáveis como `relationship`, `marital_status`, `education`, `occupation` e `native_country` permanecem disponíveis. O fine-tuning foi uma busca pequena, não exaustiva. As métricas de grupo e globais vêm de uma única partição final e não capturam toda a variância de repetição de splits.

A sobreposição de assinaturas entre treino e teste, incluindo alvos discordantes, reduz a força da alegação de independência do teste. A análise de sensibilidade excluindo as 26 linhas sobrepostas reteve 16.255 linhas e produziu ROC-AUC 0,9232, PR-AUC 0,8170, Brier 0,0903, precisão 0,6326, recall 0,8135 e F1 0,7118, muito próximos do benchmark principal. O resultado permanece exploratório e não elegível para produção.

## Reprodutibilidade e artefatos

Os contratos e resultados estão em:

- `data/processed/tuning/`: métricas e contrato do fine-tuning;
- `data/processed/selection/`: métricas, auditoria e contrato da seleção;
- `data/processed/final_evaluation/`: métricas, auditoria e contrato do teste final;
- `src/hyperparameter_tuning.py`: ajuste do estimator;
- `src/threshold_selection.py`: calibração e seleção do threshold;
- `src/final_evaluation.py`: avaliação final única;
- `pt-BR/05a_ajuste_hiperparametros.ipynb`, `pt-BR/05_selecao_modelo_threshold.ipynb` e `pt-BR/06_avaliacao_final_test.ipynb`: narrativa executada.

## Decisão de governança

O modelo pode ser apresentado como vencedor exploratório deste portfolio, sob a exceção de gate aprovada. Ele não deve ser apresentado como modelo pronto, validado externamente, causalmente justo ou adequado para produção.
