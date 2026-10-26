# Protocolo de seleção de modelo e limiar

Este documento descreve o contrato da etapa de desenvolvimento representada pelo Notebook 5. A etapa seleciona um candidato e um limiar sem consultar o `test_official`. Ela não autoriza, por si só, o uso do modelo em produção nem substitui a avaliação final no teste oficial.

## Contrato da etapa

O conjunto `train_official` é dividido em três papéis por `StratifiedGroupKFold`, com `random_state=42`:

- `fit`: ajuste do pipeline candidato;
- `calibration`: ajuste exclusivo do calibrador sigmoide de probabilidades;
- `selection`: comparação dos limiares e escolha operacional.

As assinaturas de grupo são calculadas sobre os preditores do modelo, após a representação explícita de valores ausentes e sem incluir o alvo. Adotamos essa assinatura exata como identidade operacional conservadora: observações com a mesma chave não atravessam os blocos de desenvolvimento. As linhas originais são preservadas; não há deduplicação silenciosa da fonte.

O `test_official` permanece reservado. O contrato materializado em `data/processed/selection/threshold_selection_contract.json` deve registrar `test_used=false` e `final_test_evaluation_pending=true`.

O protocolo também registra, como diagnóstico de qualidade separado do ajuste e da seleção, assinaturas de preditores repetidas entre `train_official` e `test_official`. No snapshot atual são 26 assinaturas, envolvendo 28 linhas de treino e 26 de teste; 3 assinaturas têm alvo discordante. A política está adotada: essas linhas permanecem no snapshot original, a sobreposição é reportada e o teste é interpretado como benchmark exploratório, não como evidência completamente independente.

## Cenário de risco

O cenário provisório é **prioridade de recall com piso de precisão**. O recall reduz falsos negativos, enquanto o piso de precisão impede que a busca por cobertura produza uma quantidade desproporcional de falsos positivos. Para tornar a decisão auditável, o grid avalia thresholds de 0,10 a 0,90 em passos de 0,01.

Os pisos atuais, `recall >= 0,80` e `precision >= 0,65`, são parâmetros de trabalho e não requisitos de negócio validados. A política escolhe o menor threshold que atende os dois pisos. Se nenhum candidato atende simultaneamente às restrições, o protocolo registra a exceção e usa o threshold com maior F1, priorizando recall e precisão em caso de empate. Essa exceção precisa ser discutida antes de qualquer alegação de adequação operacional.

O grid usa incrementos de 0,01 para não ocultar soluções intermediárias entre 0,30 e 0,35. Se ainda assim nenhum ponto atender os dois pisos, o contrato registra `fallback_max_f1_no_threshold_met_floors` e o threshold com maior F1. Esta escolha é uma decisão de desenvolvimento, não uma confirmação de risco aceitável.

Na execução atual, os thresholds 0,33 e 0,34 continuam separados pelos pisos: em 0,33 o recall é aproximadamente 0,808 e a precisão 0,647; em 0,34 a precisão sobe para aproximadamente 0,655, mas o recall cai para 0,799. Portanto, mesmo com passo de 0,01 não há vencedor que atenda o gate original. O fallback por maior F1 seleciona 0,38.

O campo `risk_gate_passed` permanece falso nesse resultado e `decision_status` registra que a revisão de risco ainda é obrigatória. Portanto, o termo “modelo selecionado” no notebook significa apenas candidato provisório para a próxima avaliação; não significa que os pisos foram atendidos.

## Ajuste de hiperparâmetros

Antes da seleção final de threshold, o Notebook 5a ajusta somente o `hist_gradient_boosting`, que foi o vencedor da comparação baseline. A busca é uma grade explícita de seis configurações, avaliada por previsões OOF group-aware dentro do bloco `fit`, com prioridade para PR-AUC, seguida por ROC-AUC e Brier. Threshold e calibração não participam do objetivo do tuning.

Na execução atual, a configuração vencedora é `max_iter=250`, `learning_rate=0,05`, `max_leaf_nodes=15` e `min_samples_leaf=30`. O Notebook 5 consome essa configuração, calibra as probabilidades em `calibration` e seleciona o threshold em `selection`.

## Calibração e seleção de modelo

Os modelos candidatos são comparados por previsões out-of-fold dentro do bloco `fit`. O candidato selecionado é então ajustado novamente apenas em `fit`. O calibrador Platt é ajustado com as probabilidades do bloco `calibration`; a tabela de limiares e as métricas usadas para a decisão vêm exclusivamente do bloco `selection`.

Neste MVP, Optuna não é usado. Uma otimização multiobjetivo de recall e precisão só deve ser considerada depois de fixar a definição do risco, o conjunto de restrições, a estratégia de validação e a forma de comunicar soluções Pareto. Caso seja adotada, deverá permanecer dentro dos blocos de desenvolvimento e não poderá acessar o teste oficial.

## Limitações e próxima etapa

As métricas de seleção ainda estão sujeitas à incerteza da amostra e à escolha dos pisos. As tabelas por sexo e raça no Notebook 5 são evidência de auditoria do threshold selecionado, não prova de ausência de disparidade. Grupos raros continuam exigindo interpretação cautelosa por causa dos intervalos largos.

## Avaliação final

O modelo ajustado, o calibrador e o threshold foram congelados antes da avaliação final. O `test_official` foi usado uma única vez pelo módulo `src/final_evaluation.py`; os artefatos estão em `data/processed/final_evaluation/` e o Notebook 6 apenas os carrega para inspeção, sem reexecutar o teste.

No teste oficial, o modelo obteve ROC-AUC de aproximadamente 0,9233, PR-AUC de 0,8170, Brier de 0,0903, precisão de 0,6330, recall de 0,8138 e F1 de 0,7121 no threshold 0,31. Esses números são evidência do snapshot avaliado, não uma autorização para produção. O contrato final mantém `production_eligible=false` e registra a política adotada e a sobreposição de assinaturas como limitação conhecida.

Como sensibilidade de independência, foram removidas apenas as 26 linhas do `test_official` cuja assinatura exata de preditores também aparece no treino. Restaram 16.255 linhas: ROC-AUC 0,9232, PR-AUC 0,8170, Brier 0,0903, precisão 0,6326, recall 0,8135 e F1 0,7118. A proximidade com o benchmark principal indica estabilidade descritiva neste snapshot, mas não converte o teste em validação externa.

Também permanece pendente a política de reajuste final: se o modelo for reajustado com mais dados de desenvolvimento antes do teste, o calibrador deverá ser refeito sobre probabilidades produzidas pelo modelo reajustado, mantendo o threshold congelado e sem consultar o teste.
