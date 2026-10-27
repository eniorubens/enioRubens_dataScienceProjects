# Model Risk & Responsible ML — UCI Adult

Auditoria reproduzível de risco, calibração e fairness aplicada ao conjunto [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult). O projeto organiza dados, modelos, probabilidades, thresholds e diferenças entre grupos em um protocolo único, rastreável e orientado a risco.

O objetivo não é produzir um modelo para decisões reais. O objetivo é mostrar como uma avaliação responsável separa desenvolvimento e teste, comunica incerteza e limita honestamente o que os dados permitem concluir.

## 💡 Filosofia do projeto

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/ModelRisk_ResponsibleML/) documenta calibração, fairness, incerteza e a fronteira do teste final.

**Produção orientada a contratos:** nos projetos com superfície de serving, APIs FastAPI/Flask expõem contratos de entrada e saída validados por testes. Nos demais, contratos de dados, artefatos, partições seladas e gates de avaliação controlam o que pode ser concluído, reproduzido e publicado.

**Superfície de engenharia:** módulos reutilizáveis de auditoria/relatório em `src/` e testes de contrato; este projeto é um artefato de auditoria, não uma API de serving.

**Pode sustentar:** revisão responsável de risco de modelo e evidência reprodutível sobre calibração, grupos e incerteza.

**Não sustenta:** decisões de crédito, emprego ou benefícios, autorização produtiva ou alegações causais de fairness.

## Resultado resumido

O protocolo selecionou um `HistGradientBoosting` com calibrador Platt e threshold `0,31`. No teste oficial congelado:

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0,9233 |
| PR-AUC | 0,8170 |
| Brier | 0,0903 |
| Precisão | 0,6330 |
| Recall / TPR | 0,8138 |
| F1 | 0,7121 |

O resultado é exploratório, `production_eligible=false` e não deve ser usado para crédito, emprego, benefícios ou triagem.

## Como ler o projeto

- [Relatório executivo PT-BR](site/index.html)
- [Executive report EN-US](site/en.html)
- [Notebook 07 — aplicação prática PT-BR](pt-BR/07_aplicacao_pratica_modelo.ipynb)
- [Notebook 07 — practical application EN-US](en-US/07_practical_model_application.ipynb)
- [Nota técnica sobre Wilson](docs/wilson_interval_reference.md)
- [Protocolo de seleção de threshold](docs/threshold_selection_protocol.md)
- [Model card PT-BR](docs/model_card_pt-BR.md)
- [Model card EN-US](docs/model_card_en-US.md)

## Estrutura

```text
pt-BR/                 narrativa canônica dos notebooks
en-US/                 espelho em inglês
src/                   lógica reutilizável e geração de artefatos
tests/                 suíte de contratos e invariantes
data/processed/        contratos, métricas, auditorias e figuras agregadas
docs/                   notas metodológicas e model cards
site/                   relatório executivo estático para GitHub Pages
```

O Notebook 07 demonstra a aplicação do estimator a uma linha do teste oficial. O intervalo de Wilson não é aplicado à linha; ele é usado para a proporção binária observada no grupo `sex = Female`, comparada à média das probabilidades calibradas do grupo.

## Reprodução

O ambiente está declarado em `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate modelrisk-responsibleml
python -m unittest -v tests/test_project_contracts.py
```

Para reproduzir os dados, faça o download do UCI Adult a partir da fonte oficial e extraia os arquivos em `data/raw/adult_uci/`. O arquivo bruto não é versionado pelo repositório; o contrato registra a proveniência e o fingerprint esperado.

## Protocolo e limitações

- O teste oficial tem 16.281 linhas e é reservado para a avaliação final.
- `sex` e `race` não entram como preditores diretos, mas proxies permanecem documentados.
- A política de duplicatas usa assinatura exata dos preditores, mantém as linhas da fonte e registra 26 assinaturas repetidas entre treino e teste, com 3 chaves de alvo discordante.
- O UCI Adult é histórico e deriva do censo norte-americano de 1994.
- O intervalo de Wilson descreve a incerteza de uma proporção binomial observada; não é um intervalo de confiança individual e não incorpora toda a variação do protocolo.

## Status de publicação

O projeto está publicado em [eniorubens/enioRubens_dataScienceProjects](https://github.com/eniorubens/enioRubens_dataScienceProjects), sob licença [MIT](LICENSE). O relatório executivo está preparado para publicação via GitHub Pages a partir da pasta `site/`.
