# ADR-001: Arquitetura inicial para Consumer Complaint Intelligence

## Status

Aceito para a S0. Esta decisão define fronteiras, mas não escolhe a tarefa de
modelagem nem o split temporal final.

## Decisões

- O código de dados usa `Polars` com scans lazy para a auditoria amostral e
  preparação. `DuckDB` executa o perfil agregado do corpus completo em
  consultas separadas, com limite de memória, threads e spill explícitos.
- O estimator/pipeline scikit-learn não conhece Flask, HTTP, JSON ou MLflow.
  O domínio expõe `Predictor`, `PredictionBatch` e `ArtifactManifest`.
- Uma aplicação futura depende de `PredictionService`. Um adaptador Flask ou
  outro servidor traduz entrada e saída da aplicação para esse contrato, sem
  importar detalhes do estimator para a camada web.
- MLflow fica na orquestração. `NullTracker` é o padrão; `MlflowTracker` só
  importa MLflow quando selecionado explicitamente. Assim, MLflow pode registrar
  parâmetros, métricas, pipeline e artefatos sem contaminar o domínio.
- Os notebooks PT-BR são canônicos. Os notebooks EN-US são editorialmente
  equivalentes e compartilham o mesmo código executável, chamando `src` em vez
  de manter lógica analítica duplicada.
- `audit_s0_sample` é um smoke test limitado ao início do Parquet e rotulado
  como evidência amostral. `audit_s0_corpus` faz os agregados completos sem
  materializar narrativas em Python.
- Duplicatas exatas do corpus usam chave MD5 mais tamanho do texto dentro do
  DuckDB. O relatório explicita o risco teórico de colisão e não expõe textos.

## Consequências

Esta estrutura permite testar contratos sem Flask, executar notebooks sem
instalar MLflow e trocar o mecanismo de serving sem reescrever o pipeline. O
preço é manter uma fronteira explícita entre texto bruto, matriz esparsa de
TF-IDF e artefato persistido. A escolha da taxonomia, deduplicação segura e
split temporal continua pendente e não deve ser inferida de uma amostra.

## Dependências futuras opcionais

Flask e MLflow não entram no `environment.yml` nesta etapa. Quando houver
serving ou tracking, podem ser adicionados pelos extras do `pyproject.toml` ou
em ambientes separados, sem alterar os contratos do pacote.
