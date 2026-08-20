# ADR-008: S7 congelamento e empacotamento do modelo

**Data:** 2026-08-16
**Status:** aceito e executado

## Decisão

O S7 congela um único candidato: `TfidfVectorizer` por palavras, com unigramas
e bigramas, `max_features=40000`, `min_df=2`, `max_df=0.98`, `sublinear_tf`
e `float32`, seguido de `LinearSVC(C=0.3, class_weight="balanced")`.

O vectorizer e o estimator foram ajustados em toda a partição `train`, de
2023-08-01 a 2024-06-30. A partição `validation`, de 2024-07-01 a
2024-12-31, foi reutilizada uma única vez para a calibração final do
`critical override threshold`, usando exatamente `search_thresholds_exact` e
a regra de margem de S6. Não haverá refit depois da calibração.

Essa reutilização transforma a validation em fonte de calibração final, não em
evidência independente. A confirmação científica ficará para S8, após a
abertura protocolar do test. `test`, `stress` e `monitor` permanecem selados.

## Resultado operacional

O ajuste final utilizou 345552 observações de `train`, e a calibração do limiar
utilizou 245980 observações de `validation`. O threshold selecionado foi
`0.1135351095114484`. Na calibração final, o macro-F1 foi
`0.7219892370336836`; para a classe crítica, precision foi
`0.39731285988483683`, recall foi `0.2320627802690583`, F1 foi
`0.2929936305732484` e support foi 892. Os três gates passaram, resultando no
status `packaged_for_confirmation`.

O runtime científico foi 267.289s. O monitoramento total registrou 284.330s,
peak RSS de 3.707GB e mínimo de memória disponível de 2.902GB. Essas métricas
são da calibração final em `validation` e não constituem evidência
confirmatória. `test`, `stress` e `monitor` seguem selados.

## Empacotamento

O bundle joblib foi publicado em
`artifacts/s7/consumer_complaint_classifier_s7.joblib`. Ele contém o vectorizer,
o estimator, o threshold, a ordem de classes, a classe crítica, a versão do
modelo e `input_language="en-US"`.

O predictor valida lotes, narrativas não vazias e o idioma declarado. Ele
reordena `decision_function` pela ordem congelada, aplica a margem crítica e
retorna `PredictionBatch`. O campo `score` significa exclusivamente
`critical_margin`; não representa probabilidade nem confiança. A entrada do
estimator é exclusivamente `en-US`, embora a API e a documentação possam ser
bilíngues.

O resultado operacional fica em `temp/s7/s7_results.json`. O manifesto
publicável fica em `config/s7_results.json` e registra hashes independentes do
bundle e do resultado, tamanhos, runtime e a ausência de evidência
confirmatória. Os caminhos publicados são relativos ao projeto; os hashes e
tamanhos são calculados sobre os arquivos reais. O manifesto não inclui o
próprio hash, evitando circularidade.

O status `packaged_for_confirmation` só é emitido se o threshold selecionado
passar os três gates. Caso contrário, o status é `calibration_gate_failed` e o
pacote não avança para confirmação nem para serving.

## Consequências

- O pacote é `packaged_for_confirmation`, `development_only`, `deploy=false` e
  `confirmatory=false`.
- Um resultado com `calibration_gate_failed` permanece explicitamente retido em
  desenvolvimento.
- Smoke é apenas diagnóstico e nunca substitui o bundle final.
- Um loader verifica manifesto, hashes, idioma, versão e ordem de classes antes
  de devolver um predictor para `PredictionService` ou Flask.
- A transformação da validation ocorre em batches e o joblib é substituído de
  forma atômica.
- MLflow permanece opcional; a superfície de serving não depende dele.
