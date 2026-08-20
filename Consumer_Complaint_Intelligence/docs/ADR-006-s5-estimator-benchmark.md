# ADR-006: benchmark de estimadores S5

**Status:** aceito para desenvolvimento
**Data:** 2026-08-15

## Contexto

O S3 estabeleceu o baseline com SGDClassifier e TF-IDF por palavras. O S4 comparou representações e pesos, mas não isolou o efeito do estimador. O S5 isola essa variável usando a mesma representação word baseline.

O experimento é exclusivo de desenvolvimento. As partições test, stress e monitor permanecem seladas e não são carregadas pelo executor.

## Decisão

Usar uma única representação TfidfVectorizer, ajustada somente em train, com analyzer="word", ngram_range=(1, 2), max_features=40000, min_df=2, max_df=0.98, sublinear_tf=True e dtype="float32".

Os candidatos locais são ajustados sequencialmente sobre a mesma matriz:

1. SGDClassifier(loss="log_loss", class_weight="balanced"), como referência.
2. LinearSVC(class_weight="balanced").
3. ComplementNB(alpha=1.0, norm=False), com pesos balanceados.

LogisticRegression(solver="saga") foi adiado por restrição de recurso. A GPU NVIDIA RTX 3050 não acelera diretamente esse pipeline esparso do scikit-learn. Uma execução posterior pode usar Colab ou uma máquina com mais RAM, sem alterar a fronteira selada.

O artefato registra runtime, avisos de convergência e paridade com a referência S4. A seleção exige os gates congelados: macro-F1 mínimo de 0,69, F1 da classe crítica mínimo de 0,2715 e precisão crítica mínima de 0,20.

## Consequências

O benchmark compara uma variável metodológica por vez e mantém o consumo de memória controlado. Os estimadores são liberados sequencialmente e o artefato JSON não contém objetos de modelo. A saída permanece consumível por notebook, MLflow opcional e futura camada Flask.
