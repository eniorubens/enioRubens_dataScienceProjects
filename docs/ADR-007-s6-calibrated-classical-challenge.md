# ADR-007: desafio clássico calibrado S6

**Status:** aceito para desenvolvimento
**Data:** 2026-08-16

## Contexto

S3 estabeleceu o baseline com TF-IDF por palavras e `SGDClassifier`. S4 comparou representação e ponderação. S5 comparou estimadores sobre a mesma representação, mas nenhum candidato atingiu simultaneamente os três gates. `LinearSVC` foi o mais próximo, com F1 da classe crítica ligeiramente abaixo do limite.

O S6 é a última rodada clássica antes de considerar uma arquitetura hierárquica ou modelos de linguagem. O estágio continua sendo exclusivamente de desenvolvimento. As partições `test`, `stress` e `monitor` não podem ser carregadas, usadas para calibração ou usadas para seleção.

## Decisão

O S6 mantém a representação word TF-IDF congelada do S5 e separa o trabalho em duas fases temporais dentro de `train`:

1. uma divisão interna de ajuste e calibração, formada somente por grupos limpos e não ambíguos de `train`;
2. uma única avaliação externa na partição `validation`, realizada depois que candidatos, hiperparâmetros e regra de decisão forem congelados.

A calibração interna usa a margem ou o escore de decisão do candidato. Para o `LinearSVC`, a regra é um threshold de margem da classe crítica, com o valor escolhido exclusivamente na divisão interna. O threshold não é recalculado na avaliação externa.

Os cinco candidatos congelados são:

1. `LinearSVC` balanceado, como referência do S5;
2. `LinearSVC` balanceado com threshold de margem calibrado;
3. `RidgeClassifier` balanceado;
4. `SGDClassifier(loss="log_loss")` balanceado com threshold calibrado;
5. `LogisticRegression(solver="saga")` balanceada, com threshold calibrado.

O `LogisticRegression(solver="saga")` é candidato de paciência: pode exigir execução em uma máquina com mais RAM ou no Colab. O notebook não tenta contornar essa restrição nem transforma uma interrupção de recurso em resultado negativo. O pipeline esparso atual permanece orientado a CPU; a RTX 3050 local não é considerada acelerador direto para esses estimadores do scikit-learn.

A seleção exige os gates congelados do S5: macro-F1 mínimo de `0.6900`, F1 da classe crítica mínimo de `0.2715` e precisão crítica mínima de `0.2000`. O S6 termina com `NO_ELIGIBLE_CALIBRATED_CLASSICAL` se nenhum candidato satisfizer os três critérios. Nesse caso, não haverá nova busca clássica ampla: a próxima fronteira será a estratégia hierárquica ou um modelo de linguagem, definida em um ADR separado.

## Consequências

A divisão interna reduz o risco de escolher diretamente sobre a avaliação externa e deixa explícita a diferença entre calibração e confirmação. A avaliação externa única evita repetir tentativas sobre `validation`. O artefato registra a divisão, o threshold, os candidatos, a métrica interna, a métrica externa, os status de recurso e as confusões críticas, sem armazenar objetos de modelo.

O S6 não produz uma autorização de produção. Mesmo que um candidato passe os gates de desenvolvimento, qualquer decisão confirmatória futura dependerá de um protocolo posterior que preserve a fronteira selada.
