# Intervalo de Wilson no projeto UCI Adult

## Objetivo

Este documento explica por que as taxas apresentadas na auditoria por sexo, raça e interseção podem ser tratadas como proporções binomiais e por que o projeto utiliza o intervalo de Wilson para comunicar a incerteza dessas estimativas.

## Por que nossos indicadores são proporções binomiais

Uma proporção binomial surge quando observamos, para cada unidade analisada, um resultado com duas possibilidades e contamos quantos sucessos ocorreram em um número definido de tentativas.

No projeto, cada métrica de grupo pode ser escrita como:

$
\hat p = \frac{x}{n}
$

onde:

- `x` é o número de sucessos observados;
- `n` é o denominador da métrica;
- `\hat p` é a taxa estimada.

O denominador depende da métrica:

| Métrica | Sucesso contado | Denominador |
|---|---|---|
| Taxa de seleção | previsão positiva | total de pessoas do grupo |
| TPR / recall | verdadeiro positivo | positivos reais do grupo |
| FPR | falso positivo | negativos reais do grupo |
| FNR | falso negativo | positivos reais do grupo |
| PPV / precisão | verdadeiro positivo | previsões positivas |

Por exemplo, para o grupo Female, a taxa de seleção é a fração de previsões positivas entre todas as observações Female. Já o TPR é a fração de verdadeiros positivos entre as observações Female que realmente pertencem à classe positiva.

Assim, para cada grupo e métrica, temos um número de sucessos em um denominador definido. Esse é o objeto estatístico de uma proporção binomial.

## O que o intervalo estima

A taxa observada `\hat p` descreve a amostra disponível no teste. O intervalo de confiança procura quantificar a incerteza de amostragem associada à proporção que seria observada em uma população ou repetição de amostras comparável.

Em termos práticos, um intervalo de 95% mais largo indica que o grupo ou o denominador fornece menos informação para estimar a taxa. Um intervalo mais estreito indica maior precisão amostral, não necessariamente ausência de viés.

O intervalo não afirma que existe uma probabilidade de 95% de a taxa verdadeira estar dentro daquele intervalo depois que os dados foram observados. A interpretação frequentista é sobre o comportamento do procedimento em repetições do processo de amostragem: intervalos construídos por esse procedimento cobririam a proporção de interesse aproximadamente em 95% das repetições, sob as condições do modelo.

## Fórmula de Wilson

Para `x` sucessos em `n` observações, com nível de confiança bilateral associado ao quantil `z`, o intervalo de Wilson é:

$
\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}
{1 + \frac{z^2}{n}}
$

com:

$
\hat p = \frac{x}{n}
$

Para 95% de confiança, usamos aproximadamente `z = 1,96`.

O denominador ajustado `1 + z²/n` e o termo adicional dentro da raiz fazem com que o intervalo se comporte melhor do que a aproximação normal simples, especialmente quando a proporção está próxima de 0 ou 1 ou quando `n` não é grande.

## Por que não usar apenas a aproximação normal

A aproximação normal clássica costuma ser escrita como:

$
\hat p \pm z\sqrt{\frac{\hat p(1-\hat p)}{n}}
$

Ela pode produzir limites abaixo de 0 ou acima de 1 e tende a apresentar cobertura inadequada em proporções extremas ou amostras pequenas. Isso é particularmente relevante na auditoria do projeto, porque algumas combinações raça × sexo têm poucos positivos, poucos negativos ou poucas previsões positivas.

O intervalo de Wilson permanece dentro do domínio `[0, 1]` e fornece uma aproximação mais adequada para esse cenário.

## Como interpretar a auditoria do projeto

No teste oficial, por exemplo:

- Female tem TPR aproximadamente `0,753`, com intervalo de Wilson de 95% aproximadamente `[0,716, 0,786]`;
- Male tem TPR aproximadamente `0,825`, com intervalo aproximadamente `[0,812, 0,838]`.

Esses intervalos comunicam a incerteza amostral de cada taxa. Eles não são, sozinhos, um teste de igualdade entre grupos. A sobreposição ou não sobreposição de intervalos individuais não substitui um intervalo específico para a diferença entre taxas, uma razão de taxas ou um teste estatístico pré-especificado.

Para grupos raciais menores, o denominador de TPR, FNR ou PPV pode ser bastante reduzido. Nesses casos, o contrato registra `low_support=true`. Uma taxa pontual elevada ou igual a zero não deve ser interpretada como risco alto ou inexistente sem considerar o denominador e a largura do intervalo.

## Hipóteses e limites

O intervalo de Wilson é apropriado para uma proporção binomial sob uma leitura de ensaios aproximadamente independentes e uma única partição observada. No nosso projeto, isso não significa que todas as fontes de incerteza estejam resolvidas:

- o teste é uma única partição do snapshot UCI Adult;
- a escolha do modelo e do threshold ocorreu antes, mas foi feita em dados de desenvolvimento;
- a escolha do threshold introduz incerteza que o Wilson simples não incorpora;
- múltiplos grupos e múltiplas métricas aumentam o número de comparações;
- repeated splits ou bootstrap poderiam capturar parte da variância de particionamento;
- a sobreposição de assinaturas entre treino e teste limita a interpretação do teste como evidência completamente independente;
- intervalos de Wilson não corrigem proxies, drift, erro de medição ou viés de seleção.

Portanto, no relatório, Wilson é usado como uma forma transparente de mostrar incerteza de amostragem em torno das taxas observadas — não como prova de fairness, independência, causalidade ou ausência de discriminação.

## Implementação no projeto

A implementação reutilizável está em `src/validation_audit.py`, na função `_wilson_interval`. As linhas de auditoria materializadas incluem:

- `estimate`: proporção observada;
- `ci_low`: limite inferior;
- `ci_high`: limite superior;
- `denominator`: denominador usado pela métrica;
- `low_support`: indicação de suporte reduzido.

Esse desenho mantém a estimativa, o denominador e a incerteza juntos no artefato CSV, permitindo que o notebook e o relatório HTML apresentem a mesma evidência.

## Referência

Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference*. **Journal of the American Statistical Association**, 22(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

