# Guia de defesa — Consumer Complaint Intelligence

Este documento reúne decisões metodológicas que podem surgir em uma entrevista
técnica. Ele descreve o que foi efetivamente implementado; não propõe mudanças
retroativas no experimento congelado.

## Por que não removemos stopwords?

### Resposta curta para entrevista

> Não aplicamos uma lista explícita de stopwords. O TF-IDF manteve as palavras
> funcionais e controlou termos pouco informativos por `max_df`, `min_df`, IDF e
> limite de vocabulário. Em reclamações, palavras como *not*, *no*, *without* e
> *never* podem inverter o sentido; removê-las com uma lista genérica poderia
> destruir sinal preditivo. Os modelos MiniLM também receberam o texto sem essa
> remoção, preservando a entrada esperada pelo modelo pré-treinado.

### O que foi implementado

- O protótipo exploratório inicial utilizou `TfidfVectorizer` sem o argumento
  `stop_words`.
- As etapas S3 a S7 construíram `TfidfVectorizer` sem `stop_words`; portanto,
  valeu o padrão `stop_words=None` do scikit-learn.
- V2 e V3 reutilizaram a representação word/char congelada, também sem remoção
  explícita de stopwords.
- MiniLM recebeu o texto preservando essas palavras. Não houve pré-processamento
  que removesse tokens por pertencerem a uma lista linguística genérica.

### O que fizemos no lugar

O vocabulário esparso foi controlado estatisticamente:

- `min_df=2` remove termos que aparecem em menos de dois documentos;
- `max_df=0.98` remove termos presentes em mais de 98% dos documentos;
- `max_features=40000` limita o ramo de palavras;
- `max_features=60000` limita o ramo de caracteres;
- TF-IDF reduz naturalmente o peso de termos muito frequentes;
- unigramas e bigramas de palavras preservam contexto local;
- n-grams `char_wb` de 3 a 5 caracteres oferecem robustez a variações de
  escrita e vocabulário.

`max_df` pode excluir alguns termos extremamente frequentes, mas isso é um
filtro baseado no corpus, não uma lista semântica de stopwords.

### Por que a decisão é adequada ao domínio

Reclamações financeiras contêm negação, obrigação, condição e temporalidade.
Uma lista genérica de inglês pode remover justamente palavras que diferenciam
frases como “the company reported the debt” e “the company did **not** report
the debt”. O classificador linear pode aprender que determinados termos são
pouco úteis, enquanto a remoção prévia é irreversível.

Para MiniLM, retirar stopwords também criaria uma distribuição de entrada
diferente daquela usada no pré-treinamento. Manter a frase natural preserva a
estrutura necessária à representação contextual.

### Limitação declarada

Não executamos um ablation test “com versus sem stopwords”. Portanto, a defesa
correta não é afirmar que provamos que removê-las seria pior. A afirmação
suportada é:

1. não houve remoção explícita;
2. a escolha preservou palavras potencialmente decisivas no domínio;
3. a frequência foi controlada por mecanismos próprios do TF-IDF;
4. o candidato selecionado foi validado temporalmente sob essa configuração;
5. introduzir stopwords agora criaria um novo candidato e quebraria a identidade
   do pacote que seguirá para confirmação.

### Se o entrevistador perguntar “vocês testariam?”

Sim, mas apenas como experimento exploratório separado. A comparação precisaria
ser pré-especificada, usar as mesmas divisões temporais e não alterar o candidato
já congelado. O resultado não poderia ser usado retroativamente para reescrever
a conclusão do V3-E3.

## Por que mantivemos gates difíceis mesmo esperando uma possível reprovação?

### Resposta curta para entrevista

> Porque o objetivo de um teste confirmatório não é fazer o modelo passar; é
> medir honestamente se o ganho sobrevive ao futuro. Congelamos cinco gates
> antes de pontuar a janela de 2026, inclusive precisão crítica mínima de 0,40.
> A expectativa de falha não muda a regra. Mudá-la para acomodar o resultado
> destruiria justamente a evidência que o projeto pretende produzir.

### Os cinco gates V3-C

1. macro-F1 pelo menos 0,69;
2. F1 crítico pelo menos 0,2715;
3. precisão crítica pelo menos 0,20, preservada para comparação histórica;
4. ganho pareado de F1 crítico sobre o S7 estritamente positivo;
5. precisão crítica do sistema V3 pelo menos 0,40, a exigência adicional V3.

Os gates 3 e 5 parecem redundantes porque usam a mesma métrica, mas cumprem
papéis diferentes. O piso 0,20 mantém a comparação exata com S8 e V2.1-C; o
piso 0,40 determina a barra de segurança realmente exigida do V3. Para o status
`CONFIRMED`, valem os cinco simultaneamente.

### O que a aprovação não autorizou

O protocolo metodológico foi aprovado em 2026-08-24, mas a execução não. A
janela registrada termina em 2026-12-31 e o snapshot atual alcança somente
2026-07-23. O projeto aguardará uma fotografia adquirida a partir de
2027-02-01, congelará sua identidade e pedirá aprovação antes da única abertura.
Isso evita transformar uma janela parcial conveniente numa confirmação.
