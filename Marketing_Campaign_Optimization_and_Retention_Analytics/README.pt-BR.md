# Otimização de Campanhas de Marketing e Analytics de Retenção

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_PT.ipynb)

**Leitura:** [Relatório executivo · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/en.html)

Pipeline completo de Marketing Analytics com análises descritiva, inferencial,
preditiva e prescritiva aplicadas à otimização de campanhas, retenção de usuários
e geração de recomendações operacionais.

---

## 🎯 Escopo e evidências

**Pode sustentar:** exploração descritiva, inferencial, preditiva e prescritiva de decisões de campanha e retenção.

**Não sustenta:** serviço produtivo de scoring, impacto causal de campanha ou decisões automáticas sem engenharia e validação adicionais.

**Superfície de engenharia:** análise orientada por notebooks mais um pequeno pacote `src/marketing_campaign` (preparação de dados, segmentação RFM, funil, teste A/B e poder, valor do cliente e ROI, modelo de retenção, recomendações) coberto por 54 testes pytest que reproduzem os números do relatório executivo. Os notebooks mantêm o código inline como narrativa da análise; o pacote reimplementa e testa as mesmas etapas. Não há API de serving.

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) conecta os achados analíticos às limitações de negócio explícitas.

## Objetivo

Este projeto implementa uma solução analítica de ponta a ponta para campanhas de
marketing digital, integrando:

- análise exploratória;
- análise do funil de conversão;
- teste A/B;
- modelagem preditiva de retenção;
- diagnóstico de viés nos erros do modelo;
- simulação de ROI;
- cálculo do valor esperado por usuário;
- geração automatizada de recomendações operacionais.

O projeto foi inspirado conceitualmente pelo artigo:

> *Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*
> Guha, Echagarruga & Tian (2021)

mas foi ampliado para um pipeline moderno de data science aplicado ao contexto
de analytics de marketing.

---

## Principais funcionalidades

### Análise descritiva
- Taxa geral de conversão
- Taxa de retenção
- Desempenho por canal
- Impacto da personalização
- Conversão e retenção por idioma

### Teste A/B
- Z-test para diferença entre proporções
- Interpretação estatística do uplift
- Validação inferencial das variantes

### Machine Learning
- Pipeline supervisionado para previsão de retenção
- Pré-processamento automatizado
- Codificação de variáveis categóricas
- Avaliação usando:
  - ROC AUC
  - Recall
  - Accuracy
  - Precision

### Diagnóstico de erros
- Análise de falsos positivos
- Sobre-representação de grupos nos erros do modelo
- Comparação entre a composição do dataset e a composição dos erros
- Diagnóstico de potenciais vieses operacionais

### Simulação financeira
- ROI estimado por canal
- Lucro esperado
- Valor esperado por usuário
- Priorização operacional baseada em impacto financeiro

### Recomendação prescritiva
Classificação automática dos usuários em ações como:

- Priorizar e escalar
- Manter e otimizar
- Análise detalhada
- Corrigir idioma
- Revisar preço
- Baixa prioridade

### Exportação
- Exportação automatizada para Excel com múltiplas planilhas
- Consolidação dos principais resultados analíticos

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- OpenPyXL

---

## Estrutura analítica

O pipeline foi estruturado em quatro camadas analíticas:

### 1. Descritiva
Compreensão do comportamento de campanhas e usuários.

### 2. Inferencial
Validação estatística das diferenças observadas por meio de testes A/B.

### 3. Preditiva
Modelagem da probabilidade de retenção do usuário.

### 4. Prescritiva
Geração de recomendações operacionais com base em valor esperado e ROI.

---

## Código e testes

```text
Marketing_Campaign_Optimization_and_Retention_Analytics/
├── notebooks/                      # Notebooks de análise (EN e PT-BR)
├── dataset/marketing_new.csv       # Dataset de entrada
├── src/marketing_campaign/         # Código de análise reutilizável e documentado
│   ├── config.py                   # Constantes de negócio e ações por segmento
│   ├── data.py                     # Leitura, limpeza e base de usuários estáveis
│   ├── segmentation.py             # Proxies RFM e segmentos K-Means
│   ├── funnel.py                   # Funil e desempenho por canal
│   ├── experiment.py               # Teste Z, tamanho de amostra, poder por Monte Carlo
│   ├── economics.py                # Valor do cliente e ROI simulado
│   ├── retention_model.py          # Regressão logística de retenção
│   ├── recommendations.py          # Valor esperado, ações e exportações
│   └── pipeline.py                 # Executa a análise completa
├── tests/                          # Suíte pytest (54 testes)
└── pyproject.toml                  # Configuração do pacote e dos testes
```

A suíte protege os números do relatório executivo: 10.037 registros brutos, 9.757 exposições limpas, 6.849 usuários estáveis (460 que trocaram de variante excluídos), 703 conversões (10,26%), 463 retidos (65,86%), z = 0,0779 com p = 0,937870, 3.916 usuários por grupo e 80,0% de poder simulado, o ROI incremental por canal, ROC-AUC 0,956 com 2 falsos positivos e a tabela de recomendações de 6.849 linhas.

```bash
cd Marketing_Campaign_Optimization_and_Retention_Analytics
pip install -e ".[dev]"
python -m pytest -q
```

---

## Principais resultados

- Nenhuma diferença estatisticamente significativa entre as variantes A/B (z = 0,0779, p = 0,94), com os dois grupos abaixo da amostra planejada
- Melhor desempenho em campanhas que usam o idioma correto
- Identificação de segmentos sobre-representados nos falsos positivos
- Simulação de ROI mostrando forte variação entre canais
- Priorização automatizada de usuários com maior valor esperado

---

## Referência acadêmica

GUHA, P.; ECHAGARRUGA, C.; TIAN, E. Q.
*Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*.
Applied Marketing Analytics, v. 7, n. 2, p. 144–153, 2021.

---

## Autor

Enio Rubens
Data Scientist | Marketing Analytics | Machine Learning | Predictive Modeling
