# Otimização de campanhas de marketing e análise de retenção

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics_EN/Marketing_Campaign_Optimization_and_Retention_Analytics_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Otimiza%C3%A7%C3%A3o_de_campanhas_de_marketing_e_an%C3%A1lise_de_reten%C3%A7%C3%A3o/Marketing%20Campaign%20Optimization%20and%20Retention%20Analytics.ipynb)

Pipeline completa de Marketing Analytics com análise descritiva, inferencial, preditiva e prescritiva aplicada à otimização de campanhas, retenção de usuários e geração de recomendações operacionais.

---

## Objetivo

Este projeto implementa uma solução end-to-end de analytics para campanhas de marketing digital, integrando:

- análise exploratória;
- funil de conversão;
- A/B testing;
- modelagem preditiva de retenção;
- diagnóstico de vieses nos erros do modelo;
- simulação de ROI;
- cálculo de valor esperado por usuário;
- geração automatizada de recomendações operacionais.

O projeto foi inspirado conceitualmente no artigo:

> *Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*  
> Guha, Echagarruga & Tian (2021)

mas expandido para um pipeline moderno de ciência de dados aplicado ao contexto de marketing analytics.

---

## Principais Funcionalidades

### Análise Descritiva
- Taxa global de conversão
- Taxa de retenção
- Performance por canal
- Impacto da personalização
- Conversão e retenção por idioma

### A/B Testing
- Teste Z para diferença entre proporções
- Interpretação estatística do uplift
- Validação inferencial de variantes

### Machine Learning
- Pipeline supervisionada para previsão de retenção
- Pré-processamento automatizado
- Encoding de variáveis categóricas
- Avaliação com:
  - ROC AUC
  - Recall
  - Accuracy
  - Precision

### Diagnóstico de Erros
- Análise de falsos positivos
- Super-representação de grupos nos erros
- Comparação entre composição da base e composição dos erros
- Diagnóstico de possíveis vieses operacionais

### Simulação Financeira
- ROI estimado por canal
- Lucro esperado
- Valor esperado por usuário
- Priorização operacional baseada em impacto financeiro

### Recomendação Prescritiva
Classificação automática de usuários em ações como:

- Priorizar e Escalar
- Manter e Otimizar
- Análise Detalhada
- Corrigir Idioma
- Revisar Precificação
- Baixa Prioridade

### Exportação
- Exportação automática para Excel multi-abas
- Consolidação dos principais resultados analíticos

---

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- OpenPyXL

---

## Estrutura Analítica

O pipeline foi estruturado em quatro camadas analíticas:

### 1. Descritiva
Compreensão do comportamento das campanhas e usuários.

### 2. Inferencial
Validação estatística de diferenças observadas via A/B testing.

### 3. Preditiva
Modelagem da probabilidade de retenção dos usuários.

### 4. Prescritiva
Geração de recomendações operacionais baseadas em valor esperado e ROI.

---

## Principais Resultados

- Identificação de diferenças estatisticamente significativas entre variantes A/B
- Melhor desempenho em campanhas com idioma correto
- Identificação de segmentos super-representados nos falsos positivos
- Simulação de ROI com forte variação entre canais
- Priorização automatizada de usuários com maior valor esperado

---

## Referência Acadêmica

GUHA, P.; ECHAGARRUGA, C.; TIAN, E. Q.  
*Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*.  
Applied Marketing Analytics, v. 7, n. 2, p. 144–153, 2021.

---

## Autor

Enio Rubens  
Data Scientist | Marketing Analytics | Machine Learning | Predictive Modeling
