# Otimização de Campanhas de Marketing e Analytics de Retenção

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics/notebooks/Marketing_Campaign_Optimization_and_Retention_Analytics_PT.ipynb)

**Leitura:** [Relatório executivo · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) · [Relatório em inglês](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/en.html)

Pipeline completo de Marketing Analytics com análises descritiva, inferencial,
preditiva e prescritiva aplicadas à otimização de campanhas, retenção de usuários
e geração de recomendações operacionais.

---

## 💡 Filosofia do projeto

**Camada de governança:** o [relatório executivo](https://eniorubens.github.io/enioRubens_dataScienceProjects/Marketing_Campaign_Optimization_and_Retention_Analytics/) conecta os achados analíticos às limitações de negócio explícitas.

**Produção orientada a contratos:** nos projetos com superfície de serving, APIs FastAPI/Flask expõem contratos de entrada e saída validados por testes. Nos demais, contratos de dados, artefatos, partições seladas e gates de avaliação controlam o que pode ser concluído, reproduzido e publicado.

**Superfície de engenharia:** análise orientada por notebooks com outputs CRM/Excel; este projeto não expõe pacote reutilizável `src/`, API de serving ou suíte automatizada de testes.

**Pode sustentar:** exploração descritiva, inferencial, preditiva e prescritiva de decisões de campanha e retenção.

**Não sustenta:** serviço produtivo de scoring, impacto causal de campanha ou decisões automáticas sem engenharia e validação adicionais.

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

## Principais resultados

- Identificação de diferenças estatisticamente significativas entre variantes A/B
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
