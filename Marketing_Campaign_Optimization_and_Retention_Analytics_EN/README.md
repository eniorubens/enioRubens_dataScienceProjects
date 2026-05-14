# Marketing Campaign Optimization and Retention Analytics

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Marketing_Campaign_Optimization_and_Retention_Analytics_EN/Marketing_Campaign_Optimization_and_Retention_Analytics_EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Otimiza%C3%A7%C3%A3o_de_campanhas_de_marketing_e_an%C3%A1lise_de_reten%C3%A7%C3%A3o/Marketing%20Campaign%20Optimization%20and%20Retention%20Analytics.ipynb)

Complete Marketing Analytics pipeline with descriptive, inferential, predictive, and prescriptive analysis applied to campaign optimization, user retention, and operational recommendation generation.

---

## Objective

This project implements an end-to-end analytics solution for digital marketing campaigns, integrating:

- exploratory analysis;
- conversion funnel analysis;
- A/B testing;
- predictive retention modeling;
- bias diagnosis in model errors;
- ROI simulation;
- expected value calculation per user;
- automated operational recommendation generation.

The project was conceptually inspired by the paper:

> *Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*  
> Guha, Echagarruga & Tian (2021)

but expanded into a modern data science pipeline applied to the marketing analytics context.

---

## Main Features

### Descriptive Analysis
- Overall conversion rate
- Retention rate
- Channel performance
- Personalization impact
- Conversion and retention by language

### A/B Testing
- Z-test for difference between proportions
- Statistical interpretation of uplift
- Inferential validation of variants

### Machine Learning
- Supervised pipeline for retention prediction
- Automated preprocessing
- Categorical variable encoding
- Evaluation using:
  - ROC AUC
  - Recall
  - Accuracy
  - Precision

### Error Diagnosis
- False positive analysis
- Over-representation of groups in model errors
- Comparison between dataset composition and error composition
- Diagnosis of potential operational biases

### Financial Simulation
- Estimated ROI by channel
- Expected profit
- Expected value per user
- Operational prioritization based on financial impact

### Prescriptive Recommendation
Automatic classification of users into actions such as:

- Prioritize and Scale
- Maintain and Optimize
- Detailed Analysis
- Fix Language
- Review Pricing
- Low Priority

### Export
- Automated multi-sheet Excel export
- Consolidation of key analytical results

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- OpenPyXL

---

## Analytical Structure

The pipeline was structured into four analytical layers:

### 1. Descriptive
Understanding campaign and user behavior.

### 2. Inferential
Statistical validation of observed differences through A/B testing.

### 3. Predictive
Modeling user retention probability.

### 4. Prescriptive
Generation of operational recommendations based on expected value and ROI.

---

## Key Results

- Identification of statistically significant differences between A/B variants
- Better performance in campaigns using the correct language
- Identification of segments over-represented in false positives
- ROI simulation showing strong variation across channels
- Automated prioritization of users with higher expected value

---

## Academic Reference

GUHA, P.; ECHAGARRUGA, C.; TIAN, E. Q.  
*Optimising Marketing Strategies by Customer Segments and Lifetime Values, with A/B Testing*.  
Applied Marketing Analytics, v. 7, n. 2, p. 144–153, 2021.

---

## Author

Enio Rubens  
Data Scientist | Marketing Analytics | Machine Learning | Predictive Modeling
