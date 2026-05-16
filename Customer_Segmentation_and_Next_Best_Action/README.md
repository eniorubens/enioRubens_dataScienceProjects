# Customer Segmentation and Next Best Action

[🇺🇸 English](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/45c9c834b1d1a922964e1d39f52f8d85c8f94e8a/Customer_Segmentation_and_Next_Best_Action/notebooks/Customer_Segmentation_and_Next_Best_Action-EN.ipynb) | [🇧🇷 Português](https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/45c9c834b1d1a922964e1d39f52f8d85c8f94e8a/Customer_Segmentation_and_Next_Best_Action/notebooks/Customer_Segmentation_and_Next_Best_Action-PT.ipynb)

## Overview

This project develops a complete customer analytics and decision framework using transactional retail data. The objective is not only to segment customers based on purchasing behavior, but also to predict future repurchase probability, simulate marketing incentives, and recommend the most economically advantageous action for each customer profile.

The project evolves from descriptive analytics into predictive and prescriptive analytics, inspired by the concepts presented in the paper *From Predictive to Prescriptive Analytics* by Dimitris Bertsimas and Nathan Kallus.

The final pipeline integrates:

Transactional Data → RFM Features → Customer Segmentation → Repurchase Prediction → Incentive Simulation → Expected Profit Optimization → Next Best Action

---

# Business Problem

Companies frequently apply the same marketing strategy to all customers, despite significant differences in customer value, purchase behavior, and retention potential.

This project addresses the following questions:

* Which customer groups generate the highest business value?
* Which customers are more likely to repurchase?
* Which marketing action maximizes expected economic return?
* How should a limited marketing budget be allocated?

---

# Dataset

Dataset used:

* Online Retail II Dataset

Main variables:

* Invoice
* InvoiceDate
* Quantity
* Price
* Customer ID
* Country

The dataset contains over one million transactional records from a retail business.

---

# Project Architecture

## 1. Data Cleaning and Preparation

The following preprocessing steps were performed:

* Removal of canceled transactions
* Removal of invalid quantities and prices
* Handling missing customer identifiers
* Revenue calculation
* Datetime conversion
* Temporal consistency validation

---

## 2. RFM Feature Engineering

Customer behavior was summarized using RFM metrics:

| Metric    | Description                  |
| --------- | ---------------------------- |
| Recency   | Days since the last purchase |
| Frequency | Number of unique purchases   |
| Monetary  | Total amount spent           |

Additional behavioral features were also created:

* Average ticket
* Total quantity purchased
* Number of unique products
* Country profile

---

## 3. Customer Segmentation with K-Means

K-Means clustering was used to identify customer behavioral groups.

The project includes:

* Elbow Method
* Silhouette Score analysis
* Cluster interpretation
* Revenue concentration analysis

Example customer segments:

* Champions
* Loyal Customers
* High Value at Risk
* Occasional Buyers
* Inactive Customers

---

# Predictive Analytics Layer

## Repurchase Probability Prediction

A supervised learning layer was added to estimate the probability of future repurchase.

Target variable:

* `repurchase_90d`

The target indicates whether a customer made a new purchase within 90 days after the temporal cutoff date.

Models evaluated:

* Logistic Regression
* Probability calibration with `CalibratedClassifierCV`

Evaluation metrics:

* ROC AUC
* Recall
* Precision

This stage transforms the project from descriptive segmentation into predictive analytics.

---

# Prescriptive Analytics Layer

## Marketing Incentive Simulation

Different marketing actions were simulated:

| Action           | Incentive Cost | Simulated Uplift |
| ---------------- | -------------- | ---------------- |
| No Action        | 0              | 0%               |
| Low Incentive    | 2              | 3%               |
| Medium Incentive | 5              | 7%               |
| High Incentive   | 10             | 12%              |

The uplift values are heuristic assumptions for simulation purposes and do not represent causal estimates.

---

## Expected Profit Optimization

For each customer, the framework estimates:

* Expected repurchase probability
* Expected revenue
* Campaign cost
* Expected profit
* Incremental profit

The system then selects the action with the highest expected economic return.

Core concept:

Best Action = argmax(Expected Profit)

---

# Budget-Constrained Optimization

The notebook also introduces business constraints through budget-aware prioritization.

The framework:

* ranks customers by ROI
* allocates incentives under budget limitation
* prioritizes customers with highest expected return

This transforms the project into a practical decision optimization pipeline.

---

# Next Best Action Framework

The final recommendation layer converts analytical results into operational actions.

Examples:

| Segment            | Recommended Action                     |
| ------------------ | -------------------------------------- |
| Champions          | VIP retention and premium offers       |
| High Value at Risk | Immediate reactivation campaigns       |
| Loyal Customers    | Cross-sell and bundle strategies       |
| Occasional Buyers  | Low-cost second-purchase campaigns     |
| Inactive Customers | Low-priority or selective reactivation |

The notebook also exports CRM-ready recommendation files for operational usage.

---

# Key Insights

* Customer value is highly concentrated in specific behavioral groups.
* Purchase recency is a strong indicator of future repurchase probability.
* Predictive models allow the estimation of future customer behavior.
* Prescriptive analytics enables economically optimized marketing decisions.
* Not every customer should receive the same retention investment.
* Marketing budget allocation can be optimized using expected profit.

---

# From Predictive to Prescriptive Analytics

One of the main contributions of this project is the transition from predictive analytics to prescriptive analytics.

Instead of answering only:

> “What is likely to happen?”

the framework also answers:

> “What decision should be taken?”

This aligns the project with modern Decision Science approaches.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# Concepts Applied

* RFM Analysis
* Customer Segmentation
* K-Means Clustering
* Predictive Analytics
* Prescriptive Analytics
* Expected Value Optimization
* ROI Prioritization
* Marketing Analytics
* Decision Science
* Budget-Constrained Optimization

---

# Future Improvements

Possible future evolutions include:

* Uplift Modeling
* Causal Inference
* A/B Testing
* Reinforcement Learning
* Dynamic Budget Optimization
* Real-time Recommendation Systems

---

## Development Notes

Parts of this project were developed with the assistance of AI-based coding and writing tools for code refinement, structuring, documentation, and analytical brainstorming.

All analytical decisions, business interpretations, modeling choices, validation procedures, and final project integration were conducted and reviewed by the author.

Author : Enio Rubens
Data Science & Analytics

```
```
