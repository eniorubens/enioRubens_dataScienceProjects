# Identifying Cross-Selling Opportunities in Retail Using Association Rules

## Overview
This project applies Market Basket Analysis to identify product associations in retail transactional data.

The objective is to distinguish between frequently purchased items and statistically relevant associations, supporting better decision-making in cross-selling strategies.

---

## Business Problem
Retail datasets often contain highly frequent products, but frequency alone does not imply meaningful association.

This project focuses on identifying product combinations that occur more frequently than expected, avoiding misleading conclusions based solely on item popularity.

---

## Methodology
- Data aggregation by transaction (customer + date)
- One-hot encoding transformation
- Apriori algorithm for frequent itemset mining
- Association rules generation
- Evaluation using:
  - Support
  - Confidence
  - Lift
  - Conviction
  - Zhang’s Metric

---

## Key Insights
- High-frequency items do not necessarily imply strong association
- Lift is essential to identify meaningful relationships
- Many frequent combinations are driven by item popularity rather than true association

---

## Business Applications
- Cross-selling strategies
- Product bundling
- Product placement optimization
- Recommendation systems

---

## Technologies
- Python
- Pandas
- MLxtend
- Matplotlib / Seaborn

---

## Repository Structure
```text
cross_sell_association_rules/
│
├── Identificação de oportunidades de cross-sell.ipynb
├── dataset/Groceries_dataset.csv
└── README.md
```
