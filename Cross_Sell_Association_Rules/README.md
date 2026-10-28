# Identifying Cross-Selling Opportunities in Retail Using Association Rules

**Read:** [Relatório executivo · Português](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) · [English report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/en.html)

## 🎯 Scope and evidence

**Can support:** exploratory market-basket analysis and candidate cross-sell hypotheses.

**Does not support:** causal incrementality, guaranteed conversion uplift, or automatic offer deployment.

**Engineering surface:** reproducible notebook and dataset analysis; this project does not expose a modular `src/` package, serving API or automated test suite.

**Governance layer:** [Executive report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Cross_Sell_Association_Rules/) presents the rules as behavioral associations, not intervention evidence.

## Overview
This project applies Market Basket Analysis to identify product associations in retail transactional data.

The main objective is to distinguish between frequently purchased items and meaningful associations supported by lift and Zhang filters, enabling more reliable decision-making in cross-selling strategies.

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
- Lift is essential to identify meaningful relationships between products
- Many frequent combinations are driven by item popularity rather than true association
- Relevant rules emerge when combining support, confidence, and lift

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
Cross_Sell_Association_Rules/
│
├── notebooks/
│   ├── Cross_Sell_Association_Rules_PT.ipynb            # Portuguese analysis notebook
│   └── Cross_Sell_Association_Rules_EN.ipynb            # English analysis notebook
├── dataset/Groceries_dataset.csv                        # Input dataset
└── README.md                                            # Project documentation
```
## Conclusion

This project demonstrates how association rules can be used to extract actionable insights from retail data.

By combining statistical metrics such as support, confidence, and lift, it is possible to differentiate between simple item popularity and meaningful product associations, supporting more effective business decisions.


## Reproducibility note

Both notebooks use the same dataset path relative to the `notebooks/` directory and the same analytical thresholds. Their saved outputs were checked for numerical consistency; a fresh execution still requires a working environment with NumPy, pandas, mlxtend, SciPy, Matplotlib and Seaborn.
