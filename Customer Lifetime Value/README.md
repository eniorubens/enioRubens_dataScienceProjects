# Customer Lifetime Value Prediction & Customer Prioritization

## Overview

This project builds a probabilistic Customer Lifetime Value (CLTV) model using transactional data to estimate future customer value and support business decision-making.

Instead of relying on historical averages, the model predicts:
- how often a customer will purchase again
- how much they are expected to spend
- their probability of remaining active

The goal is to move from descriptive analytics to **forward-looking revenue prediction**.

---

## Business Problem

Companies often treat all customers equally, but not all customers generate the same long-term value.

This project answers:
> Which customers are most valuable in the future, and where should retention efforts be prioritized?

---

## Methodology

### 1. Data Preparation
- Transactional dataset (Online Retail)
- Removed cancellations and invalid transactions
- Created TotalPrice = Quantity × UnitPrice

---

### 2. RFM Feature Engineering

Customer behavior summarized into:
- **Recency** → time since last purchase
- **Frequency** → number of repeat purchases
- **Monetary Value** → average spend per transaction

---

### 3. Probabilistic Modeling

#### BG/NBD Model
Predicts:
- expected number of future transactions
- probability that a customer is still active

#### Gamma-Gamma Model
Predicts:
- expected monetary value per transaction

---

### 4. Customer Lifetime Value (CLTV)

CLTV is calculated by combining:
- expected purchase frequency
- expected monetary value

This results in **future revenue estimation per customer**.

---

### 5. Customer Segmentation

Customers are segmented into:
- Low Value
- Mid Value
- High Value
- Top Value

---

### 6. Decision Framework

Customers are prioritized based on:
- CLTV (future value)
- Probability of being active (implicit churn risk)

| Segment | Action |
|--------|--------|
| High CLTV + Low Probability Alive | Immediate retention action |
| High CLTV + High Probability Alive | Retain & grow |
| Low CLTV + High Probability Alive | Low-cost engagement |
| Low CLTV + Low Probability Alive | Low priority |

---

## Key Insights

- A small percentage of customers drives most of the future revenue
- Customer inactivity (recency) is a strong indicator of churn risk
- CLTV provides a more meaningful prioritization than raw revenue
- Not all customers should receive the same retention investment

---

## Business Impact

This model enables:
- prioritization of high-value customers
- targeted retention strategies
- more efficient marketing spend
- proactive revenue protection

> The real value is not measuring customers — but deciding how to act on them.

---

## Tech Stack

- Python
- Pandas / NumPy
- Lifetimes (BG/NBD & Gamma-Gamma)
- Matplotlib / Seaborn

---

## Author

Enio Rubens  
Data Science & Analytics
