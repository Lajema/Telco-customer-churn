# 📉 Customer Churn Prediction & Retention Risk App

An end-to-end churn prediction project — from raw data cleaning through model comparison to a deployed Streamlit app — built on the real IBM/Kaggle Telco Customer Churn dataset (7,043 customers).

This isn't just "train a model and deploy it." The value of this project is in the analyst judgment behind it: catching a data quality issue before it broke the pipeline, comparing three models honestly instead of defaulting to the most complex one, and diagnosing a collinearity problem that was quietly distorting the business narrative.

**[https://lajema-telco-customer-churn-app-dznsze.streamlit.app/]**

---

## 🧠 What This Project Actually Does

Given a customer's account and service details, the app returns a churn probability and a clear risk verdict — the kind of tool a retention team could use to decide who to prioritize for outreach, and why.

---

## 📊 The Process

### 1. Data Cleaning
`TotalCharges` loaded as `object` dtype instead of numeric — turned out to be 11 blank-string rows, all belonging to brand-new customers (`tenure == 0`) who hadn't been billed yet. Verified the pattern before fixing it, rather than blindly dropping or filling.

### 2. Exploratory Analysis
Churn is imbalanced (~26.5% positive class). Visual EDA confirmed the two strongest patterns before any model touched the data: **month-to-month contracts** and **short tenure** both associate heavily with churn.

### 3. Collinearity Check
`TotalCharges` looked informative in isolation, but a scatter plot colored by tenure showed it's mechanically close to `tenure × MonthlyCharges` — not independent information. Confirmed by re-running the model comparison with and without it: ROC-AUC moved by less than 0.01 either way. **Dropped it from the feature set** — this cleaned up the interpretability layer significantly (see below).

### 4. Model Comparison — Done Honestly, Not by Default
Three models compared on identical train/test splits, not just picking Random Forest because it's the "powerful" one:

| Model | ROC-AUC | Churn Precision | Churn Recall | Churn F1 |
|---|---|---|---|---|
| Logistic Regression | 0.839 | 0.50 | 0.78 | 0.61 |
| Decision Tree (depth 4) | 0.836 | 0.50 | 0.82 | 0.62 |
| **Random Forest** | 0.836 | **0.53** | 0.77 | **0.63** |

All three are within 0.003 ROC-AUC of each other — the ranking quality is essentially tied. **Random Forest was chosen at the precision/recall level, not the AUC level**: it flags fewer false alarms than the other two while catching a comparable share of real churners, meaning fewer wasted retention offers on customers who were never going to leave.

### 5. Interpretability Layer
Random Forest predicts; Logistic Regression's coefficients explain *why*, converted to odds ratios for a plain-English business narrative:

- **Fiber optic internet** (OR ≈ 2.08) and **month-to-month contracts** (OR ≈ 1.93) are the strongest independent risk factors.
- **Tenure** (OR ≈ 0.47) is the strongest protective factor — risk concentrates hardest in a customer's first year.
- **Electronic check payment** (OR ≈ 1.26) is a smaller but actionable lever — nudging customers to autopay is a low-cost retention move.

The Decision Tree adds a layer the coefficients can't: **combinations**, not just individual effects. The single riskiest customer profile the tree found is *month-to-month contract + fiber optic + short tenure* — a leaf with gini ≈ 0.11, meaning the tree is highly confident about that specific segment, not just guessing. That's the actionable version of the finding: don't blanket-target every fiber customer, target the fiber + month-to-month + new-customer intersection.

### 6. Deployment
Preprocessing (scaling + one-hot encoding) and the classifier are bundled into a single `sklearn.Pipeline`, saved as one `.pkl` artifact. The Streamlit app never touches an encoder directly — it just calls `.predict_proba()` on a raw dataframe, which eliminates train/serve skew risk entirely.

---

## 🛠️ Tech Stack
`Python` · `pandas` · `scikit-learn` · `Streamlit` · `matplotlib` / `seaborn` for EDA

---

## 📂 Project Structure
```
├── Telco-Customer-Churn.csv       # Raw dataset (IBM/Kaggle)
├── churn_model_training.ipynb     # Full analysis: EDA, model comparison, interpretability
├── train_model.py                 # Clean training script — produces churn_model.pkl
├── app.py                         # Streamlit deployment
├── churn_model.pkl                # Trained pipeline (preprocessing + Random Forest)
└── requirements.txt
```

---

## 🚀 Running Locally

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

To retrain from scratch:
```bash
python train_model.py
```

## ☁️ Deploying
Pushed to Streamlit Community Cloud — connect the repo at [share.streamlit.io](https://share.streamlit.io), point it at `app.py`, dependencies build automatically from `requirements.txt`.

---

## 🔑 Key Takeaway

Model selection here wasn't "pick the fanciest algorithm" — three models landed within a point of each other on ROC-AUC, and the real decision came down to what the *business* cares about (fewer wasted retention offers vs. catching every possible churner) at the actual deployment threshold. The strongest single insight, tenure's protective effect, was initially overstated by ~50% due to a collinear feature that looked harmless until it was checked.

*Dataset: [Telco Customer Churn, Kaggle/IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)*
