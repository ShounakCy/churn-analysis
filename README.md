# Gaming Customer Churn Prediction

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
python churn_analysis.py

```

## A Solution for Customer Retention

This project develops a machine learning solution to predict and prevent customer churn in the gaming industry. By identifying players at risk of leaving, we enable proactive retention efforts that preserve revenue and enhance customer lifetime value.

## Problem Definition and Approach

In the gaming industry, understanding and preventing customer churn is crucial for maintaining a healthy business. I defined churn as 14 days of inactivity in betting behavior, after analysis of player patterns and industry knowledge, payout of salary. This window provides enough time to identify genuine disengagement while ensuring adequate opportunity for successful intervention. This definition tries to balance both sensitivity (catching genuine churn cases) with specificity (avoiding false alarms that could lead to unnecessary interventions).

## Data Analysis and Behavioral Patterns

### Temporal Analysis

![Average Daily Turnover](assets/activity_patterns.png)

The analysis of daily turnover demonstrates distinct player preferences and behavior:

- Gaming activity consistently dominates betting, with 5-10x higher turnover
- Post-2020 stabilization showing mature market dynamics
- Regular seasonal patterns inform our prediction timing
- Betting serves as a complementary engagement channel

Login behavior analysis reveals critical engagement trends:

- Peak engagement of 1.8 daily logins during 2021
- Recent decline to 1.4 daily logins suggests increased churn risk
- Clear weekly patterns inform intervention timing
- Year-over-year comparison showing seasonal effects

Deposit and withdrawal patterns provide early warning signals:

- Consistent growth in transaction volumes from 2020-2023
- Increasing volatility in recent periods
- Weekly patterns could be aligned with salary payments

## Feature Engineering and Model Development

### Feature Importance

![Feature Importance](assets/feature_importance.png)

Through careful analysis of behavioral patterns, I engineered features across multiple dimensions using catboost's inbuilt feature_importances_ method:

1. Financial Indicators (Primary Drivers):
   - Deposit sum (importance: 25.0)
   - Gaming NGR (importance: 20.5)
   - Net position (importance: 17.0)

2. Behavioral Metrics:
   - Login frequency trends
   - Gaming-to-betting ratios
   - Session duration patterns

3. Temporal Features:
   - Weekly volatility measures
   - Seasonal adjustment factors
   - Activity trend indicators

## Model Performance

I selected CatBoost as the primary model based on its proven handling of temporal dependencies and robust performance with behavioral data, I also tested on XGBoost, LightGBM, RandomForest, but Catboost gave sligtly better results:

- Learning rate options: [0.05, 0.1]
- Tree depth options: [4, 6]
- L2 regularization options: [2, 3]
- Bootstrap types: ["Bayesian", "Bernoulli"]
- Used GridSearchCV to choose the best performant parameters

  ### Best parameters

      - **bootstrap_type**: Bernoulli
      - **depth**: 4
      - **grow_policy**: SymmetricTree
      - **l2_leaf_reg**: 2
      - **learning_rate**: 0.05
  - Best CV F1 score: 0.9205422491759044

![ROC and PR Curves](assets/performance_curves.png)

The model demonstrates strong predictive capability:

- ROC AUC: 0.811 (strong discriminative ability)
- PR AUC: 0.944 (strong precision-recall balance)

### Prediction Accuracy

![Confusion Matrix](assets/confusion_matrix.png)

- 1: Churned
- 0: Not Churned
  
At our optimal threshold of 0.55, the model achieves:

- True Positives: 2015 (correctly identified churners)
- True Negatives: 153 (correctly identified active players)
- False Positives: 273 (incorrect predictions)
- False Negatives: 65 (missed churners)

![Metrics by Threshold](assets/threshold_metrics.png)

### Risk Distribution Analysis

![Score Distribution](assets/score_distribution.png)

The distribution of churn risk scores shows:

- High-risk concentration in -1.8-1.0 range
- Low-risk stability below -1.4
- Intervention opportunity in mid-range scores

## Business Impact Analysis

### Revenue Impact

Based on model predictions:

#### True Positives (2015 cases)

- Potential retained revenue: €402,000 (40% intervention success) (assuming each retained customer contributes €200)
- Retention campaign cost: €40,300 (€20 per intervention)
- Net revenue saved: €361,700

#### False Positives (273 cases)

- Wasted campaign costs: €5,460

#### False Negatives (65 cases)

- Missed revenue opportunity: €13,000 (assuming each missed customer contributes €200)

### Intervention Strategy

Our temporal analysis informs optimal intervention timing:

1. Early Week (Monday-Tuesday):
   - Focus on re-engagement campaigns
   - Higher response rates observed
   - Aligned with deposit patterns

2. Pre-Weekend (Thursday-Friday):
   - Retention offer deployment
   - Activity stimulation campaigns
   - Preparation for peak gaming periods

3. Seasonal Adjustments:
   - Enhanced monitoring during low seasons
   - Adjusted thresholds for holiday periods
   - Special campaign timing for major events
