# Case

We are currently looking to hire a data scientist for our Data Science team here
at LeoVegas. The ideal candidate should possess strong problem-solving skills,
along with expertise in machine learning and coding. This test has been sent to
you to assess your skills and knowledge in these areas.

Best of luck!

## Introduction

Customer churn is a significant concern for all customer-oriented businesses,
including gambling operators. To improve retention, we need to identify players
likely to churn so the CRM team can proactively engage them with relevant
offers. This approach can significantly reduce churn rates, improve customer
loyalty, and increase revenue and customer base stability.

Specifically, this model aims to provide the CRM team with a list of players
likely to churn, enabling targeted campaigns (or other retention actions). This
will help LeoVegas reduce churn rates and increase player lifetime value.

Imagine presenting this project to a product manager focused on business
outcomes. How would you explain your work?

## Task

You have a dataset containing player, daily, and vertical aggregations for
approximately 12,500 customers who made their first deposit between within a
2.5 year priod. This dataset includes various features to help understand
customer behavior and predict churn.

You must define churn and justify your choice. For example, you could define a
churned player as one who hasn't placed a bet in the last 7, 14, or 30 days, or
use a different definition. Justify your choice and its business implications.

Then, illustrate your workflow for creating a predictive model suitable for
sharing with a stakeholder.  Use any tools and packages you deem appropriate.

You are *not* required to build everything from scratch.  Feel free to use any
appropriate open-source packages.

## Data

The dataset contains 12,500 customers described by the following features:

* `player_key`: A unique customer identifier
* `birth_year`: The customer's birth year
* `date`: The date of the aggregated metrics below
* `gaming_turnover_sum`: The sum of bets placed on live casino and casino games
* `gaming_turnover_num`: The number of bets placed on live casino and casino
  games
* `gaming_NGR`: The net gaming revenue (NGR) from live casino and casino games
  (i.e., how much the individual lost)
* `betting_turnover_sum`: The sum of bets placed on sports betting
* `betting_turnover_num`: The number of bets placed on sports betting
* `betting_NGR`: The net gaming revenue (NGR) from sports betting (i.e., how
  much the individual lost)
* `deposit_sum`: The sum of deposits from the customer's bank account to their
  gambling account
* `deposit_num`: The number of deposits from the customer's bank account to
  their gambling account
* `withdrawal_sum`: The sum of withdrawals from the customer's gambling account
  to their bank account
* `withdrawal_num`: The number of withdrawals from the customer's gambling
  account to their bank account
* `login_num`: The number of logins made

**Example values of the data:**

| player_key                     | birth_year | date       | gaming_turnover_sum | gaming_turnover_num | gaming_NGR | betting_turnover_sum | betting_turnover_num | betting_NGR | deposit_sum | deposit_num | withdrawal_sum | withdrawal_num | login_num |
| ------------------------------ | ---------- | ---------- | ------------------- | ------------------- | ---------- | ------------------- | ------------------- | ---------- | ------------ | ------------ | --------------- | --------------- | ---------- |
| -2930472881471393003          | 2000       | 2021-05-03 | 245.99              | 11                  | 58.04      | 7.09                | 4                  | 7.0        | 65.0         | 2            | 0.0             | 0              | 4         |
| 3019988275617763302          | 1990       | 2020-10-31 | 284.21              | 304                 | 100.41     | 8.7                 | 8                  | 9.0        | 83.0         | 2            | 0.0             | 0              | 4         |
| 3626642277166270101          | 1959       | 2022-09-30 | 62.52               | 294                 | -18.93     | 2.06                | 2                  | 2.0        | 29.0         | 2            | 45.0            | 2              | 6         |

The data is derived from real customers but has been transformed to protect
their privacy.

## Grading

This assignment assesses your understanding of key data science principles, not
model performance.  Submissions will be evaluated based on problem-solving,
machine learning principles, design decisions, and coding style.  Limit
additional work to written text.


Specifically, we will assess:

* **A clear definition of your churn target** and its justification.
* **Exploratory Data Analysis (EDA):** Your exploration of data distributions,
  missing values, and relationships.
* **A well-reasoned feature selection** and understanding of their predictive
  value.  Include any engineered features.
* **Clear and documented code** with comments and explanations.
* **A well-justified choice of model(s)** and evaluation metrics.
* **An understanding of model limitations** and potential improvements.
* **An understanding of the business implications** and consequences of
  incorrect predictions.
* **Clear and concise presentation** of your work, suitable for a stakeholder.

Allocate approximately a few hours to this assignment. Make justified trade-offs
to complete within a reasonable timeframe.  A perfectly ideal solution is not
expected; functionality and adequacy within the time constraint are paramount.

