import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

sns.set_style("darkgrid")

class ChurnPredictor:
    def __init__(self, churn_window, test_size):
        self.churn_window = churn_window
        self.test_size = test_size
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()

    def _get_last_bet_dates(self, df):
        # Identifying players who have placed bets
        has_bet = (df["gaming_turnover_num"] > 0) | (df["betting_turnover_num"] > 0)
        # Getting the last bet date for each player
        return df[has_bet].groupby("player_key")["date"].max()

    def _calculate_churn(self, df):
        # Maximum date in the dataset
        max_date = df["date"].max()
        # Getting the last bet dates for each player
        last_bet_dates = self._get_last_bet_dates(df)
        # Getting the last seen dates for each player
        last_seen_dates = df.groupby("player_key")["date"].max()

        # DataFrame with last bet and last seen dates
        last_activity = pd.DataFrame(
            {"last_bet": last_bet_dates, "last_seen": last_seen_dates}
        )
        # Final date of activity for each player
        # (last bet date if available, last seen date otherwise)
        last_activity["final_date"] = last_activity["last_bet"].fillna(
            last_activity["last_seen"]
        )
        # Calculate the number of days inactive
        last_activity["days_inactive"] = (
            max_date - last_activity["final_date"]
        ).dt.days

        # Determine if a player has churned based on inactivity
        # (days inactive greater than or equal to churn window)
        return (last_activity["days_inactive"] >= self.churn_window).astype(int)

    def _engineer_features(self, df):
        # Create a copy of the DataFrame to engineer features
        features = df.copy()
        # avoid division by zero
        safe_div = lambda x, y: x / y.replace(0, 1)

        # Financial metrics
        features["net_position"] = features["gaming_NGR"].fillna(0) + features[
            "betting_NGR"
        ].fillna(0)
        features["deposit_withdrawal_ratio"] = safe_div(
            features["withdrawal_sum"].fillna(0), features["deposit_sum"].fillna(0)
        )

        # Betting behavior
        features["avg_bet_size"] = safe_div(
            features["gaming_turnover_sum"].fillna(0)
            + features["betting_turnover_sum"].fillna(0),
            features["gaming_turnover_num"].fillna(0)
            + features["betting_turnover_num"].fillna(0),
        )
        total_turnover = features["gaming_turnover_sum"].fillna(0) + features[
            "betting_turnover_sum"
        ].fillna(0)
        features["gaming_ratio"] = safe_div(
            features["gaming_turnover_sum"].fillna(0), total_turnover
        )

        # Activity patterns
        features["deposit_frequency"] = safe_div(
            features["deposit_num"].fillna(0), features["login_num"].fillna(0)
        )
        features["withdrawal_ratio"] = safe_div(
            features["withdrawal_sum"].fillna(0), features["deposit_sum"].fillna(0)
        )
        features["avg_gaming_bet"] = safe_div(
            features["gaming_turnover_sum"].fillna(0),
            features["gaming_turnover_num"].fillna(0),
        )
        features["avg_betting_bet"] = safe_div(
            features["betting_turnover_sum"].fillna(0),
            features["betting_turnover_num"].fillna(0),
        )

        # Select numerical features for modeling
        # (features with float64 or int64 data types)
        self.feature_names = features.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()
        return features[self.feature_names]

    def create_visualizations(self, df: pd.DataFrame, y_pred_proba: np.ndarray):
        sns.set_style()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        df_grouped = df.groupby('date').agg({
            'gaming_turnover_sum': 'mean',
            'betting_turnover_sum': 'mean'
        }).rolling(7).mean()
        plt.plot(df_grouped.index, df_grouped['gaming_turnover_sum'], label='Gaming')
        plt.plot(df_grouped.index, df_grouped['betting_turnover_sum'], label='Betting')
        plt.title('Average Daily Turnover')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        login_pattern = df.groupby('date')['login_num'].mean().rolling(7).mean()
        plt.plot(login_pattern.index, login_pattern.values)
        plt.title('Average Daily Logins')
        
        plt.subplot(2, 2, 3)
        money_flow = df.groupby('date').agg({
            'deposit_sum': 'sum',
            'withdrawal_sum': 'sum'
        }).rolling(7).mean()
        plt.plot(money_flow.index, money_flow['deposit_sum'], label='Deposits')
        plt.plot(money_flow.index, money_flow['withdrawal_sum'], label='Withdrawals')
        plt.title('Money Flow Patterns')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        sns.histplot(y_pred_proba, bins=50)
        plt.axvline(0.5, color='r', linestyle='--')
        plt.title('Churn Probability Distribution')
        
        plt.tight_layout()
        plt.savefig('assets/activity_patterns.png')
        plt.close()
        
    
    def _plot_threshold_metrics(self, y_true, y_pred_proba):
        # Generate a range of thresholds
        thresholds = np.linspace(0, 1, 100)
        metrics = []

        # Calculate precision, recall, and f1 score for each threshold
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics.append(
                {
                    "threshold": threshold,
                    "precision": precision_score(y_true, y_pred, zero_division=1),
                    "recall": recall_score(y_true, y_pred, zero_division=1),
                    "f1": f1_score(y_true, y_pred, zero_division=1),
                }
            )

        metrics_df = pd.DataFrame(metrics)

        # Plot the metrics by threshold
        plt.figure(figsize=(10, 6))
        for metric in ["precision", "recall", "f1"]:
            plt.plot(
                metrics_df["threshold"], metrics_df[metric], label=metric.capitalize()
            )
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Metrics by Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig("assets/threshold_metrics.png")
        plt.close()

        return metrics_df

    def _plot_distributions(self, y_true, y_pred_proba):
        # Plot the distribution of predicted probabilities for each class
        plt.figure(figsize=(10, 6))
        plt.hist(
            y_pred_proba[y_true == 0],
            bins=50,
            alpha=0.5,
            label="Non-churned",
            density=True,
        )
        plt.hist(
            y_pred_proba[y_true == 1], bins=50, alpha=0.5, label="Churned", density=True
        )
        plt.xlabel("Predicted Probability of Churn")
        plt.ylabel("Density")
        plt.title("Score Distribution by Class")
        plt.legend()
        plt.savefig("assets/score_distribution.png")
        plt.close()

    def _evaluate_model(self, y_true, y_pred_proba):
        # Plot threshold metrics and find the optimal threshold
        threshold_metrics = self._plot_threshold_metrics(y_true, y_pred_proba)
        optimal_threshold = threshold_metrics.loc[
            threshold_metrics["f1"].idxmax(), "threshold"
        ]
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=1),
            "recall": recall_score(y_true, y_pred, zero_division=1),
            "f1": f1_score(y_true, y_pred, zero_division=1),
        }

        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        metrics["roc_auc"] = auc(fpr, tpr)
        metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)
        metrics["optimal_threshold"] = optimal_threshold

        # Plot ROC and PR curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("assets/performance_curves.png")
        plt.close()

        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix (threshold={optimal_threshold:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("assets/confusion_matrix.png")
        plt.close()

        # Plot score distributions
        self._plot_distributions(y_true, y_pred_proba)

        # Print metrics
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

        return metrics

    def _plot_feature_importance(self, importance_df):
        # Plot the top 10 features influencing churn
        plt.figure(figsize=(12, 6))
        sns.barplot(x="importance", y="feature", data=importance_df.head(10))
        plt.title("Top 10 Features Influencing Churn")
        plt.tight_layout()
        plt.savefig("assets/feature_importance.png")
        plt.close()

    def train_and_evaluate(self, data_path):
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])

        # Calculate churn status
        y = self._calculate_churn(df)

        # Get latest snapshot and features
        latest_data = df.sort_values("date").groupby("player_key").last()
        X = self._engineer_features(latest_data)

        # Align features and target
        y = y[X.index]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter tuning
        print("Starting hyperparameter tuning...")
        param_grid = {
            "learning_rate": [0.01, 0.05],
            "depth": [4, 6],
            "l2_leaf_reg": [2, 3],
            "bootstrap_type": ["Bayesian", "Bernoulli"],
            "grow_policy": ["SymmetricTree"],
        }

        base_model = cb.CatBoostClassifier(
            verbose=False, thread_count=-1, random_seed=42
        )
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=2,
        )

        # Fit the model using grid search
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_

        print("\nBest parameters:", grid_search.best_params_)
        print("Best CV F1 score:", grid_search.best_score_)

        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate model
        metrics = self._evaluate_model(y_test, y_pred_proba)

        # Feature importance
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Plot feature importance
        self._plot_feature_importance(importance_df)

        # Create visualizations
        self.create_visualizations(df, y_pred_proba)

        # Save artifacts
        joblib.dump(self.model, "model/catboost_model.joblib")
        importance_df.to_csv("assets/feature_importance.csv", index=False)

        return {
            "importance": importance_df,
            "predictions": y_pred_proba,
            "metrics": metrics,
        }


def main():
    predictor = ChurnPredictor(churn_window=14, test_size=0.20)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    try:
        # Train and evaluate the model
        results = predictor.train_and_evaluate("data.csv")
        print("\nTop 5 Important Features:")
        print(results["importance"].head())
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
