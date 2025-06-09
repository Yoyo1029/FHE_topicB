import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
import pickle
import os
import joblib
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


class CreditScoringModel:
    """
    Credit scoring model that supports both traditional and FHE inference.
    Uses logistic regression for credit default prediction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.traditional_model = None
        self.fhe_model = None
        self.is_fitted = False
        self.feature_names = None

    def train_traditional_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> dict:
        """
        Train traditional logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features for interpretability

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names

        # Train traditional model
        self.traditional_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight="balanced",  # Handle class imbalance
        )

        self.traditional_model.fit(X_train, y_train)

        # Calculate training metrics
        y_pred = self.traditional_model.predict(X_train)
        y_prob = self.traditional_model.predict_proba(X_train)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_train, y_pred),
            "auc_roc": roc_auc_score(y_train, y_prob),
        }

        print("Traditional Model Training Results:")
        print(f"Training Accuracy: {metrics['accuracy']:.4f}")
        print(f"Training AUC-ROC: {metrics['auc_roc']:.4f}")

        self.is_fitted = True
        return metrics

    def train_fhe_model(
        self, X_train: np.ndarray, y_train: np.ndarray, n_bits: int = 8
    ) -> dict:
        """
        Train FHE-compatible logistic regression model using Concrete-ML.

        Args:
            X_train: Training features
            y_train: Training labels
            n_bits: Number of bits for quantization (affects precision vs speed)

        Returns:
            Dictionary with training metrics
        """
        # Train FHE model
        self.fhe_model = ConcreteLogisticRegression(
            n_bits=n_bits, random_state=self.random_state, max_iter=1000
        )

        self.fhe_model.fit(X_train, y_train)

        # Calculate training metrics
        y_pred = self.fhe_model.predict(X_train)
        y_prob = self.fhe_model.predict_proba(X_train)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_train, y_pred),
            "auc_roc": roc_auc_score(y_train, y_prob),
        }

        print("FHE Model Training Results:")
        print(f"Training Accuracy: {metrics['accuracy']:.4f}")
        print(f"Training AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Quantization bits: {n_bits}")

        return metrics

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray, model_type: str = "both"
    ) -> dict:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            model_type: 'traditional', 'fhe', or 'both'

        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        if model_type in ["traditional", "both"] and self.traditional_model is not None:
            y_pred_trad = self.traditional_model.predict(X_test)
            y_prob_trad = self.traditional_model.predict_proba(X_test)[:, 1]

            results["traditional"] = {
                "accuracy": accuracy_score(y_test, y_pred_trad),
                "auc_roc": roc_auc_score(y_test, y_prob_trad),
                "predictions": y_pred_trad,
                "probabilities": y_prob_trad,
            }

        if model_type in ["fhe", "both"] and self.fhe_model is not None:
            y_pred_fhe = self.fhe_model.predict(X_test)
            y_prob_fhe = self.fhe_model.predict_proba(X_test)[:, 1]

            results["fhe"] = {
                "accuracy": accuracy_score(y_test, y_pred_fhe),
                "auc_roc": roc_auc_score(y_test, y_prob_fhe),
                "predictions": y_pred_fhe,
                "probabilities": y_prob_fhe,
            }

        # Print results
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} Model Test Results:")
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")

        return results

    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Compare traditional and FHE model performance.
        """
        if self.traditional_model is None or self.fhe_model is None:
            raise ValueError("Both models must be trained before comparison")

        results = self.evaluate_model(X_test, y_test, "both")

        # Calculate differences
        acc_diff = results["fhe"]["accuracy"] - results["traditional"]["accuracy"]
        auc_diff = results["fhe"]["auc_roc"] - results["traditional"]["auc_roc"]

        # Calculate prediction agreement
        pred_agreement = np.mean(
            results["traditional"]["predictions"] == results["fhe"]["predictions"]
        )

        comparison = {
            "accuracy_difference": acc_diff,
            "auc_difference": auc_diff,
            "prediction_agreement": pred_agreement,
            "traditional_metrics": results["traditional"],
            "fhe_metrics": results["fhe"],
        }

        print(f"\nModel Comparison:")
        print(f"Accuracy difference (FHE - Traditional): {acc_diff:.4f}")
        print(f"AUC difference (FHE - Traditional): {auc_diff:.4f}")
        print(f"Prediction agreement: {pred_agreement:.4f}")

        return comparison

    def get_feature_importance(self, model_type: str = "traditional") -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        """
        if model_type == "traditional":
            if self.traditional_model is None:
                raise ValueError("Traditional model not trained")
            coefficients = self.traditional_model.coef_[0]
        elif model_type == "fhe":
            if self.fhe_model is None:
                raise ValueError("FHE model not trained")
            coefficients = self.fhe_model.coef_[0]
        else:
            raise ValueError("model_type must be 'traditional' or 'fhe'")

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)

        return importance_df

    def plot_feature_importance(
        self, model_type: str = "traditional", save_path: Optional[str] = None
    ):
        """
        Plot feature importance.
        """
        importance_df = self.get_feature_importance(model_type)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x="abs_coefficient", y="feature")
        plt.title(f"Top 10 Feature Importance ({model_type.upper()} Model)")
        plt.xlabel("Absolute Coefficient Value")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_models(self, traditional_path: str, fhe_path: str):
        """Save both traditional and FHE models."""
        os.makedirs(os.path.dirname(traditional_path), exist_ok=True)
        os.makedirs(os.path.dirname(fhe_path), exist_ok=True)

        if self.traditional_model is not None:
            joblib.dump(self.traditional_model, traditional_path)
            print(f"Traditional model saved to {traditional_path}")

        if self.fhe_model is not None:
            joblib.dump(self.fhe_model, fhe_path)
            print(f"FHE model saved to {fhe_path}")

    def load_models(self, traditional_path: str, fhe_path: str):
        """Load both traditional and FHE models."""
        if os.path.exists(traditional_path):
            self.traditional_model = joblib.load(traditional_path)
            print(f"Traditional model loaded from {traditional_path}")

        if os.path.exists(fhe_path):
            self.fhe_model = joblib.load(fhe_path)
            print(f"FHE model loaded from {fhe_path}")

        self.is_fitted = True

    def predict_single_sample(
        self, sample: np.ndarray, model_type: str = "fhe"
    ) -> dict:
        """
        Predict credit default for a single sample.

        Args:
            sample: Single sample features (1D array)
            model_type: 'traditional' or 'fhe'

        Returns:
            Dictionary with prediction and probability
        """
        if model_type == "traditional":
            if self.traditional_model is None:
                raise ValueError("Traditional model not trained")
            model = self.traditional_model
        elif model_type == "fhe":
            if self.fhe_model is None:
                raise ValueError("FHE model not trained")
            model = self.fhe_model
        else:
            raise ValueError("model_type must be 'traditional' or 'fhe'")

        # Reshape for prediction
        sample = sample.reshape(1, -1)

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0, 1]

        result = {
            "prediction": int(prediction),
            "default_probability": float(probability),
            "credit_score": int(
                (1 - probability) * 850 + 300
            ),  # Convert to credit score scale
            "risk_level": (
                "High"
                if probability > 0.5
                else "Medium" if probability > 0.3 else "Low"
            ),
        }

        return result


if __name__ == "__main__":
    # Demo usage
    from data_generator import CreditDataGenerator

    # Generate data
    generator = CreditDataGenerator()
    X_train, X_test, y_train, y_test = generator.generate_data(n_samples=5000)
    X_train_scaled, X_test_scaled = generator.preprocess_data(X_train, X_test)

    # Train models
    model = CreditScoringModel()
    model.train_traditional_model(X_train_scaled, y_train, list(X_train.columns))
    model.train_fhe_model(X_train_scaled, y_train, n_bits=8)

    # Evaluate and compare
    comparison = model.compare_models(X_test_scaled, y_test)

    # Show feature importance
    print("\nFeature Importance (Traditional Model):")
    print(model.get_feature_importance("traditional").head())
