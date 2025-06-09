import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
import os


class CreditDataGenerator:
    """
    Generates synthetic credit scoring data for FHE demonstration.
    Features include income, age, debt ratio, credit history length, etc.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = None

    def generate_data(
        self, n_samples: int = 10000, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generate synthetic credit scoring dataset.

        Returns:
            X_train, X_test, y_train, y_test
        """
        np.random.seed(self.random_state)

        # Generate base features
        age = np.random.normal(40, 12, n_samples)
        age = np.clip(age, 18, 80)

        # Income (correlated with age, with some noise)
        income = 30000 + age * 800 + np.random.normal(0, 15000, n_samples)
        income = np.clip(income, 15000, 200000)

        # Credit history length (correlated with age)
        credit_history = np.maximum(0, age - 18 - np.random.exponential(2, n_samples))
        credit_history = np.clip(credit_history, 0, 50)

        # Number of credit accounts
        num_accounts = np.random.poisson(3, n_samples) + 1
        num_accounts = np.clip(num_accounts, 1, 15)

        # Debt to income ratio
        debt_ratio = np.random.beta(2, 5, n_samples) * 0.8

        # Monthly debt payments
        monthly_debt = income * debt_ratio / 12

        # Employment status (1 = employed, 0 = unemployed)
        employment_prob = 0.9 - (age > 65) * 0.3  # Lower employment after 65
        employed = np.random.binomial(1, employment_prob, n_samples)

        # Education level (0-4: High school, Some college, Bachelor, Master, PhD)
        education = np.random.choice(
            [0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.25, 0.15, 0.05]
        )

        # Number of late payments in last 2 years
        late_payments = np.random.poisson(1.5, n_samples)
        late_payments = np.clip(late_payments, 0, 10)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "age": age,
                "income": income,
                "credit_history_length": credit_history,
                "num_accounts": num_accounts,
                "debt_to_income_ratio": debt_ratio,
                "monthly_debt": monthly_debt,
                "employed": employed,
                "education_level": education,
                "late_payments": late_payments,
            }
        )

        # Generate target variable (default probability)
        # Higher risk factors: high debt ratio, low income, unemployment, many late payments
        risk_score = (
            -0.3 * np.log(income / 50000)  # Lower income = higher risk
            + 2.0 * debt_ratio  # High debt ratio = higher risk
            + -0.5 * employed  # Unemployment = higher risk
            + 0.1 * late_payments  # Late payments = higher risk
            + -0.05 * credit_history  # Short credit history = higher risk
            + -0.1 * education  # Lower education = slightly higher risk
            + np.random.normal(0, 0.5, n_samples)  # Random noise
        )

        # Convert to probability using sigmoid
        default_prob = 1 / (1 + np.exp(-risk_score))

        # Generate binary target (1 = default, 0 = no default)
        y = np.random.binomial(1, default_prob, n_samples)

        # Split into train/test
        split_idx = int(n_samples * (1 - test_size))

        X_train = data.iloc[:split_idx].copy()
        X_test = data.iloc[split_idx:].copy()
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        print(f"Generated {n_samples} samples:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Default rate: {y.mean():.3f}")

        return X_train, X_test, y_train, y_test

    def preprocess_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for ML model training.
        Standardizes numerical features.
        """
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def save_scaler(self, filepath: str):
        """Save the fitted scaler for later use in FHE inference."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, filepath: str):
        """Load a previously fitted scaler."""
        with open(filepath, "rb") as f:
            self.scaler = pickle.load(f)

    def get_sample_for_inference(
        self, X_test: pd.DataFrame, idx: int = 0
    ) -> np.ndarray:
        """
        Get a single sample for FHE inference testing.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call preprocess_data first.")

        sample = X_test.iloc[idx : idx + 1]
        sample_scaled = self.scaler.transform(sample)
        return sample_scaled[0]


if __name__ == "__main__":
    # Demo usage
    generator = CreditDataGenerator()
    X_train, X_test, y_train, y_test = generator.generate_data(n_samples=5000)
    X_train_scaled, X_test_scaled = generator.preprocess_data(X_train, X_test)

    print("\nFeature statistics:")
    print(X_train.describe())

    print(f"\nTraining data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
