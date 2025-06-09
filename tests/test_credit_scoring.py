#!/usr/bin/env python3
"""
Unit tests for FHE Credit Scoring System

Tests cover data generation, model training, FHE inference, and end-to-end scenarios.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import CreditDataGenerator
from credit_model import CreditScoringModel
from fhe_inference import FHECreditInference, FHECreditScenario

# Suppress warnings during testing
warnings.filterwarnings("ignore")


class TestCreditDataGenerator(unittest.TestCase):
    """Test cases for CreditDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = CreditDataGenerator(random_state=42)

    def test_data_generation(self):
        """Test synthetic data generation."""
        X_train, X_test, y_train, y_test = self.generator.generate_data(
            n_samples=1000, test_size=0.2
        )

        # Check shapes
        self.assertEqual(len(X_train), 800)
        self.assertEqual(len(X_test), 200)
        self.assertEqual(len(y_train), 800)
        self.assertEqual(len(y_test), 200)

        # Check feature count
        self.assertEqual(X_train.shape[1], 9)  # 9 features

        # Check data types
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, np.ndarray)

        # Check target values are binary
        self.assertTrue(np.all(np.isin(y_train, [0, 1])))
        self.assertTrue(np.all(np.isin(y_test, [0, 1])))

    def test_data_preprocessing(self):
        """Test data preprocessing and scaling."""
        X_train, X_test, y_train, y_test = self.generator.generate_data(n_samples=500)
        X_train_scaled, X_test_scaled = self.generator.preprocess_data(X_train, X_test)

        # Check shapes preserved
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)

        # Check scaling (approximately zero mean and unit variance for training data)
        self.assertTrue(np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10))

    def test_scaler_persistence(self):
        """Test scaler saving and loading."""
        X_train, X_test, _, _ = self.generator.generate_data(n_samples=500)
        X_train_scaled, _ = self.generator.preprocess_data(X_train, X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            scaler_path = tmp.name

        try:
            # Save scaler
            self.generator.save_scaler(scaler_path)
            self.assertTrue(os.path.exists(scaler_path))

            # Create new generator and load scaler
            new_generator = CreditDataGenerator()
            new_generator.load_scaler(scaler_path)

            # Test that scaling produces same result
            X_test_scaled_new = new_generator.scaler.transform(X_test)
            X_test_scaled_orig = self.generator.scaler.transform(X_test)

            np.testing.assert_array_almost_equal(X_test_scaled_new, X_test_scaled_orig)

        finally:
            if os.path.exists(scaler_path):
                os.unlink(scaler_path)


class TestCreditScoringModel(unittest.TestCase):
    """Test cases for CreditScoringModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = CreditDataGenerator(random_state=42)
        X_train, X_test, y_train, y_test = self.generator.generate_data(n_samples=1000)
        self.X_train_scaled, self.X_test_scaled = self.generator.preprocess_data(
            X_train, X_test
        )
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = list(X_train.columns)
        self.model = CreditScoringModel(random_state=42)

    def test_traditional_model_training(self):
        """Test traditional logistic regression training."""
        metrics = self.model.train_traditional_model(
            self.X_train_scaled, self.y_train, self.feature_names
        )

        # Check that model is trained
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.traditional_model)

        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("auc_roc", metrics)
        self.assertGreater(metrics["accuracy"], 0.5)  # Better than random
        self.assertGreater(metrics["auc_roc"], 0.5)

    def test_fhe_model_training(self):
        """Test FHE model training."""
        metrics = self.model.train_fhe_model(
            self.X_train_scaled, self.y_train, n_bits=8
        )

        # Check that FHE model is trained
        self.assertIsNotNone(self.model.fhe_model)

        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("auc_roc", metrics)
        self.assertGreater(metrics["accuracy"], 0.5)
        self.assertGreater(metrics["auc_roc"], 0.5)

    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Train both models
        self.model.train_traditional_model(
            self.X_train_scaled, self.y_train, self.feature_names
        )
        self.model.train_fhe_model(self.X_train_scaled, self.y_train)

        # Compare models
        comparison = self.model.compare_models(self.X_test_scaled, self.y_test)

        # Check comparison structure
        self.assertIn("traditional_metrics", comparison)
        self.assertIn("fhe_metrics", comparison)
        self.assertIn("accuracy_difference", comparison)
        self.assertIn("prediction_agreement", comparison)

        # Check agreement is reasonable
        self.assertGreater(comparison["prediction_agreement"], 0.7)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model.train_traditional_model(
            self.X_train_scaled, self.y_train, self.feature_names
        )

        importance_df = self.model.get_feature_importance("traditional")

        # Check structure
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), len(self.feature_names))
        self.assertIn("feature", importance_df.columns)
        self.assertIn("coefficient", importance_df.columns)
        self.assertIn("abs_coefficient", importance_df.columns)

    def test_single_sample_prediction(self):
        """Test single sample prediction."""
        self.model.train_traditional_model(
            self.X_train_scaled, self.y_train, self.feature_names
        )

        sample = self.X_test_scaled[0]
        result = self.model.predict_single_sample(sample, model_type="traditional")

        # Check result structure
        self.assertIn("prediction", result)
        self.assertIn("default_probability", result)
        self.assertIn("credit_score", result)
        self.assertIn("risk_level", result)

        # Check value ranges
        self.assertIn(result["prediction"], [0, 1])
        self.assertGreaterEqual(result["default_probability"], 0)
        self.assertLessEqual(result["default_probability"], 1)
        self.assertGreaterEqual(result["credit_score"], 300)
        self.assertLessEqual(result["credit_score"], 850)


class TestFHEInference(unittest.TestCase):
    """Test cases for FHE inference functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate small dataset for testing
        self.generator = CreditDataGenerator(random_state=42)
        X_train, X_test, y_train, y_test = self.generator.generate_data(n_samples=500)
        self.X_train_scaled, self.X_test_scaled = self.generator.preprocess_data(
            X_train, X_test
        )
        self.y_train = y_train
        self.y_test = y_test

        # Train FHE model
        self.model = CreditScoringModel(random_state=42)
        self.model.train_fhe_model(self.X_train_scaled, self.y_train, n_bits=8)

        # Save model temporarily
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_fhe_model.pkl")
        import joblib

        joblib.dump(self.model.fhe_model, self.model_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fhe_model_loading(self):
        """Test FHE model loading."""
        fhe_inference = FHECreditInference(self.model_path)
        self.assertIsNotNone(fhe_inference.model)

    def test_fhe_compilation(self):
        """Test FHE model compilation."""
        fhe_inference = FHECreditInference(self.model_path)

        # Compile with sample data
        sample = self.X_test_scaled[0]
        fhe_inference.compile_model(sample)

        self.assertTrue(fhe_inference.is_compiled)
        self.assertIsNotNone(fhe_inference.fhe_circuit)
        self.assertIsNotNone(fhe_inference.client)
        self.assertIsNotNone(fhe_inference.server)

        fhe_inference.cleanup()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline with small dataset."""
        # Step 1: Generate data
        generator = CreditDataGenerator(random_state=42)
        X_train, X_test, y_train, y_test = generator.generate_data(n_samples=500)
        X_train_scaled, X_test_scaled = generator.preprocess_data(X_train, X_test)

        # Step 2: Train models
        model = CreditScoringModel(random_state=42)
        model.train_traditional_model(X_train_scaled, y_train, list(X_train.columns))
        model.train_fhe_model(X_train_scaled, y_train, n_bits=8)

        # Step 3: Compare models
        comparison = model.compare_models(X_test_scaled, y_test)

        # Basic checks
        self.assertGreater(comparison["prediction_agreement"], 0.5)
        self.assertIsInstance(comparison["traditional_metrics"]["accuracy"], float)
        self.assertIsInstance(comparison["fhe_metrics"]["accuracy"], float)

    def test_scenario_simulation(self):
        """Test loan application scenario simulation."""
        # Set up minimal system
        generator = CreditDataGenerator(random_state=42)
        X_train, X_test, y_train, y_test = generator.generate_data(n_samples=500)
        X_train_scaled, X_test_scaled = generator.preprocess_data(X_train, X_test)

        model = CreditScoringModel(random_state=42)
        model.train_fhe_model(X_train_scaled, y_train, n_bits=8)

        # Save model temporarily
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "test_model.pkl")

        try:
            import joblib

            joblib.dump(model.fhe_model, model_path)

            # Set up FHE inference
            fhe_inference = FHECreditInference(model_path)
            fhe_inference.compile_model(X_test_scaled[0])

            # Create scenario
            scenario = FHECreditScenario(fhe_inference)

            # Test applicant data (using feature order from dataset)
            feature_names = list(X_train.columns)
            applicant_scaled_dict = {
                name: val for name, val in zip(feature_names, X_test_scaled[0])
            }

            # Simulate application
            decision = scenario.simulate_bank_loan_application(applicant_scaled_dict)

            # Check decision structure
            self.assertIn("approved", decision)
            self.assertIn("credit_score", decision)
            self.assertIn("risk_level", decision)
            self.assertIn("processing_time", decision)
            self.assertIn("recommendation", decision)

            # Check value types and ranges
            self.assertIsInstance(decision["approved"], bool)
            self.assertIsInstance(decision["credit_score"], int)
            self.assertGreaterEqual(decision["credit_score"], 300)
            self.assertLessEqual(decision["credit_score"], 850)
            self.assertIn(decision["risk_level"], ["Low", "Medium", "High"])

            fhe_inference.cleanup()

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""

    def test_small_dataset(self):
        """Test with minimum viable dataset size."""
        generator = CreditDataGenerator(random_state=42)
        X_train, X_test, y_train, y_test = generator.generate_data(n_samples=100)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

    def test_feature_ranges(self):
        """Test that generated features are within expected ranges."""
        generator = CreditDataGenerator(random_state=42)
        X_train, X_test, y_train, y_test = generator.generate_data(n_samples=1000)

        # Check age range
        self.assertTrue(X_train["age"].min() >= 18)
        self.assertTrue(X_train["age"].max() <= 80)

        # Check income is positive
        self.assertTrue(X_train["income"].min() > 0)

        # Check debt ratio is between 0 and 1
        self.assertTrue(X_train["debt_to_income_ratio"].min() >= 0)
        self.assertTrue(X_train["debt_to_income_ratio"].max() <= 1)

        # Check employment is binary
        self.assertTrue(set(X_train["employed"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCreditDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestCreditScoringModel))
    suite.addTests(loader.loadTestsFromTestCase(TestFHEInference))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)
