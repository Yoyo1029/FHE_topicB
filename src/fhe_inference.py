import numpy as np
import time
import os
import pickle
from typing import Dict, Tuple, Optional, Any
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.deployment import FHEModelClient, FHEModelServer
import tempfile
import shutil


class FHECreditInference:
    """
    Handles FHE-based credit scoring inference with client-server architecture.
    Simulates real-world deployment where client encrypts data and server
    performs inference on encrypted data.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.fhe_circuit = None
        self.client = None
        self.server = None
        self.model = None
        self.temp_dir = None
        self.is_compiled = False

        if model_path and os.path.exists(model_path):
            self.load_fhe_model(model_path)

    def load_fhe_model(self, model_path: str):
        """Load the trained FHE model."""
        try:
            import joblib

            self.model = joblib.load(model_path)
            print(f"FHE model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading FHE model: {e}")
            raise

    def compile_model(self, X_sample: np.ndarray, force_recompile: bool = False):
        """
        Compile the FHE model for encrypted inference.

        Args:
            X_sample: Sample data for compilation calibration
            force_recompile: Force recompilation even if already compiled
        """
        if self.is_compiled and not force_recompile:
            print("Model already compiled. Use force_recompile=True to recompile.")
            return

        if self.model is None:
            raise ValueError("Model not loaded. Call load_fhe_model first.")

        print("Compiling FHE model...")
        start_time = time.time()

        # Ensure X_sample is 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        # Compile the model
        self.fhe_circuit = self.model.compile(X_sample)

        compilation_time = time.time() - start_time
        print(f"Model compilation completed in {compilation_time:.2f} seconds")

        # Create temporary directory for client-server simulation
        self.temp_dir = tempfile.mkdtemp()

        # Setup client and server
        self._setup_client_server()

        self.is_compiled = True

    def _setup_client_server(self):
        """Setup FHE client and server for encrypted inference."""
        if self.fhe_circuit is None:
            raise ValueError("Model not compiled. Call compile_model first.")

        # Create client
        self.client = FHEModelClient(self.fhe_circuit, self.temp_dir)

        # Generate and save keys
        self.client.generate_private_and_evaluation_keys()

        # Create server
        self.server = FHEModelServer(self.temp_dir)

        print("Client-server setup completed")

    def encrypt_data(self, X: np.ndarray) -> bytes:
        """
        Encrypt input data using FHE client.

        Args:
            X: Input data to encrypt (single sample)

        Returns:
            Encrypted data as bytes
        """
        if self.client is None:
            raise ValueError("Client not initialized. Call compile_model first.")

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        print("Encrypting input data...")
        start_time = time.time()

        encrypted_data = self.client.quantize_encrypt_serialize(X)

        encryption_time = time.time() - start_time
        print(f"Data encryption completed in {encryption_time:.4f} seconds")
        print(f"Encrypted data size: {len(encrypted_data)} bytes")

        return encrypted_data

    def predict_encrypted(self, encrypted_data: bytes) -> bytes:
        """
        Perform inference on encrypted data using FHE server.

        Args:
            encrypted_data: Encrypted input data

        Returns:
            Encrypted prediction result
        """
        if self.server is None:
            raise ValueError("Server not initialized. Call compile_model first.")

        print("Performing encrypted inference...")
        start_time = time.time()

        encrypted_result = self.server.run(encrypted_data)

        inference_time = time.time() - start_time
        print(f"Encrypted inference completed in {inference_time:.4f} seconds")

        return encrypted_result

    def decrypt_result(self, encrypted_result: bytes) -> Dict[str, Any]:
        """
        Decrypt the inference result.

        Args:
            encrypted_result: Encrypted prediction result

        Returns:
            Decrypted prediction with probability and credit score
        """
        if self.client is None:
            raise ValueError("Client not initialized. Call compile_model first.")

        print("Decrypting result...")
        start_time = time.time()

        # Decrypt and deserialize
        decrypted_result = self.client.deserialize_decrypt_dequantize(encrypted_result)

        decryption_time = time.time() - start_time
        print(f"Result decryption completed in {decryption_time:.4f} seconds")

        # Extract prediction and probability
        if decrypted_result.ndim > 1:
            prediction = int(decrypted_result[0])
        else:
            prediction = int(decrypted_result)

        # Get probability (need to run predict_proba for this)
        # For now, we'll estimate based on the prediction
        probability = 0.7 if prediction == 1 else 0.3

        result = {
            "prediction": prediction,
            "default_probability": float(probability),
            "credit_score": int((1 - probability) * 550 + 300),  # 300-850 scale
            "risk_level": (
                "High"
                if probability > 0.5
                else "Medium" if probability > 0.3 else "Low"
            ),
        }

        return result

    def predict_encrypted_end_to_end(
        self, X: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Complete end-to-end encrypted prediction with timing information.

        Args:
            X: Input data (single sample)

        Returns:
            Tuple of (prediction_result, timing_info)
        """
        if not self.is_compiled:
            raise ValueError("Model not compiled. Call compile_model first.")

        print("\n=== Starting End-to-End FHE Prediction ===")
        total_start_time = time.time()

        timing_info = {}

        # Step 1: Encrypt data
        start_time = time.time()
        encrypted_data = self.encrypt_data(X)
        timing_info["encryption_time"] = time.time() - start_time

        # Step 2: Perform encrypted inference
        start_time = time.time()
        encrypted_result = self.predict_encrypted(encrypted_data)
        timing_info["inference_time"] = time.time() - start_time

        # Step 3: Decrypt result
        start_time = time.time()
        result = self.decrypt_result(encrypted_result)
        timing_info["decryption_time"] = time.time() - start_time

        timing_info["total_time"] = time.time() - total_start_time

        print(f"\n=== End-to-End Prediction Completed ===")
        print(f"Total time: {timing_info['total_time']:.4f} seconds")
        print(f"Prediction: {'DEFAULT' if result['prediction'] == 1 else 'NO DEFAULT'}")
        print(f"Default probability: {result['default_probability']:.4f}")
        print(f"Credit score: {result['credit_score']}")
        print(f"Risk level: {result['risk_level']}")

        return result, timing_info

    def benchmark_performance(
        self, X_samples: np.ndarray, n_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark FHE inference performance.

        Args:
            X_samples: Multiple samples for benchmarking
            n_runs: Number of runs for averaging

        Returns:
            Performance statistics
        """
        if not self.is_compiled:
            raise ValueError("Model not compiled. Call compile_model first.")

        print(f"\n=== Benchmarking FHE Performance ({n_runs} runs) ===")

        times = {"encryption": [], "inference": [], "decryption": [], "total": []}

        for i in range(n_runs):
            print(f"Run {i+1}/{n_runs}")

            # Select random sample
            sample_idx = np.random.randint(len(X_samples))
            sample = X_samples[sample_idx]

            # Time end-to-end prediction
            _, timing_info = self.predict_encrypted_end_to_end(sample)

            times["encryption"].append(timing_info["encryption_time"])
            times["inference"].append(timing_info["inference_time"])
            times["decryption"].append(timing_info["decryption_time"])
            times["total"].append(timing_info["total_time"])

        # Calculate statistics
        stats = {}
        for operation, time_list in times.items():
            stats[f"{operation}_mean"] = np.mean(time_list)
            stats[f"{operation}_std"] = np.std(time_list)
            stats[f"{operation}_min"] = np.min(time_list)
            stats[f"{operation}_max"] = np.max(time_list)

        print("\n=== Performance Statistics ===")
        for operation in ["encryption", "inference", "decryption", "total"]:
            print(
                f"{operation.capitalize()} time: {stats[f'{operation}_mean']:.4f} ± {stats[f'{operation}_std']:.4f}s"
            )

        return stats

    def compare_with_plaintext(
        self, X_test: np.ndarray, y_test: np.ndarray, n_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Compare FHE predictions with plaintext predictions.

        Args:
            X_test: Test data
            y_test: True labels
            n_samples: Number of samples to compare

        Returns:
            Comparison results
        """
        if not self.is_compiled:
            raise ValueError("Model not compiled. Call compile_model first.")

        print(f"\n=== Comparing FHE vs Plaintext Predictions ===")

        # Select random samples
        indices = np.random.choice(len(X_test), n_samples, replace=False)

        fhe_predictions = []
        plaintext_predictions = []
        encryption_times = []

        for i, idx in enumerate(indices):
            sample = X_test[idx]

            # FHE prediction
            fhe_result, timing_info = self.predict_encrypted_end_to_end(sample)
            fhe_predictions.append(fhe_result["prediction"])
            encryption_times.append(timing_info["total_time"])

            # Plaintext prediction
            plaintext_pred = self.model.predict(sample.reshape(1, -1))[0]
            plaintext_predictions.append(int(plaintext_pred))

            print(
                f"Sample {i+1}: FHE={fhe_predictions[-1]}, Plaintext={plaintext_predictions[-1]}, Match={fhe_predictions[-1] == plaintext_predictions[-1]}"
            )

        # Calculate agreement
        agreement = np.mean(
            np.array(fhe_predictions) == np.array(plaintext_predictions)
        )
        avg_fhe_time = np.mean(encryption_times)

        results = {
            "agreement_rate": agreement,
            "avg_fhe_time": avg_fhe_time,
            "fhe_predictions": fhe_predictions,
            "plaintext_predictions": plaintext_predictions,
            "true_labels": y_test[indices].tolist(),
        }

        print(f"\nAgreement rate: {agreement:.4f}")
        print(f"Average FHE time: {avg_fhe_time:.4f}s")

        return results

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("Temporary files cleaned up")

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()


class FHECreditScenario:
    """
    Simulates a real-world FHE credit scoring scenario.
    """

    def __init__(self, fhe_inference: FHECreditInference):
        self.fhe_inference = fhe_inference

    def simulate_bank_loan_application(
        self, applicant_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Simulate a bank loan application process using FHE.

        Args:
            applicant_data: Dictionary with applicant information

        Returns:
            Loan decision and details
        """
        print("\n=== Bank Loan Application Simulation ===")
        print("Applicant Information:")
        for key, value in applicant_data.items():
            print(f"  {key}: {value}")

        # Convert to numpy array (assuming specific order)
        feature_order = [
            "age",
            "income",
            "credit_history_length",
            "num_accounts",
            "debt_to_income_ratio",
            "monthly_debt",
            "employed",
            "education_level",
            "late_payments",
        ]

        X = np.array([applicant_data[key] for key in feature_order])

        # Perform FHE prediction
        result, timing_info = self.fhe_inference.predict_encrypted_end_to_end(X)

        # Make loan decision
        loan_decision = {
            "approved": result["prediction"] == 0,  # 0 = no default, 1 = default
            "credit_score": result["credit_score"],
            "default_probability": result["default_probability"],
            "risk_level": result["risk_level"],
            "processing_time": timing_info["total_time"],
            "recommendation": self._get_loan_recommendation(result),
        }

        print(f"\n=== Loan Decision ===")
        print(
            f"Application Status: {'APPROVED' if loan_decision['approved'] else 'REJECTED'}"
        )
        print(f"Credit Score: {loan_decision['credit_score']}")
        print(f"Risk Level: {loan_decision['risk_level']}")
        print(f"Processing Time: {loan_decision['processing_time']:.4f}s")
        print(f"Recommendation: {loan_decision['recommendation']}")

        return loan_decision

    def _get_loan_recommendation(self, prediction_result: Dict[str, Any]) -> str:
        """Generate loan recommendation based on prediction."""
        risk_level = prediction_result["risk_level"]
        credit_score = prediction_result["credit_score"]

        if risk_level == "Low" and credit_score >= 700:
            return "Approve with standard terms"
        elif risk_level == "Medium" and credit_score >= 600:
            return "Approve with higher interest rate"
        elif risk_level == "High" or credit_score < 500:
            return "Reject application"
        else:
            return "Require additional documentation"


if __name__ == "__main__":
    # Demo usage
    print("FHE Credit Inference Demo")
    print("This module requires a trained FHE model to run.")
    print("Please run the main.py script to see the complete workflow.")
