#!/usr/bin/env python3
"""
FHE Credit Scoring System - Main Pipeline

This is the main entry point for the FHE-based credit scoring system.
It orchestrates the complete pipeline from data generation to FHE inference.
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import our modules
    from data_generator import CreditDataGenerator
    from credit_model import CreditScoringModel
    from fhe_inference import FHECreditInference

    print("All modules imported successfully")
except ImportError as e:
    print(f"Module import failed: {e}")
    print(
        "Please check that all source files are present and dependencies are installed."
    )
    sys.exit(1)


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="FHE Credit Scoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --quick           # Quick demo with fewer samples
  python main.py --samples 10000   # Custom number of samples
  python main.py --bits 6         # Lower quantization for faster FHE
  python main.py --benchmark-runs 5 # Multiple runs for benchmarking
        """,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of credit samples to generate (default: 5000)",
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Number of quantization bits for FHE (default: 8, range: 4-12)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )

    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of FHE inference runs for benchmarking (default: 3)",
    )

    parser.add_argument(
        "--quick", action="store_true", help="Quick demo mode (1000 samples, 6 bits)"
    )

    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models to disk",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def validate_arguments(args):
    """Validate command line arguments"""
    if args.samples < 100:
        print("Error: Number of samples must be at least 100")
        return False

    if args.bits < 4 or args.bits > 12:
        print("Error: Quantization bits must be between 4 and 12")
        return False

    if args.test_size <= 0 or args.test_size >= 1:
        print("Error: Test size must be between 0 and 1")
        return False

    if args.benchmark_runs < 1:
        print("Error: Benchmark runs must be at least 1")
        return False

    return True


def setup_directories():
    """Set up necessary directories"""
    directories = ["models", "data", "logs", "results"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    return True


def step1_generate_data(args):
    """Step 1: Generate synthetic credit data"""
    print("STEP 1: DATA GENERATION")
    print("=" * 50)

    print("Generating synthetic credit data...")
    generator = CreditDataGenerator(random_state=42)

    # Generate data
    X_train, X_test, y_train, y_test = generator.generate_data(
        n_samples=args.samples, test_size=args.test_size
    )

    # Preprocess data
    X_train_scaled, X_test_scaled = generator.preprocess_data(X_train, X_test)

    # Display statistics
    print(f"Dataset size: {len(X_train) + len(X_test)} samples")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")
    print(
        f"Default rate: {(list(y_train) + list(y_test)).count(1) / len(list(y_train) + list(y_test)):.3f}"
    )

    if args.verbose:
        print("\nFeature statistics:")
        print(X_train.describe())

    return {
        "generator": generator,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    }


def step2_train_models(data, args):
    """Step 2: Train traditional and FHE models"""
    print("\nSTEP 2: MODEL TRAINING")
    print("=" * 50)

    model = CreditScoringModel()

    # Train traditional model
    print("Training traditional logistic regression...")
    traditional_results = model.train_traditional_model(
        data["X_train_scaled"], data["y_train"], list(data["X_train"].columns)
    )

    print(f"Traditional model accuracy: {traditional_results['accuracy']:.4f}")

    # Train FHE model
    print("Training FHE-compatible model...")
    try:
        fhe_results = model.train_fhe_model(
            data["X_train_scaled"], data["y_train"], n_bits=args.bits
        )
        print(f"FHE model quantization bits: {args.bits}")
        print("FHE model training completed")
        fhe_available = True
    except Exception as e:
        print(f"FHE model training failed: {e}")
        print("Continuing with traditional model only...")
        fhe_available = False
        fhe_results = None

    # Compare models if both available
    if fhe_available and fhe_results:
        print("\nComparing traditional and FHE models...")
        comparison = model.compare_models(data["X_test_scaled"], data["y_test"])
        print(f"Prediction agreement: {comparison['prediction_agreement']:.4f}")
        print(f"Traditional accuracy: {comparison['traditional_accuracy']:.4f}")
        print(f"FHE accuracy: {comparison['fhe_accuracy']:.4f}")
    else:
        comparison = None

    return {
        "model": model,
        "traditional_results": traditional_results,
        "fhe_results": fhe_results,
        "comparison": comparison,
        "fhe_available": fhe_available,
    }


def step3_fhe_inference(model_results, data, args):
    """Step 3: Demonstrate FHE inference"""
    print("\nSTEP 3: FHE INFERENCE")
    print("=" * 50)

    if not model_results["fhe_available"]:
        print("FHE model not available, skipping FHE inference...")
        return None

    try:
        # Save model temporarily
        model_path = "models/temp_fhe_model.pkl"
        model_results["model"].save_fhe_model(model_path)

        # Initialize FHE inference
        print("Initializing FHE inference engine...")
        fhe_inference = FHECreditInference(model_path)

        # Compile model
        print("Compiling FHE model...")
        sample_input = data["X_test_scaled"][0]
        fhe_inference.compile_model(sample_input)

        # Run benchmark
        print(f"Running FHE inference benchmark ({args.benchmark_runs} samples)...")
        benchmark_results = []

        for i in range(min(args.benchmark_runs, len(data["X_test_scaled"]))):
            sample = data["X_test_scaled"][i]
            print(
                f"Processing sample {i+1}/{min(args.benchmark_runs, len(data['X_test_scaled']))}"
            )

            result, timing = fhe_inference.predict_encrypted_end_to_end(sample)
            benchmark_results.append({"result": result, "timing": timing})

            if args.verbose:
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Total time: {timing['total_time']:.4f}s")

        # Calculate average timing
        avg_timing = {
            "encryption_time": sum(
                r["timing"]["encryption_time"] for r in benchmark_results
            )
            / len(benchmark_results),
            "computation_time": sum(
                r["timing"]["computation_time"] for r in benchmark_results
            )
            / len(benchmark_results),
            "decryption_time": sum(
                r["timing"]["decryption_time"] for r in benchmark_results
            )
            / len(benchmark_results),
            "total_time": sum(r["timing"]["total_time"] for r in benchmark_results)
            / len(benchmark_results),
        }

        print("\nFHE Performance Summary:")
        print(f"Average encryption time: {avg_timing['encryption_time']:.4f}s")
        print(f"Average computation time: {avg_timing['computation_time']:.4f}s")
        print(f"Average decryption time: {avg_timing['decryption_time']:.4f}s")
        print(f"Average total time: {avg_timing['total_time']:.4f}s")

        return {
            "fhe_inference": fhe_inference,
            "benchmark_results": benchmark_results,
            "avg_timing": avg_timing,
        }

    except Exception as e:
        print(f"FHE inference failed: {e}")
        return None


def step4_loan_scenarios(model_results, fhe_results, data):
    """Step 4: Demonstrate loan application scenarios"""
    print("\nSTEP 4: LOAN APPLICATION SCENARIOS")
    print("=" * 50)

    # Create some sample loan applications
    loan_applications = [
        {
            "name": "Young Professional",
            "age": 28,
            "income": 55000,
            "credit_history_length": 4,
            "num_accounts": 3,
            "debt_to_income_ratio": 0.3,
            "monthly_debt": 1375,
            "employed": 1,
            "education_level": 2,
            "late_payments": 1,
        },
        {
            "name": "Experienced Manager",
            "age": 42,
            "income": 85000,
            "credit_history_length": 18,
            "num_accounts": 6,
            "debt_to_income_ratio": 0.25,
            "monthly_debt": 1770,
            "employed": 1,
            "education_level": 3,
            "late_payments": 0,
        },
        {
            "name": "Recent Graduate",
            "age": 24,
            "income": 38000,
            "credit_history_length": 1,
            "num_accounts": 2,
            "debt_to_income_ratio": 0.4,
            "monthly_debt": 1267,
            "employed": 1,
            "education_level": 2,
            "late_payments": 2,
        },
    ]

    print("Processing loan applications...")

    for i, app in enumerate(loan_applications):
        print(f"\nApplication {i+1}: {app['name']}")
        print(f"Age: {app['age']}, Income: ${app['income']:,}")
        print(f"Credit History: {app['credit_history_length']} years")
        print(f"Debt-to-Income: {app['debt_to_income_ratio']:.1%}")

        # Convert to feature vector
        features = data["generator"].scaler.transform(
            [
                [
                    app["age"],
                    app["income"],
                    app["credit_history_length"],
                    app["num_accounts"],
                    app["debt_to_income_ratio"],
                    app["monthly_debt"],
                    app["employed"],
                    app["education_level"],
                    app["late_payments"],
                ]
            ]
        )

        # Traditional prediction
        trad_pred = model_results["model"].traditional_model.predict(features)[0]
        trad_prob = model_results["model"].traditional_model.predict_proba(features)[0][
            1
        ]

        print(
            f"Traditional Model - Risk: {trad_prob:.3f}, Decision: {'DENY' if trad_pred == 1 else 'APPROVE'}"
        )

        # FHE prediction if available
        if fhe_results and model_results["fhe_available"]:
            try:
                fhe_result, fhe_timing = fhe_results[
                    "fhe_inference"
                ].predict_encrypted_end_to_end(features[0])
                print(
                    f"FHE Model - Risk: {fhe_result['confidence']:.3f}, Decision: {'DENY' if fhe_result['prediction'] == 1 else 'APPROVE'}"
                )
                print(f"FHE Processing Time: {fhe_timing['total_time']:.3f}s")
            except Exception as e:
                print(f"FHE prediction failed: {e}")

    return loan_applications


def save_results(all_results, args):
    """Save results to files"""
    if not args.save_models:
        return

    print("\nSaving results...")

    # Save model if available
    if all_results["model_results"]["fhe_available"]:
        model_path = "models/fhe_credit_model.pkl"
        all_results["model_results"]["model"].save_fhe_model(model_path)
        print(f"FHE model saved to {model_path}")

    # Save traditional model
    traditional_path = "models/traditional_credit_model.pkl"
    all_results["model_results"]["model"].save_traditional_model(traditional_path)
    print(f"Traditional model saved to {traditional_path}")


def main():
    """Main pipeline orchestrator"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        args.samples = 1000
        args.bits = 6
        args.benchmark_runs = 2
        print("Quick mode enabled: 1000 samples, 6 bits quantization")

    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)

    # Setup
    print("FHE CREDIT SCORING SYSTEM")
    print("=" * 50)
    print(f"Samples: {args.samples}")
    print(f"Quantization bits: {args.bits}")
    print(f"Benchmark runs: {args.benchmark_runs}")
    print()

    setup_directories()

    # Record start time
    start_time = time.time()

    try:
        # Run pipeline steps
        data_results = step1_generate_data(args)
        model_results = step2_train_models(data_results, args)
        fhe_results = step3_fhe_inference(model_results, data_results, args)
        loan_results = step4_loan_scenarios(model_results, fhe_results, data_results)

        # Combine all results
        all_results = {
            "data_results": data_results,
            "model_results": model_results,
            "fhe_results": fhe_results,
            "loan_results": loan_results,
            "args": args,
        }

        # Save results
        save_results(all_results, args)

        # Calculate total time
        total_time = time.time() - start_time

        print(f"\nPipeline completed successfully in {total_time:.2f} seconds!")

        return all_results

    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        temp_files = ["models/temp_fhe_model.pkl"]
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()


if __name__ == "__main__":
    print("Results saved to models/ directory")
    print("Check the generated visualizations and model files")
    results = main()
