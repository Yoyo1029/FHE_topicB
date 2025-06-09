#!/usr/bin/env python3
"""
FHE Credit Scoring System - Automatic Demo Runner

This script automatically detects your Python environment and
runs the most appropriate demo mode.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def check_environment():
    """Check Python version and dependencies"""
    version = sys.version_info

    print("Checking your environment...")
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3:
        return "unsupported"

    if version.minor >= 13:
        return "demo"  # Python 3.13+ - demo mode only
    elif 8 <= version.minor <= 12:
        return "full"  # Python 3.8-3.12 - can use full FHE
    else:
        return "demo"  # Python < 3.8 - demo mode only


def check_concrete_availability():
    """Check if Concrete-ML is available"""
    try:
        import concrete.ml

        return True
    except ImportError:
        return False


def get_available_modes():
    """Get available execution modes"""
    modes = [
        {
            "key": "1",
            "name": "Demo Mode (All Python versions)",
            "command": ["python", "demo_credit_scoring.py"],
            "description": "Simulates FHE concepts, works everywhere",
        },
        {
            "key": "2",
            "name": "Quick Test",
            "command": ["python", "QUICK_TEST.py"],
            "description": "Fast functionality check",
        },
        {
            "key": "3",
            "name": "Setup Environment",
            "command": ["python", "setup.py"],
            "description": "Install dependencies and configure system",
        },
    ]

    # Add full FHE mode if available
    env_status = check_environment()
    if env_status == "full" and check_concrete_availability():
        modes.insert(
            1,
            {
                "key": "full",
                "name": "Full FHE Mode (Concrete-ML)",
                "command": ["python", "src/main.py"],
                "description": "Real FHE implementation with Concrete-ML",
            },
        )

    return modes


def show_menu():
    """Show execution options menu"""
    print("\nFHE Credit Scoring System")
    print("Privacy-Preserving Machine Learning Demo")
    print("=" * 50)

    env_status = check_environment()
    concrete_available = check_concrete_availability()

    print(f"Environment Status: {env_status}")
    print(f"Concrete-ML Available: {'Yes' if concrete_available else 'No'}")

    modes = get_available_modes()

    print("\nAvailable Options:")
    for mode in modes:
        print(f"{mode['key']}. {mode['name']}")
        print(f"   {mode['description']}")

    return modes


def run_mode(mode):
    """Run selected mode"""
    print(f"\nStarting: {mode['name']}")
    print("=" * 50)

    try:
        result = subprocess.run(mode["command"], cwd=os.getcwd(), text=True)

        if result.returncode == 0:
            print("\nDemo completed successfully!")
        else:
            print(f"\nDemo exited with code {result.returncode}")

        return result.returncode

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running demo: {e}")
        return 1


def auto_select_mode(modes):
    """Automatically select the best mode"""
    env_status = check_environment()
    concrete_available = check_concrete_availability()

    # Preference order
    if env_status == "full" and concrete_available:
        # Try to find full FHE mode
        for mode in modes:
            if "Full FHE" in mode["name"]:
                print("RECOMMENDED: Full FHE mode with actual Concrete-ML")
                return mode

    # Default to demo mode
    print("RECOMMENDED: Demo mode (Python 3.13+ detected)")
    return modes[0]  # Demo mode


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FHE Credit Scoring Demo Runner")
    parser.add_argument(
        "--auto", action="store_true", help="Automatically select best mode"
    )
    parser.add_argument(
        "--mode", type=str, help="Directly specify mode: demo, quick, setup, full"
    )

    args = parser.parse_args()

    modes = get_available_modes()

    if args.mode:
        # Direct mode selection
        mode_map = {m["key"]: m for m in modes}
        if args.mode in mode_map:
            selected_mode = mode_map[args.mode]
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
    elif args.auto:
        # Auto selection
        selected_mode = auto_select_mode(modes)
    else:
        # Interactive selection
        show_menu()
        choice = input("\nSelect option (1-3): ").strip()

        mode_map = {m["key"]: m for m in modes}
        if choice in mode_map:
            selected_mode = mode_map[choice]
        else:
            print("Invalid selection")
            return 1

    # Run selected mode
    result = run_mode(selected_mode)

    print("\nThank you for trying the FHE Credit Scoring System!")
    print("Privacy-preserving machine learning made practical!")

    return result


if __name__ == "__main__":
    sys.exit(main())
