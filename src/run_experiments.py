"""
run_experiments.py
------------------
Automated threshold sweep runner.

Iterates over SCAN_DEPTH_THRESHOLD values, runs main.py for each, then
generates all comparison plots and paper figures when every run succeeds.

Usage:
    python run_experiments.py
"""

import subprocess
import os
import re
import time
import sys

# =================================================================
# ====== CONFIGURATION ======
# =================================================================
# Add or remove thresholds freely
THRESHOLDS = [5, 10, 15, 20, 25]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MAIN_FILE   = os.path.join(SCRIPT_DIR, "main.py")
OUTPUT_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
# =================================================================


def update_main_py_threshold(new_threshold):
    """Overwrites SCAN_DEPTH_THRESHOLD in main.py."""
    if not os.path.exists(MAIN_FILE):
        print(f"Error: {MAIN_FILE} not found.")
        return False

    with open(MAIN_FILE, 'r') as f:
        content = f.read()

    pattern     = r"(SCAN_DEPTH_THRESHOLD\s*=\s*)\d+"
    replacement = rf"\g<1>{new_threshold}"
    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        print("Error: Could not find SCAN_DEPTH_THRESHOLD in main.py")
        return False

    with open(MAIN_FILE, 'w') as f:
        f.write(new_content)

    print(f"Updated main.py: SCAN_DEPTH_THRESHOLD = {new_threshold}")
    return True


def run_experiment(threshold):
    """Runs main.py with the given threshold in batch mode."""
    print(f"\n>>> Experiment: SCAN_DEPTH_THRESHOLD = {threshold}")

    if not update_main_py_threshold(threshold):
        return False

    env = os.environ.copy()
    env["BATCH_MODE"] = "1"

    start_time = time.time()
    try:
        subprocess.run([sys.executable, "-u", MAIN_FILE], env=env, check=True)
        duration = time.time() - start_time
        print(f"Run completed in {duration:.2f}s  SUCCESS.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed (exit code {e.returncode})")
        return False


def main():
    print("=" * 50)
    print("AUTOMATED THRESHOLD EXPERIMENT RUNNER")
    print("=" * 50)
    print(f"Thresholds  : {THRESHOLDS}")
    print(f"Output root : {OUTPUT_ROOT}")

    results = []
    for i, threshold in enumerate(THRESHOLDS):
        print(f"\n--- RUN {i+1}/{len(THRESHOLDS)} ---")
        success = run_experiment(threshold)
        results.append((threshold, success))
        if not success:
            print(f"Stopping: failure at threshold {threshold}.")
            break

    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    for threshold, success in results:
        print(f"  Threshold {threshold:2d}: {'PASSED' if success else 'FAILED'}")
    print("=" * 50)

    all_passed = all(success for _, success in results)
    if not all_passed:
        print("\nSkipping post-processing: not all experiments passed.")
        return

    print("\n" + "=" * 50)
    print("POST-PROCESSING: GENERATING COMPARISONS")
    print("=" * 50)
    try:
        figures_dir = os.path.join(OUTPUT_ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        print("\n>>> Generating Q-Learning Safety Comparison...")
        from compare_safety import generate_comparison as gen_qlearning
        qlearning_csv = os.path.join(figures_dir, "safety_comparison_qlearning.csv")
        gen_qlearning(data_dir=OUTPUT_ROOT, output_path=qlearning_csv, thresholds=THRESHOLDS)

        print(">>> Generating SARSA Safety Comparison...")
        from compare_safety_sarsa import generate_comparison as gen_sarsa
        sarsa_csv = os.path.join(figures_dir, "safety_comparison_sarsa.csv")
        gen_sarsa(data_dir=OUTPUT_ROOT, output_path=sarsa_csv, thresholds=THRESHOLDS)

        print(">>> Generating Algorithm Comparison Plots...")
        from plot_comparison_qlearning_vs_sarsa import plot_comparison
        plot_comparison(qlearning_csv=qlearning_csv, sarsa_csv=sarsa_csv, output_dir=figures_dir)

        print(">>> Generating Paper Figures...")
        from generate_paper_figures import generate_all_paper_figures
        generate_all_paper_figures(output_dir=figures_dir, data_dir=OUTPUT_ROOT, thresholds=THRESHOLDS)

        print("\n" + "=" * 50)
        print(f"POST-PROCESSING COMPLETE  ->  {OUTPUT_ROOT}")
        print("=" * 50)

    except Exception as e:
        print(f"ERROR during post-processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
