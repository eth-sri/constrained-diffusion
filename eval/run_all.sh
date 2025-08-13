# This script runs all evaluation scripts reported in the paper in sequence.
echo "This script will run all evaluation scripts reported in the paper in sequence."
echo "Outputs are stored in results/ and will be skipped if present. Partial results will be completed."
echo "You can safely re-run this script to complete the results."
set -ex
echo "Downloading models..."
PYTHONPATH=. python3 eval/download_models.py
echo "Running evaluation scripts..."
PYTHONPATH=. python3 eval/dllm/run_dllm.py
PYTHONPATH=. python3 eval/fim/run_fim.py
PYTHONPATH=. python3 eval/dllm/run_dllm_step_ablation.py
echo "Completed!"
echo "Outputs are stored in results/ and were be skipped if present. Partial results were completed."
echo "You can safely re-run this script to complete the results."
