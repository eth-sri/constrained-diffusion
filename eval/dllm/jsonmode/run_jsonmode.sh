pip install -r requirements.txt
for i in {0..4}
do
  for temp in 0 0.5 1
  do
    echo "Running inference with seed $i"
    # Set the seed for reproducibility
    PYTHONPATH=. python3 "eval/jsonmode/llada_inference_jsonmode.py" --constrained True --trace False --seed $i --temp $temp --output_file "results/jsonmode_outputs_s=${i}_t=${temp}_c.jsonl"
  done
done


