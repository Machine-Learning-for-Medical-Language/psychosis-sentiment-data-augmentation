# psychosis-sentiment-data-augmentation

1. Build the container for sentiment classification: <code>docker build -t psych-sentiment .</code>

2. Start it up, pointing it to a model: <code>docker run -v /home/tmill/Projects/psychosis-sentiment-data-augmentation/models/seed_18/:/model -p 8000:8000 --rm -it --entrypoint bash psych-sentiment</code>, or to run in the background: <code>docker run -v /home/tmill/Projects/psychosis-sentiment-data-augmentation/sentiment/models/seed_18/:/model -p 8001:8000 --rm -d psych-sentiment</code>

3. Inside the container, to process a directory: <code>python3 run_glue.py --model_name_or_path /model/model --task_name psy-se --data_dir data/ --output_dir output/ --do_predict</code>

Results are 0, 1, 2, which map to original labels of 0, 1, 9, which in turn map to Positive, Neutral, Negative

Next steps: 
 * Try to wrap in a FastAPI script
 * Try to get it to use CUDA since nlp-gpu has it but right now it's processing on CPU.
 * See if I get different results if I pass in the same text with different domains using the psy-se task. there is a psy-se-domain task but I don't think it's active code.

