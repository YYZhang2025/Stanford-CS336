#!/bin/bash 


MODEL_NAME="Qwen/Qwen2.5-Math-1.5B"
DATA_PATH="./data/gsm8k/test.jsonl"
PROMPT_PATH="./prompts/r1_zero.prompt"


python cs336_alignment/evaluate.py \
--model_name $MODEL_NAME \
--data_path $DATA_PATH \
--prompt_path $PROMPT_PATH \
--temperature 1.0 \
--top_p 1.0 \
--max_tokens 1024