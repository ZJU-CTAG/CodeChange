#!/bin/bash

CURRENT_DIR=`pwd`
NCCL_DEBUG=INFO
GPU_ID=0
PRETRAINED_MODEL_DIR="$CURRENT_DIR/models/pre-training/Gen"
MODEL_PATH="$CURRENT_DIR/models/pre-training/Gen/pytorch_model.bin"
FINETUNED_MODEL_PATH="$CURRENT_DIR/models/fine-tuning/CodeReview/pytorch_model.bin"
EVAL_FLAG=false

usage() {
  echo "Usage: ${0} [-g] [-e]" 1>&2
  exit 1 
}

while getopts ":g:e:" opt; do
    case $opt in
        g)  GPU_ID="$OPTARG"
            ;;
        e)  MODEL_PATH="$OPTARG"
            EVAL_FLAG=true
          ;;
        \?)
          # if invalid option is provided, print error message and exit
          echo "Invalid option: -$OPTARG" >&2
          exit 1
          ;;
        :)
        # if -e flag is provided without a parameter, set eval variable to true
        EVAL_FLAG=true
        MODEL_PATH=$FINETUNED_MODEL_PATH
        ;;
    esac
done

function finetune() {
    SCRIPT_PATH="src/fine_tuning/finetune_msg_gen.py"
    if [[ $EVAL_FLAG == false ]]; then
      python $SCRIPT_PATH \
          --do_train \
          --do_test \
          --train_filename /LLM4CC/Dataset/CodeReview/train.jsonl \
          --dev_filename /LLM4CC/Dataset/CodeReview/valid.jsonl \
          --test_filename /LLM4CC/Dataset/CodeReview/test_cache.jsonl \
          --model_type codet5_CC \
          --warmup_steps 500 \
          --learning_rate 3e-4 \
          --tokenizer_name codet5-base \
          --model_name_or_path "codet5-base" \
          --load_model_path $MODEL_PATH \
          --output_dir ${CURRENT_DIR}/outputs/models/fine-tuning/CodeReview \
          --always_save_model \
          --train_batch_size 32 \
          --gradient_accumulation_steps 4 \
          --eval_batch_size 32 \
          --max_source_length 512 \
          --max_target_length 128 \
          --gpu_id ${GPU_ID} \
          --save_steps 3000 \
          --log_steps 5 \
          --train_steps 150000 \
          --evaluate_sample_size -1
    else
      python $SCRIPT_PATH \
          --do_test \
          --train_filename /Dataset/CodeReview/train_cache_diff.pkl \
          --dev_filename ${CURRENT_DIR}/Dataset/fine-tuning/CodeReview/valid.jsonl \
          --test_filename /Dataset/CodeReview/test_diff_dataset_cache.pkl \
          --model_type codet5_CC \
          --warmup_steps 500 \
          --learning_rate 3e-4 \
          --tokenizer_name /CodeT5/CodeT5/models/codet5-base \
          --model_name_or_path "/CodeT5/models/codet5-base" \
          --load_model_path $MODEL_PATH \
          --output_dir ${CURRENT_DIR}/outputs/models/fine-tuning/CodeReview/${LANG} \
          --always_save_model \
          --train_batch_size 32 \
          --gradient_accumulation_steps 4 \
          --eval_batch_size 32 \
          --max_source_length 512 \
          --max_target_length 128 \
          --gpu_id ${GPU_ID} \
          --save_steps 3000 \
          --log_steps 5 \
          --train_steps 150000 \
          --evaluate_sample_size -1
    fi 
}

finetune;
