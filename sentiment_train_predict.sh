# # 1. run training and prediction on BERT models
#model_name=(bert-base-cased bert-base-uncased bert-large-cased bert-large-uncased bert-base-multilingual-cased bert-base-multilingual-uncased roberta-base roberta-large)
#export CUDA_VISIBLE_DEVICES=1
#
#for i in {0..7}
#do
#
#echo "START: the pretrained_model is ${model_name[$i]}"
#
export DATA_DIR=./data
export MODEL_NAME=vinai/bertweet-large
export OUTPUT_DIR=./models/senti_${MODEL_NAME}
export CUDA_VISIBLE_DEVICES=1,2,3

#   --use_fast_tokenizer False \
#  --evaluation_strategy epoch \
python run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --cache_dir ./cache/ \
  --train_file $DATA_DIR/train.csv \
  --validation_file $DATA_DIR/dev.csv \
  --test_file $DATA_DIR/test.csv \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --do_predict \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir

#python run_glue.py \
#  --model_name_or_path $OUTPUT_DIR \
#  --cache_dir ./cache/ \
#  --train_file $DATA_DIR/train.csv \
#  --validation_file $DATA_DIR/dev.csv \
#  --test_file $DATA_DIR/test.csv \
#  --do_eval \
#  --do_predict \
#  --max_seq_length 128 \
#  --per_device_eval_batch_size 32 \
#  --output_dir $OUTPUT_DIR \
#  --overwrite_output_dir
#
#echo "END: the pretrained_model is ${model_name[$i]}"
#done

# # 2. multi-task with 1 head
#export CUDA_VISIBLE_DEVICES=1
#export DATA_DIR=../data
#export MODEL_NAME=roberta-large
#export OUTPUT_DIR=../models/senti_v_ei_$MODEL_NAME
#
#python run_glue.py \
#  --model_name_or_path $MODEL_NAME \
#  --train_file $DATA_DIR/EI_data_clean.csv \
#  --validation_file $DATA_DIR/dev_df_clean.csv \
#  --test_file $DATA_DIR/test_df_clean.csv \
#  --do_train \
#  --do_eval \
#  --evaluation_strategy epoch \
#  --do_predict \
#  --max_seq_length 64 \
#  --per_device_train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 20 \
#  --output_dir $OUTPUT_DIR \
#  --overwrite_output_dir
#
#python run_glue.py \
#  --model_name_or_path $OUTPUT_DIR \
#  --train_file $DATA_DIR/train_df_clean.csv \
#  --validation_file $DATA_DIR/dev_df_clean.csv \
#  --test_file $DATA_DIR/test_df_clean.csv \
#  --do_train \
#  --do_eval \
#  --evaluation_strategy epoch \
#  --do_predict \
#  --max_seq_length 64 \
#  --per_device_train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 20 \
#  --output_dir $OUTPUT_DIR \
#  --overwrite_output_dir

# 3. mulit-tasks with 2 heads
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export DATA_DIR=../data
#export MODEL_NAME=roberta-large
#export OUTPUT_DIR=../models/senti_v_ei_${MODEL_NAME}_heads-1
#
#python run_glue_modified.py \
#  --model_name_or_path $MODEL_NAME \
#  --train_file $DATA_DIR/EI_data_clean.csv \
#  --validation_file $DATA_DIR/dev_df_clean.csv \
#  --test_file $DATA_DIR/test_df_clean.csv \
#  --do_train \
#  --do_eval \
#  --evaluation_strategy epoch \
#  --do_predict \
#  --max_seq_length 64 \
#  --per_device_train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 20 \
#  --output_dir $OUTPUT_DIR \
#  --overwrite_output_dir
#
#python run_glue_modified.py \
#  --model_name_or_path $OUTPUT_DIR \
#  --train_file $DATA_DIR/train_df_clean.csv \
#  --validation_file $DATA_DIR/dev_df_clean.csv \
#  --test_file $DATA_DIR/test_df_clean.csv \
#  --do_train \
#  --do_eval \
#  --evaluation_strategy epoch \
#  --do_predict \
#  --max_seq_length 64 \
#  --per_device_train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 20 \
#  --output_dir $OUTPUT_DIR \
#  --overwrite_output_dir