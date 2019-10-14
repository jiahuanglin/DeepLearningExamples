DATA_DIR=/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/nvidia/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
RESULTS_DIR=results
BERT_CONFIG=bert_config.json
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

# # optimizers to loop over
# declare -a optimizers=("fusedadam" "adam")

# ## now loop through the above array
# for i in "${optimizers[@]}"
# do
#     srun --mem=20G -c 8 --gres=gpu:1 -p p100 \
#     python3 run_pretraining.py \
#         --input_dir=$DATA_DIR \
#         --output_dir=$CHECKPOINTS_DIR \
#         --config_file=$BERT_CONFIG \
#         --optimizers=$i \
#         --bert_model=bert-large-uncased \
#         --train_batch_size=16 \
#         --max_seq_length=128 \
#         --max_predictions_per_seq=20 \
#         --max_steps=2200 \
#         --warmup_proportion=0.128 \
#         --num_steps_per_checkpoint=10000 \
#         --learning_rate=2e-5 \
#         --fp16 \
#         --do_train
# done


srun --mem=20G -c 10 --gres=gpu:1 -p t4 \
    python3 run_pretraining.py \
        --input_dir=$DATA_DIR \
        --output_dir=$CHECKPOINTS_DIR \
        --config_file=$BERT_CONFIG \
        --optimizer="fusedadam" \
        --bert_model=bert-large-uncased \
        --train_batch_size=32 \
        --max_seq_length=128 \
        --max_predictions_per_seq=20 \
        --max_steps=2100 \
        --warmup_proportion=0.128 \
        --num_steps_per_checkpoint=10000 \
        --learning_rate=2e-5 \
        --fp16 \
        --gradient_accumulation_steps=2 \
        --do_train \
        --benchmark \
        --benchmark_partition t4

# --fp16 \