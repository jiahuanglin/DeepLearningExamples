#!/bin/bash

for i in "$@"
do
case $i in
    -p=*|--partition=*)
    PARTITION="${i#*=}"
    shift # past argument=value
    ;;
    -b=*|--batch_size=*)
    BATCH_SIZE="${i#*=}"
    shift # past argument=value
    ;;
    -a=*|--gradient_accumulation=*)
    GRADIENT_ACCUMULATION="${i#*=}"
    shift # past argument=value
    ;;
    -g=*|--ngpu_per_node=*)
    NUM_GPU_PER_NODE="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--nnodes=*)
    NNODES="${i#*=}"
    shift # past argument=value
    ;;
    -r=*|--node_rank=*)
    NODE_RANK="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--master_dir=*)
    MASTER_DIR="${i#*=}"
    shift # past argument=value
    ;;
    -o=*|--master_output=*)
    MASTER_OUTPUT="${i#*=}"
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done
echo "PARTITION                 = ${PARTITION}"
echo "BATCH_SIZE                = ${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION     = ${GRADIENT_ACCUMULATION}"
echo "NNODES                    = ${NNODES}"
echo "NODE_RANK                 = ${NODE_RANK}"
echo "NUM_GPU_PER_NODE          = ${NUM_GPU_PER_NODE}"
echo "MASTER_DIR                = ${MASTER_DIR}"
echo "MASTER_OUTPUT             = ${MASTER_OUTPUT}"
# echo "Number files in MASTER_DIR with ip as extension:" $(ls -1 "${MASTER_DIR}"/*.ip | wc -l)
# echo "Number files in MASTER_DIR with MASTER_OUTPUT as extension:" $(ls -1 "${MASTER_DIR}"/*."${MASTER_OUTPUT}" | wc -l)
echo "scheduled to node: $SLURMD_NODENAME"

MASTER_FILE="${MASTER_DIR}/${MASTER_OUTPUT}"

SUBNET=172.17.8.

if [ $NODE_RANK = 0 ] ; then
    IP=$(ip a | grep $SUBNET | awk '{ print $2 }')
    MASTER_ADDR=`/bin/hostname -s`
    # #Get a random unused port on this host(MASTER) between 2000 and 9999
    # #First line gets list of unused ports
    # #2nd line restricts between 2000 and 9999s
    # #3rd line gets single random port (first line) from the list
    # MASTER_PORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
    #     grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
    #     sort | uniq | shuf | head -n 1`
    MASTER_PORT=6000
    echo "MASTER IP addr $IP"    
    echo "MASTER NODE $MASTER_ADDR"
    echo "MASTER PROT = ${MASTER_PORT}"
    echo "${MASTER_ADDR} ${MASTER_PORT}" > $MASTER_FILE
else
    while [ ! -f $MASTER_FILE ] ; do
        echo "Waiting for master to write to file: ${MASTER_FILE}"
        sleep 30
    done
    MASTER_ADDR=$(cat $MASTER_FILE | awk '{print $1}')
    MASTER_PORT=$(cat $MASTER_FILE | awk '{print $2}')
    echo "MASTER NODE $MASTER_ADDR"
    echo "MASTER PROT = ${MASTER_PORT}"
fi


DATA_DIR=/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/nvidia/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
RESULTS_DIR=results
BERT_CONFIG=bert_config.json
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

mkdir -p $CHECKPOINTS_DIR

DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPU_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

NCCL_DEBUG=TRACE python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
run_pretraining.py \
    --input_dir=$DATA_DIR \
    --output_dir=$CHECKPOINTS_DIR \
    --config_file=$BERT_CONFIG \
    --bert_model=bert-large-uncased \
    --train_batch_size=$BATCH_SIZE \
    --optimizer="fusedadam" \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --max_steps=2100 \
    --warmup_proportion=0.128 \
    --num_steps_per_checkpoint=10000 \
    --learning_rate=2e-5 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --fp16 \
    --do_train \
    --nproc_per_node=$NUM_GPU_PER_NODE \
    --benchmark \
    --benchmark_partition=$PARTITION
