# run hybrid search (bm25+embedding)

# TATQA, HybridQA, ConvFinQA
DATASET_NAME="ConvFinQA"

# train, dev, test
SPLIT="test"

# mpnet, e5, granite, qwen
MODEL_SHORT="granite"

if [ "$MODEL_SHORT" = "granite" ]; then
    MODEL="ibm-granite/granite-embedding-125m-english"
    EMB_DIM=768
elif [ "$MODEL_SHORT" = "mpnet" ]; then
    MODEL="sentence-transformers/all-mpnet-base-v2"
    EMB_DIM=768
elif [ "$MODEL_SHORT" = "e5" ]; then
    MODEL="intfloat/multilingual-e5-large-instruct"
    EMB_DIM=1024
elif [ "$MODEL_SHORT" = "qwen" ]; then
    MODEL="Qwen/Qwen3-Embedding-8B"
    EMB_DIM=4096
else
    echo "Unknown model short name: $MODEL_SHORT"
    exit 1
fi

DB_PATH="data/${DATASET_NAME}/${DATASET_NAME}_${MODEL_SHORT}.db"
OUTPUT_PATH="data/${DATASET_NAME}/${DATASET_NAME}_${SPLIT}_results_${MODEL_SHORT}.json"
EVAL_OUTPUT_PATH="data/${DATASET_NAME}/${DATASET_NAME}_${SPLIT}_results_eval_${MODEL_SHORT}.json"

echo "Dataset: ${DATASET_NAME} - ${SPLIT}"
echo "EMB MODEL: ${MODEL}"
echo "Milvus DB: ${DB_PATH}"
echo "Output PATH: ${OUTPUT_PATH}"
echo "Eval Output PATH: ${EVAL_OUTPUT_PATH}"

python src/baseline.py \
    --db_path "$DB_PATH" \
    --dataset "$DATASET_NAME" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --emb_dim "$EMB_DIM" \
    --index-type AUTOINDEX \
    --metric-type IP \
    --top_text 20 \
    --top_table 20 \
    --output_path "$OUTPUT_PATH"

python src/baseline_eval.py \
    --dataset "$DATASET_NAME" \
    --split "$SPLIT" \
    --sys_output "$OUTPUT_PATH" \
    --eval_output "$EVAL_OUTPUT_PATH"
