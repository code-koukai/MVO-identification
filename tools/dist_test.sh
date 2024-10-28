CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29835}

<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=2 \
=======
CUDA_VISIBLE_DEVICES=2 \
>>>>>>> f06d041bdbefa637f250d885f55c20ff896e9655
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
