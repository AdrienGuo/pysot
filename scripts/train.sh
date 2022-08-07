time=$(date +"%Y_%m_%d--%H-%M")
echo "save log to ./training_logs/${time}.log"

torchrun --nproc_per_node=1 \
    ./tools/train.py \
    --cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    # >> ./training_logs/${time}.log