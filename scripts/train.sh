python3 -m torch.distributed.run --nproc_per_node=1 \
    ./tools/train.py \
    --cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml