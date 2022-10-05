# !/bin/sh

search=(255)
crop_method="old"
anchors=(11)
dataset="train"

sleep 2


python3 ./kmeans/demo.py \
	--crop_method ${crop_method} \
    --anchors ${anchors} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./datasets/${dataset}/ \
	--annotation ./data/${dataset}/ \
