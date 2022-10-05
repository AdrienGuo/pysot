# bin/bash

search=(255)
crop_method="new"
bg="2"
anchors=(11)
epochs=(200)
batch_sizes=(32)
# normalized=True
dataset="./datasets/train/"


for epoch in ${epochs[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        time=`TZ=Asia/Taipei date +"%Y.%m.%d-%H:%M"`
        echo "Save log to ./training_logs/x${search}_${crop_method}_bg${bg}_k${anchors}_e${epochs}_b${batch_size}_${time}.log"
        sleep 3

        python3 \
            ./tools/train.py \
            --crop_method ${crop_method} \
            --bg ${bg} \
            --anchors ${anchors} \
            --epoch ${epoch} \
            --batch_size ${batch_size} \
            --dataset ${dataset} \
            --cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
            # > ./training_logs/x${search}_${crop_method}_bg${bg}_k${anchors}_e${epoch}_b${batch_size}_${time}.log
    done
done

# torchrun --nproc_per_node=1 \
