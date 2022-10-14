# bin/bash

search=(255)
crop_method="new"
bg="1.3"
anchors=(11)
epochs=(200)
batch_sizes=(32)
# normalized=True
neg=(0.0)
dataset="text_new"
criteria="above"


echo "crop_method: ${crop_method}"
echo "anchors: ${anchors}"
echo "background: ${bg}"
echo "negative pair ratio: ${neg}"
echo "dataset: ${dataset}"
echo "criteria: ${criteria}"
echo "Check Your Anchor Setting !!"
echo "Check Your import PCBDataset library"
sleep 5

for epoch in ${epochs[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        time=`TZ=Asia/Taipei date +"%Y.%m.%d-%H:%M"`
        echo "Save log to ./training_logs/x${search}_${crop_method}_bg${bg}_${dataset}_${criteria}_k${anchors}_neg${neg}_e${epochs}_b${batch_size}_${time}.log"
        sleep 5

        python3 \
            ./tools/train.py \
            --crop_method ${crop_method} \
            --bg ${bg} \
            --neg ${neg} \
            --anchors ${anchors} \
            --epoch ${epoch} \
            --batch_size ${batch_size} \
            --dataset_path ./datasets/train/${dataset} \
            --dataset_name ${dataset} \
            --criteria ${criteria} \
            --cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
            # > ./training_logs/x${search}_${crop_method}_bg${bg}_${dataset}_${criteria}_k${anchors}_neg${neg}_e${epochs}_b${batch_size}_${time}.log
    done
done

# torchrun --nproc_per_node=1 \
