# bin/bash

dataset="all"
criteria="above"
bk="BK200"
neg=(0.0)
anchors=(11)
search=(255)
crop_method="new"
bg="1.3"
epochs=(1000)
batch_sizes=(2)
accum_iter=(16)
test_dataset="all"


echo "crop_method: ${crop_method}"
echo "Anchors: ${anchors}"
echo "Background: ${bg}"
echo "Neg pair ratio: ${neg}"
echo "Dataset: ${dataset}"
echo "Criteria: ${criteria}"
echo "Test Dataset: ${test_dataset}"
echo "Check Your Anchor Setting !!"
echo "Check Your import PCBDataset library"
sleep 5

for epoch in ${epochs[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        time=`TZ=Asia/Taipei date +"%Y.%m.%d-%H:%M"`
        echo "Save log to ./training_logs/x${search}_${crop_method}_bg${bg}_${dataset}_${criteria}_k${anchors}_neg${neg}_e${epochs}_b${batch_size}_${time}.log"
        sleep 3

        python3 \
            ./tools/train.py \
            --crop_method ${crop_method} \
            --bg ${bg} \
            --neg ${neg} \
            --anchors ${anchors} \
            --epoch ${epoch} \
            --batch_size ${batch_size} \
            --accum_iter ${accum_iter} \
            --dataset_path ./datasets/train/${dataset} \
            --dataset_name ${dataset} \
            --criteria ${criteria} \
            --bk ${bk} \
            --cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
            --test_dataset ./datasets/test/${dataset} \
            # > ./training_logs/x${search}_${crop_method}_bg${bg}_${dataset}_${criteria}_k${anchors}_neg${neg}_e${epochs}_b${batch_size}_${time}.log
    done
done

# torchrun --nproc_per_node=1 \
