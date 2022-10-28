# !/bin/sh

part="test"    # train / test
dataset="PatternMatch_test"
criteria="above"
search=(255)
crop_method="new"
bg="1"    # 不影響結果
neg=(0.0)
anchors=(11)


echo "Search: ${search}"
echo "Crop method: ${crop_method}"
echo "Anchors: ${anchors}"
echo "Dataset: ${dataset}"
echo "Criteria: ${criteria}"
sleep 5

python3 ./kmeans/demo.py \
	--crop_method ${crop_method} \
	--bg ${bg} \
	--neg ${neg} \
    --anchors ${anchors} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--part ${part} \
	--dataset_path ./datasets/${part}/${dataset}/ \
	--dataset_name ${dataset} \
	--criteria ${criteria} \
	> ./kmeans/demo/${part}/${dataset}/${criteria}/k${anchors}.txt
