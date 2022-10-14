# !/bin/sh

search=(255)
crop_method="new"
bg="1"    # 不影響結果
neg=(0.0)
anchors=(2)
dataset="text_new"
criteria="above"


echo "search: ${search}"
echo "crop_method: ${crop_method}"
echo "anchors: ${anchors}"
echo "dataset: ${dataset}"
echo "criteria: ${criteria}"
sleep 5

python3 ./kmeans/demo.py \
	--crop_method ${crop_method} \
	--bg ${bg} \
	--neg ${neg} \
    --anchors ${anchors} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset_path ./datasets/train/${dataset}/ \
	--dataset_name ${dataset} \
	--criteria ${criteria}
	# > ./kmeans/demo/${dataset}_${criteria}_${anchors}.txt
