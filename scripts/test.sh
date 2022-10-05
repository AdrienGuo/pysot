# !/bin/sh

search=(255)
crop_method="new"
bg="2"
kmeans=(11)
epochs=(200)
epoch_num=(200)
batch=(32)
dataset="test"

model_dir="./save_models"
model="x${search}_${crop_method}_bg${bg}_k${kmeans}_e${epochs}_b${batch}/model_e${epoch_num}.pth"
# model="siamrpn_r50_l234_dwxcorr/model.pth"    # 官方權重檔
save_dir="./results/${dataset}"


echo "Load model from: ${model}"
sleep 3

python3 ./tools/test_pcb.py \
    --model ${model_dir}/${model} \
	--crop_method ${crop_method} \
	--bg ${bg} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./datasets/${dataset}/ \
	--annotation ./datasets/${dataset}/ \
	--save_dir ${save_dir}
