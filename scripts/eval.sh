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
# model="x${search}_${crop_method}_k${kmeans}_e${epochs}_b${batch}/model_e${epoch_num}.pth"
# model="siamrpn_r50_l234_dwxcorr/model.pth"    # 官方權重檔


echo "Load model from: ${model}"
echo "Dataset: ${dataset}"
sleep 3

python3 ./tools/eval_pcb.py \
    --model ${model_dir}/${model} \
	--crop_method ${crop_method} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./datasets/${dataset}/ \
	--annotation ./datasets/${dataset}/ \
