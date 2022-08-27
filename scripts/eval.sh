# !/bin/sh

kmeans=(5)
resolution=(255)
epochs=(200)
epoch_num=(20)
batch=(32)

# 決定 template image 是否要有 bg
# bg: background, nbg: no background
template_bg="bg"
# 要加入多少的 bg
template_context_amount=(2)


model_dir="./save_models"
model="k${kmeans}_r${resolution}_e${epochs}_b${batch}_${template_bg}${template_context_amount}/model_e${epoch_num}"
dataset="val"
save_dir="./results/${dataset}"

python3 ./tools/eval_pcb.py \
    --model ${model_dir}/${model}.pth \
	--template_bg ${template_bg} \
	--template_context_amount ${template_context_amount} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./testing_dataset/PCB/${dataset}/ \
	--annotation ./testing_dataset/PCB/${dataset}/ \
