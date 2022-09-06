# !/bin/sh

kmeans=(11)
search=(600)

# 決定 template image 是否要有 bg
# bg: background, nbg: no background
template_bg="bg"
# 要加入多少的 bg
template_context_amount=(2)

epochs=(200)
batch=(16)
epoch_num=(200)


model_dir="./save_models"
model="k${kmeans}_s${search}_${template_bg}${template_context_amount}-teacher_e${epochs}_b${batch}/model_e${epoch_num}.pth"
# model="siamrpn_r50_l234_dwxcorr/model.pth"    # 官方權重檔
dataset="teacher"
save_dir="./results/${dataset}"


python3 ./tools/test_pcb.py \
    --model ${model_dir}/${model} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./testing_dataset/PCB/${dataset}/ \
	--annotation ./testing_dataset/PCB/${dataset}/ \
	--template_bg ${template_bg} \
	--template_context_amount ${template_context_amount} \
	--save_dir ${save_dir}
