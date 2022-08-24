# !/bin/sh

# For different model (config 也要改)
# --model ./save_models/my_model/model_eXX.pth						(my_model)
# --model ./experiments/siamrpn_r50_l234_dwxcorr/model.pth			(siamrpn_r50_l234_dwxcorr)
# --model ./experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth		(siamrpn_r50_l234_dwxcorr_otb)
# --model ./experiments/siammask_r50_l3/model.pth					(siammask_r50_l3)

# 亭儀的
# --model ./tf/pysot/model/siamrpn_r50_l234_dwxcorr/model.pth
# --config ./tf/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml


kmeans=(5)
resolution=(255)
epochs=(200)
epoch_num=(20)
batch=(32)
# 決定 template image 是否要有 bg
# bg: background, nbg: no background
template_bg="bg"
# 要加入多少的 bg
template_context_amount=(0.5)


model_dir="./save_models"
# model="official/model"
model="k${kmeans}_r${resolution}_e${epochs}_b${batch}_${template_bg}${template_context_amount}/model_e${epoch_num}"
dataset="val"
save_dir="./results/${dataset}"

python ./tools/test_pcb.py \
    --model ${model_dir}/${model}.pth \
	--template_bg ${template_bg} \
	--template_context_amount ${template_context_amount} \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./testing_dataset/PCB/${dataset}/ \
	--annotation ./testing_dataset/PCB/${dataset}/ \
	--save_dir ${save_dir}
