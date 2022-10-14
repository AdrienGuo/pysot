# !/bin/sh

train_dataset="text_new"
train_criteria="above"
search=(255)
crop_method="new"
bg="1.3"
neg=(0.0)
anchors=(11)
epochs=(200)
batch=(32)
epoch_num=(200)
test_dataset="text_new"
test_criteria="above"

model_dir="./save_models/${train_dataset}_${train_criteria}"
model="x${search}_${crop_method}_bg${bg}_${train_dataset}_${train_criteria}_k${anchors}_neg${neg}_e${epochs}_b${batch}"
model_epoch="model_e${epoch_num}.pth"
# model="siamrpn_r50_l234_dwxcorr/model.pth"    # 官方權重檔


echo "Load model from: ${model}"
echo "Epoch number: ${epoch_num}"
echo "Train dataset: ${train_dataset}"
echo "Train criteria: ${train_criteria}"
echo "Test dataset: ${test_dataset}"
echo "Test criteria: ${test_criteria}"
echo "Check Your Anchor Setting !!"
echo "Check Your import PCBDataset library"
sleep 5

python3 ./tools/eval_pcb.py \
    --model ${model_dir}/${model}/${model_epoch} \
	--crop_method ${crop_method} \
	--bg ${bg} \
	--neg ${neg} \
	--dataset_name ${test_dataset} \
	--dataset_path ./datasets/test/${test_dataset} \
	--criteria ${test_criteria} \
	--cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml
