# !/bin/sh

# For different model (config 也要改)
# --snapshot ./snapshot/checkpoint_e20.pth
# --snapshot ./experiments/siamrpn_r50_l234_dwxcorr/model.pth
# --snapshot ./experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth
# --snapshot ./experiments/siammask_r50_l3/model.pth

# 亭儀的
# --snapshot ./tf/pysot/model/siamrpn_r50_l234_dwxcorr/model.pth
# --config ./tf/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml

# For different testing dataset (記得 annotation 也要改)
# --dataset ./testing_dataset/PCB/train/
# --dataset ./testing_dataset/PCB/val/

python ./tools/test_pcb.py 	\
    --snapshot ./snapshot/checkpoint_e20.pth \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./testing_dataset/PCB/val/ \
	--annotation ./testing_dataset/PCB/val/ \
	--save_dir ./results/
