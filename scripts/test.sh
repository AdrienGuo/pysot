# !/bin/sh

# For different model (config 也要改)
# --model ./save_models/my_model/model_eXX.pth						(my_model)
# --model ./experiments/siamrpn_r50_l234_dwxcorr/model.pth			(siamrpn_r50_l234_dwxcorr)
# --model ./experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth		(siamrpn_r50_l234_dwxcorr_otb)
# --model ./experiments/siammask_r50_l3/model.pth					(siammask_r50_l3)

# 亭儀的
# --model ./tf/pysot/model/siamrpn_r50_l234_dwxcorr/model.pth
# --config ./tf/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml

# For different testing dataset (記得 annotation 也要改)
# --dataset ./testing_dataset/PCB/train/
# --dataset ./testing_dataset/PCB/val/

python ./tools/test_pcb.py \
<<<<<<< HEAD
    --model ./save_models/my_model/model_e100.pth \
=======
    --model ./save_models/my_model/model_e30.pth \
>>>>>>> anchor
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
	--dataset ./testing_dataset/PCB/train/ \
	--annotation ./testing_dataset/PCB/train/ \
	--save_dir ./results/
