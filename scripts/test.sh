# test
# ./snapshot/checkpoint_e20.pth
# ./experiments/siamrpn_r50_l234_dwxcorr/model.pth
python ./tools/test_pcb.py 	\
    --snapshot ./snapshot/checkpoint_e20.pth \
	--dataset ./PCB/ \
	--config ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml
