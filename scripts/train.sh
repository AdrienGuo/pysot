epochs=(200)
batch_sizes=(16)

# 決定 template image 是否要有 bg
# bg: background
# nbg: no background
template_bg="bg"
# 要加入多少的 bg
template_context_amount=(2)


for epoch in ${epochs[@]}; do
    for batch_size in ${batch_sizes[@]}; do
        time=`TZ=Asia/Taipei date +"%Y.%m.%d-%H:%M"`
        echo "save log to ./training_logs/e${epoch}_b${batch_size}_${template_bg}${template_context_amount}_${time}.log"
        sleep 2

        torchrun --nproc_per_node=1 \
            ./tools/train.py \
            --epoch ${epoch} \
            --batch_size ${batch_size} \
            --template_bg ${template_bg} \
            --template_context_amount ${template_context_amount} \
            --cfg ./experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
            >> ./training_logs/e${epoch}_b${batch_size}_${template_bg}${template_context_amount}_${time}.log
    done
done
