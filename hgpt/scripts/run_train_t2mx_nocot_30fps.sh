# cd /inspire/hdd/project/embodied-multimodality/public/hgpt/pli/HumanoidGPT
# source /opt/conda/bin/activate
# conda activate hgpt

# TRAIN_PID=$!
# echo "PID: $TRAIN_PID"

torchrun --nnodes=1 --nproc-per-node=8 train.py --cfg_assets ./configs/assets.yaml --cfg configs/exp/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_nocot_30fps.yaml --task t2m --nodebug

# nohup torchrun --nnodes=1 --nproc-per-node=8 train.py --cfg_assets ./configs/assets.yaml --cfg configs/exp/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_nocot_30fps.yaml --task t2m --nodebug > outs/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_nocot_30fps.out 2>&1 &