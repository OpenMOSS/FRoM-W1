# cd /inspire/hdd/project/embodied-multimodality/public/hgpt/pli/HumanoidGPT
# source /opt/conda/bin/activate
# conda activate hgpt
torchrun --nnodes=1 --nproc-per-node=1 train.py --cfg_assets ./configs/assets.yaml --cfg configs/exp/1114_8gpu_config_t2mx_stage1_body_hands_vqvae512x512_30fps.yaml --task vae --nodebug

# nohup torchrun --nnodes=1 --nproc-per-node=8 train.py --cfg_assets ./configs/assets.yaml --cfg configs/exp/1114_8gpu_config_t2mx_stage1_body_hands_vqvae512x512_30fps.yaml --task vae --nodebug > outs/1114_8gpu_config_t2mx_stage1_body_hands_vqvae512x512_30fps.out 2>&1 &