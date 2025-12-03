# cd /inspire/hdd/project/embodied-multimodality/public/hgpt/pli/HumanoidGPT
# source /opt/conda/bin/activate
# conda activate hgpt
torchrun --nnodes=1 --nproc-per-node=1 get_motion_code.py --cfg_assets ./configs/assets.yaml --cfg configs/exp/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_nocot_30fps.yaml --task token --nodebug

# nohup torchrun --nnodes=1 --nproc-per-node=1 get_motion_code.py --cfg_assets ./configs/assets.yaml --cfg configs/exp/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_nocot_30fps.yaml --task token --nodebug > outs/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_get_code.out 2>&1 &