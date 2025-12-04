# H-GPT

## Environment Setup
Run the following commands to setup the environment for H-GPT
```bash
conda create -f env.yaml
conda activate hgpt
```

## Dependencies
The dependencies and their usage for H-GPT are listed below, please download and put them to `./deps` folder.

| Model | Usage | Note |
| :--- | :--- | :--- |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) | Data preparation | Put this under `./deps/body_models`|
| [GloVe](https://github.com/EricGuo5513/text-to-motion/tree/main/glove) | Evaluation | Put this under `./deps` and rename it to `glove_t2m` |
| [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Base model for the Motion Generator | Put this under `./deps` |
|[CoT data](https://huggingface.co/datasets/OpenMOSS-Team/FRoM-W1-Datasets/tree/main)| for training w. CoT | | 
| [Evaluation meta & model](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main) | Evaluation | Put the meta (`mean.npy` and `std.npy`) under `./deps/t2m/t2mx/text_mot_match/meta`<br><br>and the model (`finest.tar`)<br><br>`./deps/t2m/t2mx/text_mot_match/model` |

Finally, your `./deps` folder should look like 
```plaintext
./deps/
|-- Meta-Llama-3.1-8B
|-- body_models
|   `-- smplx
|-- glove_t2m
`-- t2m
    `-- t2mx
        `-- text_mot_match
            |-- meta
            `-- model
```

## Data Preparation

To be done

## Training

### Whole-Body Motion Tokenizer
> [!IMPORTANT]  
> Modify the properties below in `configs/exp/1114_8gpu_config_t2mx_stage1_body_hands_vqvae512x512_30fps.yaml` to fit your own path:

```yaml
DATASET:
  MOTIONX:
    DATA_ROOT: "" # the root of your processed data
    SPLIT_PATH: "" # the root folder to the split file (train.txt, etc.)
    MOTION_FEAT_PATH: "" # the root folder to the motion feature
    SEMANTIC_TEXT_PATH: "" # the root folder to the labeled text data 
    COT_PATH: "" # the root folder of CoT data (not used for Tokenizer training)
    MEAN_STD_PATH: "" # the root folder of the meta at training stage
    EVAL_MEAN_STD_PATH: "" # the root folder of the meta at evaluation stage
    MOTION_TOKEN_PATH: "" # the root folder to the tokenized motion (at tokenization stage)
```
Run 

```bash
sh scripts/run_train_t2mx_vqvae_30fps.sh
```
to train the Whole-Body Motion Tokenizer

> [!NOTE]
> You can also try other VQ settings by replacing 
```yaml
model:
    motion_vae: ${vq.vqvae_512_512}
```
with one of `[vq.vq_1k_1k, vq.vq_1k_2k, vq.vq_2k_1k, vq.vq_2k_2k]`




### Motion Generator
#### Tokenization
> [!IMPORTANT]  
> Modify the property below to the path to the pretrained motion tokenizer:

```yaml
TRAIN:
  PRETRAINED_VAE: '' # VQ model path
```

> [!NOTE]
> The tokenized motion will be saved at:
```yaml
DATASET:
  MOTIONX:
    MOTION_TOKEN_PATH: "" # the root folder to the tokenized motion (at tokenization stage)
```
Run

```bash
sh scripts/run_tokenize_t2mx_30fps.sh
```
to tokenize the motion data



#### Training
> [!IMPORTANT]  
> 1. Modify the properties `hgpt/configs/exp/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_cotv3_30fps.yaml` and `hgpt/configs/exp/1114_8gpu_config_t2mx_stage2_body_hands_llama_vqvae512x512_nocot_30fps.yaml` to fit your own path as mentioned [here](#whole-body-motion-tokenizer).
> 2. Remember to set this property to your CoT data path to train w. CoT
```yaml
DATASET:
  MOTIONX:
    COT_PATH: "" # the root folder of CoT data
```

- Train w.o. CoT
<br>
Run

```bash
sh scripts/run_train_t2mx_nocot_30fps.sh
```

- Train w. CoT
<br>
Run

```bash
sh scripts/run_train_t2mx_cot_30fps.sh
```


## Evaluation
> [!IMPORTANT]  
> Modify the property below to the path to the pretrained motion generator:

```yaml
TEST:
  CHECKPOINTS: ""
```

- Evaluation w.o. CoT
<br>
Run

```bash
sh scripts/run_test_t2mx_nocot_30fps.sh
```

- Evaluation w. CoT
<br>
Run

```bash
sh scripts/run_test_t2mx_cot_30fps.sh
```

For more details of evaluation, see [EVAL.md](./EVAL.md)
