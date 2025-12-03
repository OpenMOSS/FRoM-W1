# H-GPT

## Environment Setup
Run the following commands to setup the environment for H-GPT
```bash
conda create -f env.yaml
conda activate hgpt
```

## Dependencies
The dependencies abd their usage for H-GPT are listed below, please download and put them to `./deps` folder.

| Model | Usage | Note |
| :--- | :--- | :--- |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) | Data preparation | Put this under `./deps/body_models`|
| [GloVe](https://github.com/EricGuo5513/text-to-motion/tree/main/glove) | Evaluation | Put this under `./deps` and rename it to `glove_t2m` |
| [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | Base model for the Motion Generator | Put this under `./deps` |
|[CoT data](https://huggingface.co/datasets/OpenMOSS-Team/FRoM-W1-Datasets/tree/main)| for training w. CoT | | 
| [Evaluation meta & model](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main) | Evaluation | Put the meta (`mean.npy` and `std.npy`) under `./deps/t2m/t2mx/text_mot_match/meta`<br><br>and the model (`finest.tar`)<br><br>`./deps/t2m/t2mx/text_mot_match/model` |

Finally, your `./deps` folder should look like 
```bash
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

Run 

```bash
sh scripts/run_train_t2mx_vqvae_30fps.sh
```
> [!IMPORTANT]  
> Modify the properties below to fit your own path:

```yaml
DATASET:
  MOTIONX:
    DATA_ROOT: "" # the root of your processed data
    SPLIT_PATH: "" # the root folder to the split file (train.txt, etc.)
    MOTION_FEAT_PATH: "" # the root folder to the motion feature
    SEMANTIC_TEXT_PATH: "" # the root folder to the labeled text data 
    COT_PATH: "" # the root folder of CoT data (not used for this step)
    MEAN_STD_PATH: "" # the root folder of the meta at training stage
    EVAL_MEAN_STD_PATH: "" # the root folder of the meta at evaluation stage
    MOTION_TOKEN_PATH: "" # the root folder to the tokenized motion (at tokenization stage)
```

> [!NOTE]
> You can also try other VQ settings by replacing 
```yaml
model:
    motion_vae: ${vq.vqvae_512_512}
```
with one of `[vq.vq_1k_1k, vq.vq_1k_2k, vq.vq_2k_1k, vq.vq_2k_2k]`




### Motion Generator
- **Tokenization**

- **Training**


## Evaluation

## Deployment

To be done