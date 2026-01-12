# DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences

### [Project Page](https://sled-group.github.io/driVLMe/) | [Paper](https://arxiv.org/abs/2406.03008) | [Video](https://youtu.be/Ep5fYLGkmsg)

Yidong Huang, Jacob Sansom, Ziqiao Ma, Felix Gervits, Joyce Chai  
University of Michigan, ARL  
IROS 2024

![Method](/method.jpg)

## Setup
The code is adopted from [video-chatgpt](https://github.com/mbzuai-oryx/Video-ChatGPT). We recommend setting up a conda environment for the project:
```shell
conda create --name=drivlme python=3.10
conda activate drivlme

git clone git@github.com:sled-group/driVLMe.git
cd driVLMe
pip install -r requirements.txt
pip install -e .
```

## Prepare Llava weights
Please follow the following instructions to get LLaVA weights.

- Get the original LLaMA weights in the Hugging Face format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
- Use the following scripts to get LLaVA weights by applying our delta.
```shell
python scripts/apply_delta.py \ 
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```

The above command will download the LLaVA-Lightening-7B-v1-1 delta from Hugging Face, apply it to the provided LLaMA 
weights and save the LLaVA-Lightening-7B-v1-1 weights in the current directory.

Alternatively you can download the ready LLaVA-Lightening-7B weights from [mmaaz60/LLaVA-Lightening-7B-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1).


## Prepare Dataset
You can get the data at [this dowanload link](https://www.dropbox.com/scl/fo/f429if26mveud6zcek54y/AEjkxF_DZ-DO87xJiOVkQTE?rlkey=shwm81sebtftttkflx8iqghxt&st=eux3itpx&dl=0) and untar all the file under folder "videos".

## Pretrain on bddx dataset

Train on 4 A40 GPUs using the command
```shell
torchrun --nproc_per_node=4 --master_port 29001 drivlme/train/train_xformers.py \
          --model_name_or_path  <Path to Llava> \
          --version v1 \
          --data_path datasets/bddx_pretrain.json \
          --video_folder videos/bdd100k_feats \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./DriVLMe_model_weights/bddx_pretrain_ckpt \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 1000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```


## Finetune on SDN dataset


Train on 4 A40 GPUs with deepspeed zero2 using the command
```shell
deepspeed --master_port=29001 drivlme/train/train_xformers.py \
          --deepspeed ./scripts/zero2.json \
          --model_name_or_path  <Path to Llava> \
          --pretrain_mm_mlp_adapter ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin \
          --version v1 \
          --lora_enable True \
          --lora_r 128 \
          --lora_alpha 256 \
          --data_path datasets/DriVLMe_sft_data.json \
          --video_folder videos/SDN_train_feats \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./model_path/DriVLMe \
          --num_train_epochs 3 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 500 \
          --save_total_limit 3 \
          --learning_rate 5e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --lazy_preprocess True
```
