
#!/usr/bin/env python3
basedir=$1
# msrvtt vqa
lr=${2:-1e-5}
bs=${3:-64}
epoch=${4:-10}
frozen=${5:-true}

output_name="VQA-11777-lr${lr}-bs${bs}-epoch${epoch}"
if [ "$frozen" = "true" ]; then
    output_name="${output_name}-frozen-v-3"
fi

python3 ./train.py \
--pretrain_dir $basedir \
--config ./config/VQA-11777.json \
--output_dir "${basedir}/${output_name}" \
--train_epoch $epoch \
--learning_rate $lr \
--train_video_sample_num 4 \
--test_video_sample_num 8 \
--save_best true \
--first_eval false \
--use_task_prompt true \
--frozen_vision $frozen \
--beam_size_qa 3 \
--max_generation_len 50



#  msvd qa 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32701 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/VQA-msvd.json \
# --output_dir $basedir'/vqa-msvd-lr1e-5-bs64-train4test8'   \
# --learning_rate 1e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8 \
# --save_best true \






#  music avqa 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/VQA-music.json \
# --output_dir $basedir'/vqa-music-lr2e-5-bs64-25epoch'   \
# --train_epoch 25 \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --train_audio_sample_num 4 \
# --test_video_sample_num 8 \
# --test_audio_sample_num 8 \
# --save_best true \



#   activitynet qa

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/VQA-activitynet.json \
# --output_dir $basedir'/vqa-activitynet-lr2e-5-bs64-train8test16'   \
# --learning_rate 2e-5  \
# --first_eval false \
# --train_video_sample_num 8 \
# --train_audio_sample_num 4 \
# --test_video_sample_num 16 \
# --test_audio_sample_num 8 \
# --train_task 'qa%tv' \
# --test_task 'qa%tv' \
# --save_best true \
# --checkpointing true 



#   tgif-frame qa 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32011 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/VQA-tgif-frame.json \
# --output_dir $basedir'/VQA-tgif-frame-lr2e-5-bs64-epoch5'   \
# --learning_rate 2e-5  \
# --train_video_sample_num 4 \
# --test_video_sample_num 8 \
# --save_best true \

#  vqav2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32611 ./train.py \
# --pretrain_dir $basedir \
# --config ./config/VQAv2_3129_woweight.json \
# --output_dir $basedir'/vqav2-lr2e-5-bs256-224-txtmapper3129-rectransform-fullmasker-withoutweights-epoch200-392'   \
# --learning_rate 2e-5  \
# --train_epoch 200 \
# --train_batch_size 256 \
# --valid_freq 3 \
# --first_eval false \
# --video_resolution 392  \
# --full_masker true \

