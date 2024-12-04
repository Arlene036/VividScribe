VIDEOPATH="/home/ec2-user/VividScribe/data/vast3k/frames_fps1/__aZILU6SqM.65"
MODELDIR="/home/ec2-user/VividScribe/VALOR/output/VALOR_base"

python inference.py \
 --video_path $VIDEOPATH \
 --task 'qa%tva' \
 --model_dir $MODELDIR \
 --question 'what is in the video'