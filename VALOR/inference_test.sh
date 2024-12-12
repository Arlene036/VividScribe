VIDEOPATH="/home/ubuntu/VividScribe/data/mix120/test120/Jk5dHnmduAg_320.000_330.000.mp4"
MODELDIR="/home/ubuntu/VividScribe/VALOR/output/VALOR_large"

python inference.py \
 --video_path $VIDEOPATH \
 --task 'qa%tva' \
 --model_dir $MODELDIR \
 --question 'what is in the video'