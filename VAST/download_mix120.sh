pip3 install gdown

gdown 1e8-NMTbTRhSbMTXrsNzEsoyT7vHF-nNI
unzip extracted_data.zip
rm extracted_data.zip

mv /opt/dlami/nvme/VividScribe/data/mix120/raw_video datasets/srcdata/mix-120/videos
mv /opt/dlami/nvme/VividScribe/data/mix120/extracted_data/audio_22050hz datasets/srcdata/mix-120/audios