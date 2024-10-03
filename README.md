# MMML-VideoCaptioning


## Original Dataset

Using Valor-32k but downloaded from [VAST](https://github.com/TXH-mercury/VAST?tab=readme-ov-file).

Downloaded here and choose 'anotation/valor32k'
[Google Drive](https://drive.google.com/file/d/1bOLUbbnPTgUp_Nc0PgORKC-174CwgwPm/view)


## Sample from Valor-32k for Evaluation

Details could be seen in 'data_process/dataset_sample.ipynb'.

Basically, sampled 120 video clips, with 60 non-verbal clips, and 60 verbal clips (20 light-verbal / 20 mid-verbal / 20 heavy-verbal).

Sampled dataset could be see in 'data/sample_v_nv_test120_new.json'

## Download Video Clips

Requirements

```
pip install yt-dlp

brew install ffmpeg 
```

Download

```
python download_clips.py 'data/sample_v_nv_test120_new.json' 'data/test120'
```