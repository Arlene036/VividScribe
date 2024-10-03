# VividScribe


## Original Dataset

Using Valor-32k but downloaded from [VAST](https://github.com/TXH-mercury/VAST?tab=readme-ov-file).

Downloaded here and choose `anotation/valor32k`
[Google Drive](https://drive.google.com/file/d/1bOLUbbnPTgUp_Nc0PgORKC-174CwgwPm/view)


## Sample from Valor-32k for Evaluation

Details could be seen in `data_process/dataset_sample.ipynb`.

In summary, 120 video clips were sampled, including:

- 60 non-verbal clips
- 60 verbal clips (divided into 20 light-verbal, 20 mid-verbal, and 20 heavy-verbal clips)

Sampled json dataset could be see in `data/sample_v_nv_test120_new.json`

Video clips are available at [test120_GoogleDrive](https://drive.google.com/drive/folders/1DOeMn5LxjNFtlSTV5frrLSOHMJ3eNCO_)

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
