# VividScribe

## Original Dataset

Using Valor-32k but downloaded from [VAST](https://github.com/TXH-mercury/VAST?tab=readme-ov-file).

Downloaded here and choose `anotation/valor32k`
[Google Drive](https://drive.google.com/file/d/1bOLUbbnPTgUp_Nc0PgORKC-174CwgwPm/view)

## Sample from VALOR-32k + VAST 27M for Evaluation

Details could be seen in `data_processing/dataset_sample.ipynb`.

In summary, 120 video clips were sampled, including:

- 60 non-verbal clips from valor 32k
- 60 verbal clips from VAST 27M

Sampled annotation json dataset could be seen in `data/mix_120/mix_120.json`.

Raw video clips and extracted data are available at [mix120_GoogleDrive](https://drive.google.com/drive/folders/1HERtDdyvf7Ts2HTnXbdUBOtSXE28waek?usp=drive_link)

> **Note:** There are 3 of the clips from the VALOR subset that cannot be processed by frame and audio wave extractor, and all of them have no subtitles, indicating that there's limited information from the audio. Hence, they are skipped in the evaluation.
>
> p_o6NQX7lmE_0.000_10.000.wav (no subtitle)
>
> niJg7Q1XLyU_50.000_60.000.wav (no subtitle)
>
> wj-gglKQ3KI_30.000_40.000.wav (no subtitle)

## Prepare Data

### Download Video Clips

Requirements

```
pip install yt-dlp

brew install ffmpeg 
```

Download

1. VALOR data

    ```
    python data_processing/download_clips.py 'data/valor120/sample_v_nv_test120_new.json' 'data/valor120/raw_video'
    ```
2. VAST data
    **TODO**

### Combine Video Clips

1. extract the 60 non-verbal clips from `data/valor120/sample_v_nv_test120_new.json`:

    ```
    python data_processing/filter_nonverbal.py
    ```

2. combine the 60 non-verbal clips with the 60 verbal clips that randomly sampled from `data/vast120/vast_test120.json`:
    
    ```
    python data_processing/combine_dataset.py
    ```

### Create Mapping
> **Note:** This step is to create a mapping from the 2 dataset sources to the video clip ids, so that we can evaluate two subsets separately.

```
python data_processing/mapping.py
```

### Extract Video Frames & Audio Waves

```
python  python data_processing/extract_frame_and_wav_multiprocess.py
```

## Evaluation

> **Note:** Before running the evaluation, make sure to prepare the captions in the right json format as below:

> ```
>    {"annotations": [
>        {
>            "video_id": "video_id1",
>            "caption": "caption1"
>        },
>        {
>            "video_id": "video_id2",
>            "caption": "caption2"
>        },
>        ...
>    ]}
>```

### Compute Metrics

```
bash eval.sh
```
