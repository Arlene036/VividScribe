# VividScribe


## Original Dataset

Using Valor-32k but downloaded from [VAST](https://github.com/TXH-mercury/VAST?tab=readme-ov-file).

Downloaded here and choose `anotation/valor32k`
[Google Drive](https://drive.google.com/file/d/1bOLUbbnPTgUp_Nc0PgORKC-174CwgwPm/view)


## Sample from Valor-32k for Evaluation

Details could be seen in `data_processing/dataset_sample.ipynb`.

In summary, 120 video clips were sampled, including:

- 60 non-verbal clips
- 60 verbal clips (divided into 20 light-verbal, 20 mid-verbal, and 20 heavy-verbal clips)

Sampled json dataset could be seen in `data/test120/sample_v_nv_test120_new.json`

Raw video clips and extracted data are available at [test120_GoogleDrive](https://drive.google.com/drive/folders/1DOeMn5LxjNFtlSTV5frrLSOHMJ3eNCO_)

> **Note:** There are four of the clips that cannot be processed by frame and audio wave extractor. They are skipped in the evaluation.
>
> p_o6NQX7lmE_0.000_10.000.wav (no subtitle)
> 
> xJ-6ewqMyxY_410.000_420.000.wav (have subtitle)
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

```
python data_processing/download_clips.py 'data/sample_v_nv_test120_new.json' 'data/test120/raw_video'
```

### Extract Video Frames & Audio Waves

```
python data_processing/dowload_clips.py
```

## Evaluation
Before running the evaluation, make sure to prepare the captions in the right json format as below:
```
{
    "annotations": [
        {
            "video_id": "video_id1",
            "caption": "caption1"
        },
        {
            "video_id": "video_id2",
            "caption": "caption2"
        },
        ...
    ]
}
```
Compute Metrics
```
python evaluation/evaluation_caption.py --generated_cap_path <generated_captions.json> --output_dir <output_results.json>
```
