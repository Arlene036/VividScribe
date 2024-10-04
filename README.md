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
python evaluate.py --generated_cap_path <generated_captions.json> --true_cap_path <true_captions.json> --output_dir <output_results.json>
```
