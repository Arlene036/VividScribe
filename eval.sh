python evaluation/evaluation_caption.py \
    --generated_cap_path 'output/unimodal_whisper_audiocap.json' \
    --output_dir 'results/unimodal_whisper_audiocap.json'

python evaluation/evaluation_caption.py \
    --generated_cap_path 'output/unimodal_gpt-4o.json' \
    --output_dir 'results/unimodal_gpt-4o.json'

python evaluation/evaluation_caption.py \
    --generated_cap_path 'output/unimodal_timesformer.json' \
    --output_dir 'results/unimodal_timesformer.json'