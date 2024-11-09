# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type audio_cap \
#     --generated_cap_path 'output/unimodal_whisper_small_audiocap.json' \
#     --output_dir 'results_intrinsic/unimodal_whisper_small_audiocap_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_whisper_small_audiocap_detailed_scores.csv'


python evaluation/eval_intrinsic_visual_audio.py \
    --annotation_type object_labels \
    --generated_cap_path 'output/unimodal_timesformer.json' \
    --output_dir 'results_intrinsic/unimodal_timesformer_results.json' \
    --individual_scores_path 'results_intrinsic/unimodal_timesformer_detailed_scores.csv'