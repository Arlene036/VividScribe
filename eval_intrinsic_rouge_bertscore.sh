# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type audio_cap \
#     --generated_cap_path 'output/unimodal_whisper_small_audiocap.json' \
#     --output_dir 'results_intrinsic/unimodal_whisper_small_audiocap_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_whisper_small_audiocap_detailed_scores.csv'

# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type audio_cap \
#     --generated_cap_path 'output/unimodal_whisper_tiny_audiocap.json' \
#     --output_dir 'results_intrinsic/unimodal_whisper_tiny_audiocap_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_whisper_tiny_audiocap_detailed_scores.csv'

# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type audio_cap \
#     --generated_cap_path 'output/unimodal_openai_whisper.json' \
#     --output_dir 'results_intrinsic/unimodal_openai_whisper_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_openai_whisper_detailed_scores.csv'


# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type object_labels \
#     --generated_cap_path 'output/unimodal_timesformer.json' \
#     --output_dir 'results_intrinsic/unimodal_timesformer_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_timesformer_detailed_scores.csv'

# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type object_labels \
#     --generated_cap_path 'output/unimodal_gpt-4o.json' \
#     --output_dir 'results_intrinsic/unimodal_gpt-4o_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_gpt-4o_detailed_scores.csv'

# python evaluation/eval_intrinsic_visual_audio.py \
#     --annotation_type object_labels \
#     --generated_cap_path 'output/unimodal_llavaNeXT.json' \
#     --output_dir 'results_intrinsic/unimodal_llavaNeXT_results.json' \
#     --individual_scores_path 'results_intrinsic/unimodal_llavaNeXT_detailed_scores.csv'

python evaluation/eval_intrinsic_visual_audio.py \
    --annotation_type object_labels \
    --generated_cap_path 'output/multimodal_gpt-4o_whisper.json' \
    --output_dir 'results_intrinsic/multimodal_gpt-4o_whisper_results.json' \
    --individual_scores_path 'results_intrinsic/multimodal_gpt-4o_whisper_detailed_scores.csv'

python evaluation/eval_intrinsic_visual_audio.py \
    --annotation_type object_labels \
    --generated_cap_path 'output/multimodal_timesformer_subtitle.json' \
    --output_dir 'results_intrinsic/multimodal_timesformer_subtitle_results.json' \
    --individual_scores_path 'results_intrinsic/multimodal_timesformer_subtitle_detailed_scores.csv'

python evaluation/eval_intrinsic_visual_audio.py \
    --annotation_type object_labels \
    --generated_cap_path 'output/multimodal_llava-whisper-ac.json' \
    --output_dir 'results_intrinsic/multimodal_llava-whisper-ac_results.json' \
    --individual_scores_path 'results_intrinsic/multimodal_llava-whisper-ac_detailed_scores.csv'

