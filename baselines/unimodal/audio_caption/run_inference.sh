#!/bin/bash

INPUT_FOLDER="/content/VividScribe/baselines/unimodal/audio_caption/extracted_data/audio_22050hz"
OUTPUT_BASE="/content/VividScribe/output/unimodal_whisper-small-audio-captioning"


mkdir -p "$OUTPUT_BASE"

temperatures=(0 0.5 0.7)
num_beams=(1 3 5 7)

echo "Begin hyperparameter search: $(date)" | tee "$OUTPUT_BASE/search_log.txt"


for temp in "${temperatures[@]}"; do
    for beam in "${num_beams[@]}"; do
        output_file="$OUTPUT_BASE/temp${temp}_beam${beam}.json"
        
        echo "Runing: temperature=$temp, num_beams=$beam" | tee -a "$OUTPUT_BASE/search_log.txt"
        

        python inference.py \
            --input_folder "$INPUT_FOLDER" \
            --output_json "$output_file" \
            --temperature "$temp" \
            --num_beams "$beam" 2>&1 | tee -a "$OUTPUT_BASE/search_log.txt"
        
        echo "Completesd: $output_file" | tee -a "$OUTPUT_BASE/search_log.txt"
        echo "----------------------------------------" | tee -a "$OUTPUT_BASE/search_log.txt"
    done
done

echo "complete: $(date)" | tee -a "$OUTPUT_BASE/search_log.txt" 