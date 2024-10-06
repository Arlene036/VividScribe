import os
import json
import librosa
import torch
import audiocap.models
import transformers
from tqdm import tqdm

def generate_captions_for_audio_folder(input_folder, output_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
    model = audiocap.WhisperForAudioCaptioning.from_pretrained(checkpoint).to(device)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)

    style_prefix = "clotho > caption: "
    style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels.to(device)

    annotations = []

    model.eval()

    wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    with torch.no_grad():
        for file_name in tqdm(wav_files, desc="Processing audio files", unit="file"):  # tqdm用于显示进度条
            file_path = os.path.join(input_folder, file_name)
            video_id = os.path.splitext(file_name)[0]

            audio, sampling_rate = librosa.load(file_path, sr=feature_extractor.sampling_rate)
            features = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

            outputs = model.generate(
                    inputs=features,
                    forced_ac_decoder_ids=style_prefix_tokens,
                    max_length=1000,
                )
            
            caption = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            caption = caption.replace(style_prefix, "").strip()
            
            annotations.append({
                "video_id": video_id,
                "caption": caption
            })

    output_dir = os.path.dirname(output_json)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_data = {"annotations": annotations}
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Captions saved to {output_json}")

if __name__ == '__main__':
    input_folder = "/root/VividScribe/data/test120/extracted_data/audio_22050hz/"
    output_json = "/root/VividScribe/inference_results/output_captions.json"
    generate_captions_for_audio_folder(input_folder, output_json)