# !pip install --upgrade -q accelerate bitsandbytes
# !pip install git+https://github.com/huggingface/transformers.git
# # we need av to be able to read the video
# !pip install -q av

from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
import av
import numpy as np
import os

SKIPPED_IDS = [
    "p_o6NQX7lmE_0.000_10.000.mp4", 
    "xJ-6ewqMyxY_410.000_420.000.mp4", 
    "niJg7Q1XLyU_50.000_60.000.mp4", 
    "wj-gglKQ3KI_30.000_40.000.mp4"]


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def process_video(video_path, fps=4):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    video_duration = video_stream.duration * video_stream.time_base  
    original_fps = video_stream.average_rate  
    
    total_frames = int(video_duration * original_fps) 
    target_frame_count = int(video_duration * fps)  

    indices = np.linspace(0, total_frames - 1, target_frame_count).astype(int)
    clip = read_video_pyav(container, indices)
    return clip

def process_all_videos_in_folder(folder_path, fps=4):
    video_clips = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {filename}")
            clip = process_video(video_path, fps=fps)
            video_clips[filename] = clip
    return video_clips


def set_prompt(audio_caption):
    PROMPT = f"""
    Given the video and the audio captioning of this video, describe this video.
    Audio captioning is as follows:
    {audio_caption}
    """

    conversation = [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": PROMPT},
                  {"type": "video"},
                  ],
          },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt


def generate(output_file, video_clips, audio_cap_dict):
    # iterate with video_clips
    generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}
    # output_file = '/content/output/generation.json'
    # make dir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for video_name, video_clip in video_clips.items():
        if video_name in SKIPPED_IDS:
            continue
        video_name = video_name.split('.mp4')[0]
        prompt = set_prompt(audio_cap_dict[video_name])
        # we need to call the processor to tokenize the prompt and get pixel_values for videos
        inputs = processor([prompt], videos=[video_clip], padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, **generate_kwargs)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        # append to the json file
        with open(output_file, 'a') as f:
            f.write(json.dumps({'video_name': video_name, 'generated_text': generated_text[0].split('ASSISTANT: ')[-1]}) + '\n')


if __name__ == '__main__':
    # load the audio captioning
    with open('output/unimodal_whisper_small_audiocap.json', 'r') as f:
        audio_cap = json.load(f)
    udio_cap = audio_cap['annotations']
    audio_cap_dict = {item['video_id']: item['caption'] for item in audio_cap}

    output_file = 'output/multimodal_llava_video_whisper.json'
    video_clips = process_all_videos_in_folder('data/mix120/raw_video', fps=4)