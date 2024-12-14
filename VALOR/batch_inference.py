import json
import os
import torch
import torchaudio
from model.pretrain import VALOR
from torchvision.transforms.transforms import *
from torchvision import transforms
from easydict import EasyDict as edict
from PIL import Image
from model.bert_tokenizer import BertTokenizer
from tqdm import tqdm

def load_from_pretrained_dir(pretrain_dir):
    checkpoint_dir = os.path.join(pretrain_dir, 'ckpt')
    checkpoint_ls = [i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
    checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
    checkpoint_ls.sort()    
    step = checkpoint_ls[-1]
    
    checkpoint_name = f'model_step_{step}.pt'
    ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}

    pretrain_cfg = edict(json.load(open(os.path.join(pretrain_dir,'log','hps.json'))))
    
    if 'video_frame_embedding' in checkpoint:
        checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num:] = checkpoint['video_frame_embedding'][:,pretrain_cfg.video_sample_num-1].clone()
    if 'audio_frame_embedding' in checkpoint:
        checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num:] = checkpoint['audio_frame_embedding'][:,pretrain_cfg.audio_sample_num-1].clone()
    
    return checkpoint, pretrain_cfg

def process_video_frames(frames_dir, sample_num=8, pretrain_cfg=None):
    if pretrain_cfg.video_encoder_type.startswith('clip'):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
        Resize((224,224)),
        Normalize(mean,std)
    ])

    frames = sorted(os.listdir(frames_dir))
    # frames_split = [frames[i:i + len(frames)//sample_num] for i in range(0, len(frames), len(frames)//sample_num)]
    
    # video_pixels = []
    # for frame_group in frames_split:
    #     mid_frame = frame_group[(len(frame_group)+1)//2-1]
    #     frame = Image.open(os.path.join(frames_dir, mid_frame))
    #     frame = transforms.ToTensor()(frame)
    #     video_pixels.append(frame.unsqueeze(0))
    
    # video_pixels = torch.cat(video_pixels, dim=0)
    # video_pixels = test_transforms(video_pixels)
    frames_splited = split(frames,sample_num)    
    video_pixels = [] 
    sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
    for i in range(sample_num):
        frame = Image.open(os.path.join(frames_dir,sample_idx[i]))
        frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
        video_pixels.append(frame.unsqueeze(0))
    video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW

    video_pixels = test_transforms(video_pixels)     

    return video_pixels

def get_model_attr(model, attr_name):
    if hasattr(model, 'module') and hasattr(model.module, attr_name):
        return getattr(model.module, attr_name)

    elif hasattr(model, attr_name):
        return getattr(model, attr_name)

    else:
        return ValueError

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]



def process_audio(audio_path, target_length=512, sample_num=1, pretrain_cfg=None):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform - waveform.mean()
    
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, 
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=pretrain_cfg.audio_melbins,
        dither=0.0,
        frame_shift=pretrain_cfg.audio_frame_shift
    )

    # Padding and splitting
    output_slices = []
    pad_len = target_length - fbank.shape[0] % target_length
    fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
    total_slice_num = fbank.shape[0] // target_length
    total_slice_num = list(range(total_slice_num))
    total_slice_num = split(total_slice_num, sample_num)

    sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]


    for i in sample_idx:
        output_slices.append(fbank[i*target_length : (i+1)*target_length])
    fbank = torch.stack(output_slices,dim=0).permute(0,2,1)   
    fbank = (fbank - pretrain_cfg.audio_mean) / (pretrain_cfg.audio_std * 2)

    return fbank.unsqueeze(0).cuda()

def main():
    # Paths
    MODEL_DIR = "/home/ubuntu/VividScribe/VALOR/output/VQA-11777-lr2e-5-bs64-epoch20-frozen-bias"  # Replace with actual path
    OUTPUT_FILE = "/home/ubuntu/VividScribe/output/gate_valor12141113.json"
    
    # Load model
    checkpoint, pretrain_cfg = load_from_pretrained_dir(MODEL_DIR)
    model = VALOR.from_pretrained(pretrain_cfg, checkpoint)
    model.eval().cuda()
    
    # Load mix120 data
    with open("/home/ubuntu/VividScribe/data/mix120/mix120.json", "r") as f:
        mix120_data = json.load(f)
    
    # Load whisper captions for non-verbal videos
    with open("/home/ubuntu/VividScribe/output/unimodal_whisper_audiocap.json", "r") as f:
        whisper_captions = json.load(f)
        
    results = {"annotations": []}
    
    # Initialize tokenizer
    bert_tokenizer = BertTokenizer("./pretrained_weights/bert-base-uncased-vocab.txt")
    cls_token = bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    sep_token = bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    
    for item in tqdm(mix120_data):
        try:
            video_id = item['video_id']
            subtitle = item.get('subtitle', '')
            
            # Process frames
            frames_dir = os.path.join("/home/ubuntu/VividScribe/data/mix120/extracted_data/frames_fps1", video_id)
            video_pixels = process_video_frames(frames_dir, pretrain_cfg=pretrain_cfg)
            video_pixels = video_pixels.unsqueeze(0).cuda()
            
            # Process audio
            audio_path = os.path.join("/home/ubuntu/VividScribe/data/mix120/extracted_data/audio_22050hz", f"{video_id}.wav")
            fbank = process_audio(audio_path, pretrain_cfg=pretrain_cfg)
            
            tokenized_text = bert_tokenizer.tokenize('What is in the video?')
            txt_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

            txt_tokens = [cls_token] + txt_tokens + [sep_token]  

            txt_tokens = {'bert_tokens':torch.tensor(txt_tokens, dtype=torch.long).unsqueeze(0).cuda()}
            
            # Create batch
            batch = {
                'ids': None,
                'question_tokens': txt_tokens,
                'video_pixels': video_pixels,
                'audio_spectrograms': fbank,
                'ids_txt': None,
                'sample_num': [1]
            }
            
            # Get model prediction
            with torch.no_grad():
                evaluation_dict = model(batch, 'qa%tva', compute_loss=False)
                answers = evaluation_dict['generated_answers_t_va']
                answer = get_model_attr(model, 'decode_sequence')(answers.data)
                
                results["annotations"].append({
                    "video_id": video_id,
                    "caption": answer[0] if isinstance(answer, list) else answer
                })
            
            # Save intermediate results
            if len(results["annotations"]) % 5 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(results, f, indent=4)
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            continue
                
    
    # Save final results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()