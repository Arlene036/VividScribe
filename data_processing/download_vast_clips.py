#download vast
import json
import os
import sys
import yt_dlp
import subprocess

def check_and_download_valor(video_id, output_dir, cookie_file=None):
    parts = video_id.split('_')
    youtube_id = '_'.join(parts[0:-2])

    if os.path.exists(f'{output_dir}/{video_id}.mp4'):
        print(f'Video clip {video_id} already exists.')
        return True

    start_time = float(parts[-2])
    end_time = float(parts[-1])
    youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
    
    if cookie_file:
        ydl_opts = {
            'format': 'mp4',  
            'outtmpl': f'{output_dir}/{youtube_id}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  
            }],
            'cookiefile': cookie_file
        }
    else:
        ydl_opts = {
            'format': 'mp4',  
            'outtmpl': f'{output_dir}/{youtube_id}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  
            }]
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        if not os.path.exists(f'{output_dir}/{youtube_id}.mp4'):
            print(f'Video {video_id} is not mp4')
            return False
        
        input_file = f'{output_dir}/{youtube_id}.mp4'
        output_file = f'{output_dir}/{video_id}.mp4'
        duration = end_time - start_time
        
        subprocess.run([
            'ffmpeg', '-ss', str(start_time), '-i', input_file, '-t', str(duration), '-c', 'copy', output_file
        ])
        
        print(f'Video {output_file} downloaded and trimmed successfully.')
        os.remove(input_file) # delete the original video, only keep the trimmed one
        return True
    except Exception as e:
        print(f'Video {video_id} fails downloading: {e}')
        return False


def check_and_download_vast(clip_id, clip_span, url, output_dir, cookie_file=None):
    # Parse the clip_span to get start and end times
    start_time_str, end_time_str = clip_span
    start_time = convert_time_to_seconds(start_time_str)
    end_time = convert_time_to_seconds(end_time_str)
    duration = end_time - start_time

    video_id = clip_id.split('.')[0]  # Extract YouTube ID from clip_id

    # Check if the trimmed video already exists
    if os.path.exists(f'{output_dir}/{clip_id}.mp4'):
        print(f'Video clip {clip_id} already exists.')
        return True

    youtube_url = url

    # Set up yt_dlp options
    if cookie_file:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': f'{output_dir}/{video_id}_og.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'cookiefile': cookie_file
        }
    else:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': f'{output_dir}/{video_id}_og.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
        }

    try:
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if not os.path.exists(f'{output_dir}/{video_id}_og.mp4'):
            print(f'Video {clip_id} is not mp4')
            return False

        input_file = f'{output_dir}/{video_id}_og.mp4'
        output_file = f'{output_dir}/{clip_id}.mp4'

        # Trim the video using ffmpeg
        subprocess.run([
            'ffmpeg', '-ss', str(start_time), '-i', input_file, '-t', str(duration), '-c', 'copy', output_file
        ], check=True)

        print(f'Video {output_file} downloaded and trimmed successfully.')
        os.remove(input_file)  # delete the original video, only keep the trimmed one
        return True
    except Exception as e:
        print(f'Video {clip_id} fails downloading: {e}')
        return False

def convert_time_to_seconds(time_str):
    """Convert a time string (hh:mm:ss.xxx) to seconds."""
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def main(json_file, output_dir, cookie_file=None):
    with open(json_file) as file:
        sample_data = json.load(file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('>>>>> Start downloading video clips <<<<<')
    for video in sample_data:
        if "clip_span" not in video:
            print(f'>>>>> Downloading Valor video {video["video_id"]} <<<<<')
            check_and_download_valor(video['video_id'], output_dir, cookie_file)
        
        else:
            print(f'>>>>> Downloading VAST video {video["video_id"]} <<<<<')   
            check_and_download_vast(video['video_id'], video['clip_span'], video['url'], output_dir, cookie_file)
    print('>>>>> Finish downloading video clips <<<<<')


json_file = "data/fewshot/fewshot_baseline.json"
output_dir = "data/fewshot/raw_video"
cookie_file = None

main(json_file, output_dir, cookie_file)