import json
import os
import sys
import yt_dlp
import subprocess

def check_and_download(video_id, output_dir, cookie_file=None):
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

def main(json_file, output_dir, cookie_file=None):
    with open(json_file) as file:
        sample_data = json.load(file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print('>>>>> Start downloading video clips <<<<<')
    for video in sample_data:
        check_and_download(video['video_id'], output_dir, cookie_file)
    print('>>>>> Finish downloading video clips <<<<<')

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python script.py <json_file> <output_dir> [cookie_file]")
        sys.exit(1)
        
    json_file = sys.argv[1]
    output_dir = sys.argv[2]

    cookie_file = sys.argv[3] if len(sys.argv) == 4 else None

    main(json_file, output_dir, cookie_file)
