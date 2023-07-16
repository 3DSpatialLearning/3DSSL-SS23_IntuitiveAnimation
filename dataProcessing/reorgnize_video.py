import os
import shutil
from moviepy.editor import VideoFileClip

def organize_mp4_files(root_directory, dist_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                group_dir = os.path.basename(os.path.dirname(root))
                sen_dir = os.path.basename(os.path.dirname(file_path))
                new_file_name = f"{group_dir}_{sen_dir}_{os.path.splitext(file)[0]}.mp4"
                new_folder_path = dist_directory
                new_file_path = os.path.join(new_folder_path, new_file_name)
                
                # Create the new "videos" folder if it doesn't exist
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                
                # Move the file to the new folder with the modified name
                shutil.move(file_path, new_file_path)
                print(f"Moved: {file_path} -> {new_file_path}")

def extract_audio_from_mp4(input_file, output_file):
    is_audio = True
    video = VideoFileClip(input_file)
    audio = video.audio
    if(audio != None):
        audio.write_audiofile(output_file)
        audio.close()
    else:
        is_audio = False
    return is_audio
# Specify the root directory where you want to organize the MP4 files
# root_directory = "/home/haifanzhang/datasets/gigaMove/sentences_front"
# dist_directory = "/home/haifanzhang/datasets/gigaMove/videos"

# Call the function to organize the MP4 files
# organize_mp4_files(root_directory, dist_directory)

# mp4_path = "/home/haifanzhang/datasets/gigaMove/videos"
# wav_path = "/home/haifanzhang/datasets/gigaMove/audios"

# if not os.path.exists(wav_path):
#     os.makedirs(wav_path)

# no_audio = []
# for file_name in os.listdir(mp4_path):
#     if file_name.endswith(".mp4"):
#         # Construct the input and output file paths
#         input_file = os.path.join(mp4_path, file_name)
#         output_file = os.path.join(wav_path, os.path.splitext(file_name)[0] + ".wav")
        
#         # Extract audio from MP4 and save as WAV
#         print(f"processing: {input_file}")
#         is_audio = extract_audio_from_mp4(input_file, output_file)
#         if is_audio == False:
#             no_audio.append(input_file)
#         print(f"Audio extracted: {input_file} -> {output_file}")
        
# print(f"Done, {len(no_audio)} videos have no audio: {no_audio}")

no_audio = ['/home/haifanzhang/datasets/gigaMove/videos/121_SEN-10-port_strong_smokey_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-05-glow_eyes_sweet_girl_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-01-cramp_small_danger_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-02-same_phrase_thirty_times_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-10-port_strong_smokey_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-01-cramp_small_danger_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-07-fond_note_fried_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-04-two_plus_seven_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/287_SEN-10-port_strong_smokey_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-04-two_plus_seven_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-07-fond_note_fried_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-08-clothes_and_lodging_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-03-pluck_bright_rose_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-06-problems_wise_chief_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/265_SEN-10-port_strong_smokey_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-02-same_phrase_thirty_times_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-09-frown_events_bad_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-09-frown_events_bad_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-05-glow_eyes_sweet_girl_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/125_SEN-06-problems_wise_chief_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-10-port_strong_smokey_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-03-pluck_bright_rose_cam_222200037.mp4', '/home/haifanzhang/datasets/gigaMove/videos/096_SEN-08-clothes_and_lodging_cam_222200037.mp4']
for file in no_audio:
    fileName = f"{os.path.splitext(file)[0].split('/')[-1]}.mp4"
    newFilePath = os.path.join("/home/haifanzhang/datasets/gigaMove/videos/no_audio", fileName)
    print(newFilePath)
    shutil.move(file, newFilePath)