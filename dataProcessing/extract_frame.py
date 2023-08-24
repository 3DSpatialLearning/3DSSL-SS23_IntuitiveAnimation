import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize variables
    frame_count = 0
    success = True

    # Read frames until the video ends
    while success:
        # Read the next frame
        success, frame = video.read()

        if success:
            # Save the frame as an image
            output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
            cv2.imwrite(output_path, frame)

            # Increment the frame count
            frame_count += 1

    # Release the video file
    video.release()

# Example usage
video_path = "/mnt/hdd/datasets/gigaMove/videos/"
output_folder = "/mnt/hdd/datasets/gigaMove/frames/"

counter = 0
for root, dirs, files in os.walk(video_path):
    for file in files:
        if file.endswith(".mp4"):
            file_name = os.path.splitext(file)[0]
            output = os.path.join(output_folder, file_name)
            if not os.path.exists(output):
                counter += 1
                video = os.path.join(video_path, file)
                output = os.path.join(output_folder, file_name)
                print(f"{counter} / 1654: processing: {video}, saving to {output}")
                extract_frames(video, output)