import cv2
# EMOCA test data:
# /home/yuxinguo/emoca/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4
# /home/yuxinguo/emoca/video_output/EMOCA_v2_lr_mse_20/processed_2023_May_27_15-54-11/82-25-854x480_affwild2/results/EMOCA_v2_lr_mse_20/video_geometry_detail_with_sound.mp4

# 
path = '/home/yuxinguo/emoca/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4'

# load video file
video = cv2.VideoCapture(path)

# get fps & resolution
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print
print("fps", fps)
print("res", frame_width, "x", frame_height)

video.release()