# under emoca/

root="/home/yuxinguo/data/RAVDESS"

for folder in "$root"/*; do

    if [[ "$folder" == *Video_Speech_Actor_2* ]]; then
        
        for actor in "$folder"/*; do
            
            for video in "$actor"/*; do
                echo "current video: $video"

                new_actor=${video/RAVDESS/RAVDESS_gen}
                new_file=${new_actor%.mp4}
                echo "new path: $new_file"

                python gdl_apps/EMOCA/demos/test_emoca_on_video.py --input_video "$video" --output_folder "$new_file" --model_name EMOCA_v2_lr_mse_20 

            done

        done

    fi

done