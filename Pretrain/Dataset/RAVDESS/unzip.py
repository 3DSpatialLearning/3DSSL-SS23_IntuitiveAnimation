import os
import zipfile

folder_path = '/home/yuxinguo/data/RAVDESS'  

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    print("unziping:", file_path)

    if zipfile.is_zipfile(file_path):
        extract_folder = os.path.splitext(file_path)[0]
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        print("done:", file_name)