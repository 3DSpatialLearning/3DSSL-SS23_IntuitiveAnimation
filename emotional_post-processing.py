import os
import glob
from speechbrain.pretrained.interfaces import foreign_class


audiodata = sorted(glob.glob(os.path.join("training_data/audio/Actor_01", '*.wav')))

# classifier to extract emotion from audio
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

for count, audio in enumerate(audiodata):

    file_name = audio.split("/")[-1].split(".")[0]
    # extract emotion
    out_prob, score, index, text_lab = classifier.classify_file(audio)
    emotion_temp = os.path.join("offset", text_lab[0] + "_template.obj")
    # add emotion onto all targets
    with open(emotion_temp) as f:
        mesh0 = f.read().split("\n")
        folder = os.path.join("voca_out", file_name, "meshes")
        targets = sorted(glob.glob(os.path.join(folder, '*.obj')))
        for target in targets: 
            with open(target) as file:
                object = target.split("/")[-1]
                mesh1 = file.read().split("\n")
                save_dir = os.path.join("emotional_targets", file_name)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, object), mode='w') as write:
                    for (one, two) in zip(mesh0, mesh1):
                        try:
                            label, x, y, z = one.split(" ")
                            label2, x2, y2, z2 = two.split(" ")
                        except:
                            break        
                        if label == "v":
                            line = label + " " + str(float(x) + float(x2)) + " " + str(float(y) + float(y2)) + " " + str(float(z) + float(z2)) + "\n"
                        else:
                            line = label + " " + x + " " + y + " " + z + "\n"
                        write.write(line)