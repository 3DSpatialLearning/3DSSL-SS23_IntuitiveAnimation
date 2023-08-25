# [VOCA: Voice Operated Character Animation](https://voca.is.tue.mpg.de)

This is an extension of the official [VOCA](https://voca.is.tue.mpg.de) repository.

<p align="center"> 
<img src="gif/speech_driven_animation.gif">
</p>

VOCA is a simple and generic speech-driven facial animation framework that works across a range of identities. This codebase demonstrates how to synthesize realistic character animations given an arbitrary speech signal and a static character mesh. For details please see the scientific publication

```
Capture, Learning, and Synthesis of 3D Speaking Styles.
D. Cudeiro*, T. Bolkart*, C. Laidlaw, A. Ranjan, M. J. Black
Computer Vision and Pattern Recognition (CVPR), 2019
```

A pre-print of the publication can be found [here](
https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/510/paper_final.pdf).
You can also check out the [VOCA Blender Addon](https://github.com/SasageyoOrg/voca-blender)

## Video

See the demo video for more details and results.

[![VOCA](https://img.youtube.com/vi/XceCxf_GyW4/0.jpg)](https://youtu.be/XceCxf_GyW4)

While showing good approximations of the lip movements, no upper face movements and hence, no emotional expressions are achieved.
This approach aims to overcome this limitation by first applying post-processing in a first step and retraining the pipeline on a generated dataset in a second step.
Please note that retraining on emotional data does not work for allowing for emotional expressions. This work is mainly made public in order to make future work on the official [VOCA](https://voca.is.tue.mpg.de) repository easier.

## Set-up
1. Install Anaconda. Due to the fact that VOCA is based on specific tensorflow versions that need specific dependencies we used anaconda to manage this problem:

2. execute:
```
conda env create -f enviornment.yml
```
This installs all needed libraries and makes it possible to run the code

3. execute:
```
conda activate voca_gpu
```
This activates the needed virtual enviornment

Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh) within the virtual environment.

## Post-Processing

Please note that you need a second environment for this as the dependencies of voca are slightly messed up and hence, are not combinable with speechbrain.

Set up and activate virtual environment:
```
mkdir <your_home_dir>/.virtualenvs
python3 -m venv <your_home_dir>/.virtualenvs/emovoca
pip install speechbrain
source <your_home_dir>/.virtualenvs/emovoca/bin/activate
```

To post-process the voca output to which you want to add emotions, define the location of the audio input and the voca output and run:
```
python emotional_post-processing.py
```

## Training

To overcome the limitations of post-processing, a dataset is generated to train a model that is capable of reflecting emotions.

### Data Generation

In order to generate the dataset, download the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) data and produce the voca animations, after defining the location of the data, by running:
```
python voca_on_RAVDESS_audio.py
```
Plese note that the default templates provided by voca are used but can be changes according to your preference.

To generate the targets, apply the post-processing to generate the targets.
```
python emotional_post-processing.py
```

## Retrain VOCA on the RAVDESS Dataset
we managed to convert the RAVDESS Data into the right format for training the voca model. If you want to train it on your own you can downloading the following link ([click here]()) and follow the given ReadMe on where to store the extracted files.They can not be part of this repository since they are far to big to handle for git.
If you managed to download the files and you stored them in the right folders, please follow this step by step Instruction for training and execution:

### Training
If not already done follow steps 1-3, if you already sat up the enviornment you can start with step 4:

1. Install Anaconda. Due to the fact that VOCA is based on specific tensorflow versions that need specific dependencies we used anaconda to manage this problem:

2. execute:
```
conda env create -f enviornment.yml
```
This installs all needed libraries and makes it possible to run the code

3. execute:
```
conda activate voca_gpu
```
This activates the needed virtual enviornment

4. execute:
```
python3 run_training.py
```


### Execution
The trainined model is saved in the 'gstep_1901.pb'. This can then be used for execution. You can either use your own audio or one of the provided ones using the following command:

```
python run_voca.py --tf_model_fname './model/gstep_1901.model' --ds_fname './ds_graph/output_graph.pb' --audio_fname './audio/test_sentence.wav' --template_fname './template/FLAME_sample.ply' --condition_idx 3 --out_path './animation_output'
```

## License

Free for non-commercial and scientific research purposes. By using this code, you acknowledge that you have read the license terms (https://voca.is.tue.mpg.de/license.html), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code.


## Referencing VOCA

If you find this code useful for your research, or you use results generated by VOCA in your research, please cite following paper:

@article{VOCA2019,
    title = {Capture, Learning, and Synthesis of {3D} Speaking Styles},
    author = {Cudeiro, Daniel and Bolkart, Timo and Laidlaw, Cassidy and Ranjan, Anurag and Black, Michael},
    journal = {Computer Vision and Pattern Recognition (CVPR)},
    pages = {10101--10111},
    year = {2019}
    url = {http://voca.is.tue.mpg.de/}
}
