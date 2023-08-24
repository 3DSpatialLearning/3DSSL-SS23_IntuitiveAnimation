# Intuitive speech driven Face animation based on [VOCA](https://github.com/TimoBolkart/voca)

This repository delivers two different usable approaches. On the one hand side you can retrain VOCA based on the RAVDESS data set on the other hand you can merge emotional blendshapes on to the output of the voca model. Both approaches deliver reasonable results, VOCA retrained on RAVDESS needs more fine tuning and delivers a result that can display emotions but also deliveres some bad formed or jumping mesh. The blendshape approach works well in context of displaying emotions but is very static and not capable of predicting a real humans behavior or face movement.

## Retrain VOCA on the RAVDESS Dataset
we managed to convert the RAVDESS Data into the right format for training the voca model. If you want to train it on your own you can downloading the following link ([click here]()) and follow the given ReadMe on where to store the extracted files.They can not be part of this repository since they are far to big to handle for git.
If you managed to download the files and you stored them in the right folders, please follow this step by step Instruction for training and execution:

### Training
1. Install Anaconda. Due to the fact that VOCA is based on specific tensorflow versions that need specific dependencies we used anaconda to manage this problem:

2. execute:
`conda env create -f enviornment.yml`
This installs all needed libraries and makes it possible to run the code

3. execute:
`conda activate voca_gpu`
This activates the needed virtual enviornment

4. execute:
`python3 run_training.py``


### Execution
The trainined model is saved in the 'gstep_1901.pb'. This can then be used for execution. You can either use your own audio or one of the provided ones using the following command:

`python run_voca.py --tf_model_fname './model/gstep_1901.model' --ds_fname './ds_graph/output_graph.pb' --audio_fname './audio/test_sentence.wav' --template_fname './template/FLAME_sample.ply' --condition_idx 3 --out_path './animation_output'`
