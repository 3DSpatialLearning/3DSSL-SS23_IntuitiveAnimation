# 3DSSL-SS23_IntuitiveAnimation

## EmoFormer: A Transformer-based Audio Driven Face Animation Network with Emotion
<div align=center><img src="images/Change_Emotion.gif" alt="drawing" width="90%"/><p>EmoFormer prediction of a single audio recording with manually added emotion</p></div>

The EmoFormer uses a transformer-based encoder decoder structure to achieve the audio driven face animation with emotion. The output mesh is based on FLAME model.

The network are based on a encoder-decoder structure. The input audio sequence are firstly processed with Wav2Vec feature extractor and its content and emotion features are then extracted with content encoder and emotion encoder respectively. The content feature is a time sequence while the emotion feature is time invariant and extracted from the whole audio sequence. These features are then feed into a transformer decoder with one decoder layer. The final output of the network is a tracked mesh based on FLAME.

### Requirements
- Install the required python packages in `requirements.txt`
- ffmpeg
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)


## First attempt: Re-fitting landmarks with FLAME
<div align=center><img src="images/transform.png" alt="drawing" width="90%"/></div>

This approch fits FLAME model to 51 landmarks of human face. User can adjust landmarks in pyVista viewer via dragging and see a deformed face.

Performance is poor since it takes a lot of time to fit FLAME model to landmarks. Won't be an option for practical use.

### Requirements
The requirements are exactly the same with toy task except pyTorch.

### Running
Before running this demo, please follow the installation step and download models as suggested in toy task below.

Then make sure that you are in "refit" folder and run
```
python refit.py
```

## Interactive FLAME model viewer (Toy Task)

<div align=center><img src="images/viewer.png" alt="drawing" width="50%"/></div>

Here's an interavtive FLAME model viewer created with PyVista. Several slider bars are added to allow user to adjust parameters of FLAME model. The code uses Python 3.8 and is tested on PyTorch 2.0.0+cu117.

### Installation
You can create a model folder in the root folder of this project and install all requirements of toyTask by running following commands from the terminal.
```
mkdir model
cd toyTask
pip install -r requirements.txt
```

### Download models
- Download FLAME model from <a href="http://flame.is.tue.mpg.de/" rel="nofollow">here</a>. Copy the downloaded model inside the model folder.
- Download Landmark embedings from <a href="https://github.com/soubhiksanyal/RingNet/tree/master/flame_model">RingNet Project</a>. Copy them inside the model folder.

### Running viewer
Make sure that you are in toyTask folder, then simply run the following command from the terminal.
```
python viewer.py
```
