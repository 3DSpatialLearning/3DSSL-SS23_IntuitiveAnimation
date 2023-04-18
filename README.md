# 3DSSL-SS23_IntuitiveAnimation

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
