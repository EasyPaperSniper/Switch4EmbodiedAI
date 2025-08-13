Only support single character for now as we use ROMP

## Installation

### Clone (with submodules)
```
git clone --recurse-submodules git@github.com:EasyPaperSniper/Switch4EmbodiedAI.git
cd Switch4EmbodiedAI
```

### Conda
```
conda create -n Switch4EAI python=3.10 Cython
conda activate Switch4EAI
```

### Install simple_romp
```
pip install --upgrade setuptools lap lapx
pip install simple_romp=1.1.3
```

### Main Package

In the directory with setup.py
```
pip install -e .
```

### For Third Party Installations
for simple_romp
```
cd third_party/ROMP/simple_romp
pip install -e .
```

for GMR
```
cd ..
cd ..
cd GMR
pip install -e .
```
```
conda install -c conda-forge libstdcxx-ng -y # (Conda) C++ stdlib fix
```

## Body Models
Download from the official sites and place here:
Switch4EmbodiedAI/utils/smpl_model_data/
```
|-- smpl_model_data
|   |-- SMPL_NEUTRAL.pkl
|   |-- SMPLX_NEUTRAL.pkl
|   |-- SMPLX_NEUTRAL.npz
```


<!-- +++++ cite from https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md +++++++
a. Meta data from this link. Please unzip it, then we get a folder named "smpl_model_data" b. SMPL model file (SMPL_NEUTRAL.pkl) from "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" in official website. Please unzip it and move the SMPL_NEUTRAL.pkl from extracted folder into the "smpl_model_data" folder.




git submodule init
git submodule update
git submodule update --init --recursive -->
