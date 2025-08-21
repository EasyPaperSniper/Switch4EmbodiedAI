Only support single character for now as we use ROMP
Repo still under refacting

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


# Citation

If you find our code useful, please consider citing our papers:
```bibtex
@article{li2025switch4eai,
      title={Switch4EAI: Leveraging Console Game Platform for Benchmarking Robotic Athletics}, 
      author={Tianyu Li and Jeonghwan Kim and Wontaek Kim and Donghoon Baek and Seungeun Rho and Sehoon Ha},
      year= {2025},
      journal= {arXiv preprint arXiv:2508.13444}
}
```

# Acknowledgement
Motion tracking module built upon [ROMP](https://github.com/Arthur151/ROMP). Motion retargeting module built upon [GMR](https://github.com/YanjieZe/GMR). Module tested on [GMT](https://gmt-humanoid.github.io/).
