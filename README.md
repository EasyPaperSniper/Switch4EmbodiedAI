Only support single character for now as we use ROMP



git clone --recurse-submodules git@github.com:EasyPaperSniper/Switch4EmbodiedAI.git



cd Switch4EmbodiedAI
pip install -e .

cd third_party/ROMP/simple_romp
pip install -e .



+++++ cite from https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md +++++++
a. Meta data from this link. Please unzip it, then we get a folder named "smpl_model_data" b. SMPL model file (SMPL_NEUTRAL.pkl) from "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" in official website. Please unzip it and move the SMPL_NEUTRAL.pkl from extracted folder into the "smpl_model_data" folder.




git submodule init
git submodule update
git submodule update --init --recursive