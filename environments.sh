# conda create --yes -n ftc python=3.8
python -m pip install numpy scipy statsmodels protobuf==3.20.* ftfy regex tqdm seaborn
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install timm==0.4.12 deepspeed==0.4.0
python -m pip install git+https://github.com/openai/CLIP.git
