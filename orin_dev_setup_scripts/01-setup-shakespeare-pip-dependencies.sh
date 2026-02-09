#!/bin/bash

sudo apt update
sudo apt install python3.10-venv -y
cd ..
python3 -m venv ReaLLM
source venv/bin/activate
sudo apt install python-pip -y
pip install sentencepiece tiktoken tqdm rich torchinfo plotly seaborn tensorboard pyyaml
pip install "triton==3.5.1"
echo "After you can run bash data/shakespeare_char/get_dataset.sh"
