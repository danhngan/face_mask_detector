#!/bin/bash#
conda create -n detector python=3.7 numpy pandas matplotlib
source activate detector
pip install --ignore-installed --upgrade tensorflow==2.5.0
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
mkdir face_mask_detector
cd face_mask_detector
mkdir workspace
mkdir workspace/models
mkdir workspace/app
git clone https://github.com/tensorflow/models workspace/models
git clone https://github.com/ workspace/app
## apt-get install protobuf-compiler
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protoc-3.20.0-linux-x86_64.zip
unzip protoc-3.20.0-linux-x86_64.zip
export PATH=/home/ec2-user/face_mask_detector/bin:$PATH
cd workspace/models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
sudo yum group install "Development Tools"
python -m pip install .
python workspace/models/research/object_detection/builders/model_builder_tf2_test.py
