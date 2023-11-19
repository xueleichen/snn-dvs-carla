This is the code for my BU EC500 (Spring 2023) course project report **Autonomous Driving using Spiking Neural Networks on Dynamic Vision Sensor Data: A Case Study of Traffic Light Change Detection **.

You can find the project report [here](https://arxiv.org/abs/2311.09225).

If you find the code useful, please star this repo. Thank you!

### Setup
Run the following commands one by one to create a conda environment and install related libraries
```bash
conda create -n snn python=3.8
```
```bash
conda activate snn
```
```bash
pip install spikingjelly
```
```bash
conda install pandas
```
Run the following commands to create a copy of this repository in your local system.
```bash
git clone https://github.com/xueleichen/snn-dvs-carla.git
```
```bash
cd snn-dvs-carla
```

The code has been tested with:

- Ubuntu 20.04.4 LTS
- CUDA Tookit 11.1
- CUDNN 8.4.0
- Nvidia RTX 3060

Note: There is a **environment.yml** file in this repository, which records the library versions that I used. You can also quickly set up using the following one-line command:
```bash
conda env create -f environment.yml
```

### Data and Files
DVS data are stored in the **./data** folder. RGB data are stored in the **./rgb_data** folder.

**./*.txt** files store training testing data path and labels.

### Training

I designed three configurations for experiments in this project: SNN with DVS data, CNN with DVS data, and CNN with RGB data.

When you run the following three commands, you will get three result curves in my report.
```bash
python main_SNN.py
```
```bash
python main_CNN.py
```
```bash
python main_CNN_Image.py
```

