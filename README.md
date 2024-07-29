TinyChirp
=====================

Bird Song Recognition Using TinyML Models on Low-power Wireless Acoustic Sensors

__Under Reconstruction...__

# Get Source Code

It is important to clone the submodules along, with `--recursive` option.

```
git clone --recursive git@github.com:TinyPART/TinyBirdSounds.git
```

Or open the git shell after clone, and type the following command.

```
git submodule init
git submodule update
```

# Prequisites

This Repo is tested under:

- Linux Mint 21.1 (5.15.0-58-generic)
- Python 3.10.8

## Prepare for datasets

please refer to `datasets/README.md`
## Prepare for model training and evaluation
(under construction...)

## (Optional) Prepare for measuring resource consumption

1. Get __RIOT__: please refer to the [RIOT's repo](https://github.com/RIOT-OS/RIOT).
2. Get __RIOT_ML__: please refer to the [RIOT-ML repo](https://github.com/TinyPART/RIOT-ML).
3. Install ARM-Toolchain: please refer to https://doc.riot-os.org/getting-started.html#the-build-system



# Pilot Study

To reproduce the results, please refer to `pilot/pilot_analysis.ipynb`.

Also it shows the performance of *baseline* and *power-saving*. 

# Evaluation of TinyML Models

## Classification Performance

To reproduce the results, please refer to `evaluate/evaluate.ipynb`.

## Resource Consumption

(under construction...)

# Repo Structure

```
├── artifacts   # different trials...
├── baseline    # Implementation of Baseline in C
├── datasets    # Datasets from kaggle
├── evaluate    # Jupyter notebook for classification evaluation
├── mel_spectrogram # Implementation of Mel-Spectrogram
├── pilot # Jupyter notebook for pilot study
└── tinyml_models # Models and their implementations in C
    ├── CNN_Mel
    ├── CNN_Time
    ├── SqueezeNet_Mel
    ├── SqueezeNet_Time
    ├── Transformer_Time
    └── utils # Dataloader, mel-transformer etc.

```
