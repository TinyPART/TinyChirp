# TinyBird Sounds
## Vision Transformer Approach
To use the model and test it, you can use the TransformerModel class : 
```
from TransformerModel import TransformerModel
model = TransformerModel(num_classes, feature_size, 1, 32, 32, num_layers)
```
You then have to import the Dataset using the class AudioDataset : 
```
from AudioDataset import AudioDataset

train_dataset = AudioDataset(/path/to/train_folder/target, /path/to/train_folder/non_target,fixed_length_spect= 800)
train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
validation_dataset = AudioDataset(validation_target_folder, validation_non_target_folder,fixed_length_spect = 90, train=False)
validation_dataloader = DataLoader(validation_dataset, batch_size = 32, shuffle = True)
```
Once the dataset is loaded, you can use functions from util to train and evaluate the model.

### Train the model
To train the model you can use the function train_and_evaluate_classic. This function will train the model and evaluate its accuracy on the validatation dataset every epoch. Each time the model surpasses its last best accuracy the state dictionary is saved. You can use a checkpoint to start from a pretrained set of weights.
```
from util import train_and_evaluate_classic

train_and_evaluate_vit(model, train_dataloader, val_dataloader,  optimizer, epochs, device, checkpoint = None)
```
### Evaluate the model
After the model is trained, you can evaluate core metrics using evaluate_model_metrics : 
```
from util import evaluate_model_metrics 

evaluate_model_metrics(dataloader, model, device)
```
This function will output True Positives, True Negatives, False Positives, False Negatives, Accuracy, F1 Score and F2 Score.


## Export the model
To export the model in a format that can be used by RIOT-ML, we use torchscript.
```
# Model in evaluation mode
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# We use random input or input from the dataset
traced = torch.jit.trace(model,input)
traced.save("traced_model.pth")
```

## Testing the model on the card

The next step is to use RIOT-ML to port the model on the card. You can use it directly with the torchscript model : 
```
TVM_HOME=/home/user/tvm PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} USE_ETHOS=1 python3 u-toe.py --per-model --board nrf52840dk --input-shape 1,800,64 ./model_zoo/traced_model.pth
```

Be aware that there may be a few preliminary steps to flash the model on a real card. 
Here is my workflow on a nrf52840dk : 
You have to recover the card to allow flashing. You can use 
```nrfjprog -f nrf52 --recover```
nrfjprog is part of nordic command-line tools. 
If you use windows and a linux subsystem you have to use usbipd to link the card with WSL.
Then you can use RIOT-ML to flash the model on the card.

## Testing it yourself
All the python workflow can be found in example_vit.ipynb


## Spectrogram approach
Spectrogram based models are implemented in tensorflow.
Training using 3 seconds audio.
In test_spectrogram_models.ipynb example of predictions using models.
