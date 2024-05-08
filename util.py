import torch
import torch.nn as nn
from tqdm import tqdm
import sys
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
sys.path.append("BirdNET-Analyzer")
import analyze

def loss_kd(teacher_outputs,student_outputs,targets,lmbd,T):
    # Loss for the knowledge distillation
    #print(f"Teacher : {teacher_outputs.device}\n Student : {student_outputs.device}")
    KD_Loss = nn.KLDivLoss()(student_outputs,teacher_outputs/T) * (1-lmbd) + nn.functional.cross_entropy(student_outputs,targets) * lmbd
    return KD_Loss


def get_total_params(model):
    # Get the total number of parameters of a model
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def distillation_epoch(student_model, dataloader, lmbd, T, optimizer, device, class_number):
    # Training epoch for the knowledge distillation
    student_model.train()
    student_model.to(device)
    loss_avg = 0
    for batch in tqdm(dataloader):
        chunks, inputs, targets = batch
        inputs, targets = inputs.squeeze(1).to(device),targets.to(device)
        student_outputs = student_model(inputs)
        student_outputs = F.softmax(student_outputs)
        teacher_outputs = analyze.predict_tensor(chunks[0], class_number)[1]
        teacher_outputs = teacher_outputs.to(device)
        loss = loss_kd(teacher_outputs,student_outputs,targets,lmbd,T)
        loss_avg += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_avg/len(dataloader)

def train_epoch_vit(student_model, dataloader, optimizer,device):
    # Training epoch for the vision transformer
    student_model.train()
    student_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_avg = 0
    for batch in tqdm(dataloader):
        inputs, targets = batch
        inputs, targets = inputs.squeeze(1).to(device),targets.to(device)
        student_outputs = student_model(inputs)
        loss = loss_fn(student_outputs,targets)
        loss_avg += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_avg/len(dataloader) 

def train_epoch(student_model, dataloader, optimizer,device):
    # Training epoch for the CNN Model
    student_model.train()
    student_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_avg = 0
    for batch in tqdm(dataloader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        student_outputs = student_model(inputs)
        
        loss = loss_fn(student_outputs,targets)
        loss_avg += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_avg/len(dataloader) 

def evaluate_model_accuracy(model, val_dataloader,device,squeeze = False):
    # Function to evaluate the model accuracy
    model.eval()
    model.to(device)
    running_acc = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            if squeeze :
                inputs, targets = inputs.squeeze(1).to(device), targets.to(device)
            else :
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs,1)
            #print(f"Preds : {preds}\nTargets : {targets}")
            running_acc += torch.sum(preds == targets)
    return running_acc/(len(val_dataloader)*32)  
def train_and_evaluate_vit(student_model, train_dataloader, val_dataloader,  optimizer, epochs, device, checkpoint = None):
    # Function to train the ViT, and output at each epoch the validation accuracy
    if checkpoint is not None:
        student_model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        
    best_val_acc = 0.0
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)
    student_model = student_model.to(device)
    print(f"Lancement du training pour {epochs} epochs")
    for epoch in range(epochs):
        scheduler.step()
        training_loss = train_epoch_vit(student_model, train_dataloader, optimizer, device)
        val_acc = evaluate_model_accuracy(student_model, val_dataloader, device)
        print(f"Epoch {epoch}: Training Loss = {training_loss}, Validation Accuracy = {100*val_acc:.2f}% ")
        if val_acc > best_val_acc:
            print("New best accuracy has been found")
            best_val_acc = val_acc
            print(f"Saving model to model model.{epoch}.{val_acc}.pth")
            torch.save(student_model.state_dict(), f"model.{epoch}.{val_acc}.pth")
def train_and_evaluate_classic(student_model, train_dataloader, val_dataloader,  optimizer, epochs, device, checkpoint = None):
    # Train the CNN and output validation accuracy
    if checkpoint is not None:
        student_model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        
    best_val_acc = 0.0
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)
    student_model = student_model.to(device)
    print(f"Lancement du training pour {epochs} epochs")
    for epoch in range(epochs):
        scheduler.step()
        training_loss = train_epoch(student_model, train_dataloader, optimizer, device)
        val_acc = evaluate_model_accuracy(student_model, val_dataloader, device)
        print(f"Epoch {epoch}: Training Loss = {training_loss}, Validation Accuracy = {100*val_acc:.2f}% ")
        if val_acc > best_val_acc:
            print("New best accuracy has been found")
            best_val_acc = val_acc
            print(f"Saving model to model model.{epoch}.{val_acc}.pth")
            torch.save(student_model.state_dict(), f"model.{epoch}.{val_acc}.pth")
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def evaluate_model_metrics(dataloader, model, device, squeeze=False):
    # Calculate the confusion matrix, the accuracy, and the F1 and F2 score of the model on a dataset
    model.eval()  # Set model to evaluation mode
    model = model.to(device)
    # Initialize counters
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            if squeeze:
                inputs = inputs.squeeze(1).to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            preds = torch.argmax(outputs,1)
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f2 = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall)

    return {
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "F2 Score": f2
    }