import torch
import torch.nn as nn
from tqdm.auto import tqdm
import copy 

def run_epoch(model,loader,criterion,optimizer,device):
    '''
    model: nn.Module
    loader: DataLoader
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device

    returns: avg_loss,acc

    automatically sets model to training mode if optimizer is not None,
    otherwise sets model to evaluation mode
    '''
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for X,y in loader:
            X = X.to(device)
            y = y.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits,y)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss,acc


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                epochs):

    '''
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    epochs: int

    returns: model,history
    '''

    history = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}

    best_loss = float("inf")
    best_model_state = None

    for epoch in tqdm(range(epochs),desc="Training"):
        train_loss,train_acc = run_epoch(model,
                                        train_loader,
                                        criterion,
                                        optimizer,
                                        device)

        val_loss,val_acc = run_epoch(model,val_loader,criterion,None,device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_state)

    return model,history

def test_model(model,test_loader,criterion,device):
    test_loss,test_acc = run_epoch(model,test_loader,criterion,optimizer=None,device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return test_loss,test_acc
