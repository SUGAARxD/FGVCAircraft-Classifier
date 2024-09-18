import torch
from tqdm import tqdm


def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for inputs, labels in tqdm(loader, desc='Train', colour='magenta'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # batch loss
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        # calculate and update gradients
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        total_accuracy += (predicted == labels).sum().item()

    # entire epoch loss
    total_loss /= len(loader.dataset)
    # entire epoch accuracy
    total_accuracy /= len(loader.dataset)
    return total_loss, total_accuracy


def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', colour='magenta'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, dim=1)
            total_accuracy += (predicted == labels).sum().item()

    # entire epoch loss
    total_loss /= len(loader.dataset)
    # entire epoch accuracy
    total_accuracy /= len(loader.dataset)
    return total_loss, total_accuracy
