import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FGVCAircraft
from torchvision import transforms
from torch.utils.data import DataLoader

from model import Classifier
from model_funcs import train_model, validate_model

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    num_workers = 6

    lr = 1e-3
    weight_decay = 1e-4

    # keep in mind that you should change the initial epoch when you continue training a trained model
    initial_epoch = 0
    num_epochs = 100

    best_loss = np.inf
    best_accuracy = 0

    trained_models_dir = 'family/trained_models/trained1'
    # path from content root or trained_models_dir + '/' + file name
    model_to_load = ''

    dataset_dir = "dataset"

    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_dataset = FGVCAircraft(root=dataset_dir, split='train', transform=train_transform, annotation_level='family',
                                 download=True)
    val_dataset = FGVCAircraft(root=dataset_dir, split='val', transform=val_transform, annotation_level='family',
                               download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    classifier = Classifier(3, 70)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()

    if os.path.isfile(f'{model_to_load}'):
        classifier.load_state_dict(torch.load(f'{model_to_load}', map_location=device))
        best_loss, best_accuracy = validate_model(classifier, val_loader, criterion, device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    train_writer = SummaryWriter(f'{trained_models_dir}/logs/train')
    val_writer = SummaryWriter(f'{trained_models_dir}/logs/val')

    for epoch in range(initial_epoch, initial_epoch + num_epochs):
        print(f'\nEpoch {epoch + 1}/{initial_epoch + num_epochs}')
        time.sleep(0.1)

        train_loss, train_accuracy = train_model(classifier, train_loader, criterion, optimizer, device)

        train_writer.add_scalar('loss', train_loss, epoch + 1)
        train_writer.add_scalar('accuracy', train_accuracy, epoch + 1)

        print(f'loss: {train_loss:.3f}  accuracy: {train_accuracy:.3f}')
        time.sleep(0.1)

        val_loss, val_accuracy = validate_model(classifier, val_loader, criterion, device)

        val_writer.add_scalar('loss', val_loss, epoch + 1)
        val_writer.add_scalar('accuracy', val_accuracy, epoch + 1)

        print(f'loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}')
        time.sleep(0.1)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(classifier.state_dict(), f'{trained_models_dir}/best_loss_model_{epoch + 1}.pth')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(classifier.state_dict(), f'{trained_models_dir}/best_accuracy_model_{epoch + 1}.pth')

        time.sleep(0.1)

    torch.save(classifier.state_dict(), f'{trained_models_dir}/last_model_{initial_epoch + num_epochs}.pth')
    train_writer.close()
    val_writer.close()
