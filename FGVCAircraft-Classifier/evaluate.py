import os
import torch
import torch.nn as nn
from torchvision.datasets import FGVCAircraft
from torchvision import transforms
from torch.utils.data import DataLoader

from model import Classifier
from model_funcs import validate_model

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = Classifier(3, 70)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()

    # path from content root
    model_to_load = 'family/trained_models/trained1/last_model_100.pth'

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    root_dir = "dataset"
    test_dataset = FGVCAircraft(root=root_dir, split='test', transform=transform, annotation_level='family',
                                download=True)

    batch_size = 64
    num_workers = 6

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if os.path.isfile(f'{model_to_load}'):
        classifier.load_state_dict(torch.load(f'{model_to_load}', map_location=device))

        test_loss, test_accuracy = validate_model(classifier, test_loader, criterion, device)

        print(f'loss: {test_loss:.3f}  accuracy: {test_accuracy:.3f}')

    else:
        print('No model found')
