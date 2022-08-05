import config as c
from train import train
from utils import load_datasets, make_dataloaders
import os
os.environ['TORCH_HOME'] = 'models\\EfficientNet'
# class_name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
#               'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class_name = ['pill', 'screw','toothbrush',]
# class_name = [c.class_name]

for i in class_name:
    c.class_name = i
    train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    train_loader, test_loader = make_dataloaders(train_set, test_set)
    model = train(train_loader, test_loader)