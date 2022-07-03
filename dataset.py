import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_data_loader_mnist(datasetname, root, batch_size):
  '''
  Digits dataset is a combination of three related datasets:
      MNIST
  '''
  if datasetname == 'digits':
    
    transform = transforms.Compose([
              transforms.Resize((32, 32)),
              transforms.ToTensor(),
    ])

    # Get the MNIST dataset
    mnist_train = datasets.MNIST(root=root,
                                train=True,
                                transform= transform,
                                download=True)
    
    mnist_valid = datasets.MNIST(root=root,
                                train=False,
                                transform = transform,
                                download=True)
    
    # Get the train loader
    train_loader = DataLoader(
        mnist_train,
        batch_size = batch_size,
        num_workers = 2,
        pin_memory = True,
        shuffle= True
    )

    # Get the test loader
    test_loader = DataLoader(
        mnist_valid,
        batch_size = batch_size,
        num_workers = 2,
        pin_memory = True,
        shuffle= True
    )
  return train_loader, test_loader


def get_data_loader_mixed(datasetname, root, batch_size):
  '''
  Digits dataset is a combination of three related datasets:
      1. SVHN 
      2. MNISTM
      3. MNIST
  '''
  if datasetname == 'digits':

    # Get the SVHN dataset
    svhn_train = datasets.SVHN(root=root, 
                              split='train',
                              download=True)
    
    svhn_test = datasets.SVHN(root=root,
                              split='test',
                              download=True)
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    svhn_train.data = normalize(torch.from_numpy(svhn_train.data).float())
    svhn_test.data  = normalize(torch.from_numpy(svhn_test.data).float())

    # Get the MNIST dataset
    mnist_train = datasets.MNIST(root=root,
                                train=True,
                                download=True)
    
    mnist_test = datasets.MNIST(root=root,
                                train=False,
                                download=True)
    
    transform = transforms.Resize((32, 32))
    mnist_train.data = normalize(transform(mnist_train.data.unsqueeze(1).repeat(1, 3, 1, 1)).float())
    mnist_test.data  = normalize(transform(mnist_test.data.unsqueeze(1).repeat(1, 3, 1, 1)).float())
    
    # Concatenate datasets
    train_dataset = torch.cat((svhn_train.data, mnist_train.data))
    test_dataset  = torch.cat((svhn_test.data,  mnist_test.data))
    
    # Get the train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = 2,
        pin_memory = True,
        shuffle= True
    )

    # Get the test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        num_workers = 2,
        pin_memory = True,
        shuffle= True
    )
  return train_loader, test_loader



def visualize_data(loader):
  '''
  Visualize the data in a grid.
  '''
  # Get a sample of data
  samples = next(iter(loader))[0]
  
  # Determine grid size
  grid_width = 8 
  grid_height = 8
  
  # Create the grid
  image_grid = make_grid(samples[:grid_width*grid_height], nrow=grid_width)
  
  # Visualization arrangement
  plt.rcParams['figure.figsize'] = [grid_height, grid_width]
  plt.imshow(image_grid.permute(1, 2, 0))
  plt.axis('off')
  plt.show()  
