#%%
import torch
import torchvision

from model import Net
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from data import data_jitter_hue, data_jitter_brightness, data_jitter_saturation, data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip, data_vflip

# Hyper-parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 40

# Show an image
def imshow(img):
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Calculate accuracy
def accuracy(preds, labels):
  """
  Computes the accuracy between preds and labels

  preds: 
    torch.tensor of size (B, N) where B is the batch size 
    and N is the number of classes
    it contains the predicted probabilities for each class
  labels:
    torch.tensor of size (B) where each item is an integer 
    taking value in [0,N-1]

  Returns:
    the accuracy between preds and labels
  """
  _, pred_idxs = torch.max(preds.data, 1)
  correct = (pred_idxs == labels).sum()
  total = labels.size(0)
  return float(correct) / total

# Main
if __name__ == '__main__':
  # Fix the seed
  torch.manual_seed(123)

  # Define the device where we want run our model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
  ])

  # Load the GTSRB training set
  train_set = torchvision.datasets.GTSRB(
    root = './data',
    split = 'train',
    download = True,
    transform = transforms
  )

  # Classes (43) which images belong to
  classes = ('Limit 20km', 'Limit 30km', 'Limit 50km', 'Limit 60km', 'Limit 70km', 'Limit 80km', 
            'End limit 80km', 'Limit 100km', 'Limit 120km', 'No overtaking', 'No overtaking of heavy vehicles', 
            'Intersection with right of way', 'Right of way', 'Give right to pass', 'Stop', 'Transit prohibition', 
            'Prohibition of heavy vehicles transit', 'Wrong way', 'Generic danger', 'Dangerous left curve', 'Dangerous right curve', 
            'Double curve', 'Danger of bumps', 'Danger of slipping', 'Asymmetrical bottleneck', 'Men at work', 'Traffic light', 
            'Pedestrian crossing', 'School', 'Cycle crossing', 'Snow', 'Wild animals', 'Go ahead', 'Right turn mandatory', 
            'Left turn mandatory', 'Mandatory direction straight', 'Directions right and straight', 'Directions left and straight', 
            'Mandatory step to the right', 'Mandatory step to the left', 'Roundabout', 'End of no overtaking', 'End of no overtaking of heavy vehicles')

  # Load data from disk and organize it in batches
  train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
      train_set, 
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_jitter_hue),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_jitter_brightness),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_jitter_saturation),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_jitter_contrast),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_rotate),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_hvflip),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_shear),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_translate),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_center),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_hflip),
      torchvision.datasets.ImageFolder('./data/gtsrb/GTSRB/Training', transform=data_vflip)
      ]), 
      batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

  print('Number of training images: {}'.format(len(train_set) * 11))

  # Get some random training images to show
  # dataiter = iter(train_loader)
  # images, labels = dataiter.next()
  # images, labels = images[:4], labels[:4]
  # Show images
  # imshow(torchvision.utils.make_grid(images))
  # Print image size
  # print(images[0].size())
  # Print labels
  # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

  # Instantiate model structure
  model = Net()

  # Move the model to the right device
  model = model.to(device)

  # Define the optimizer (Adam)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

  # Set the model in training mode
  model.train()

  # Train the model
  for e in range(EPOCHS):
    for i, batch in enumerate(train_loader):
      x, y = batch

      # Move the data to the right device
      x, y = x.to(device), y.to(device)

      # Perform the forward pass with Net
      out = model(x)

      # Define the loss function
      loss = F.nll_loss(out, y)

      # Perfom the update of the model's parameter using the optimizer
      optimizer.zero_grad() # Clean previous gradients
      loss.backward()
      optimizer.step()

      # Print information about the training
      if i % 100 == 0:
        # Obtain probabilities over the logits
        a = accuracy(torch.softmax(out, dim=1), y)
        print('Loss: {:.05f} - Accuracy {:.05f}'.format(loss.item(), a))

    print('Epoch {} done!'.format(e))
    # Save the model
    model_file = 'model_' + str(e) + '.pth'
    torch.save(model.state_dict(), './model/' + model_file)

  print('Training done!')



        




# %%
