#%%
import numpy as np
import sys
import torch
import torchvision
from model import Net

# Model file to evaluate
state_dict = 'model.pth'

# Classes (43) which images belong to
classes = ('Limit 20km', 'Limit 30km', 'Limit 50km', 'Limit 60km', 'Limit 70km', 'Limit 80km', 
        'End limit 80km', 'Limit 100km', 'Limit 120km', 'No overtaking', 'No overtaking of heavy vehicles', 
        'Intersection with right of way', 'Right of way', 'Give right to pass', 'Stop', 'Transit prohibition', 
        'Prohibition of heavy vehicles transit', 'Wrong way', 'Generic danger', 'Dangerous left curve', 'Dangerous right curve', 
        'Double curve', 'Danger of bumps', 'Danger of slipping', 'Asymmetrical bottleneck', 'Men at work', 'Traffic light', 
        'Pedestrian crossing', 'School', 'Cycle crossing', 'Snow', 'Wild animals', 'Go ahead', 'Right turn mandatory', 
        'Left turn mandatory', 'Mandatory direction straight', 'Directions right and straight', 'Directions left and straight', 
        'Mandatory step to the right', 'Mandatory step to the left', 'Roundabout', 'End of no overtaking', 'End of no overtaking of heavy vehicles')

def enable_dropout(model):
  """ Function to enable the dropout layers during test-time """
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()

# Main
if __name__ == '__main__':
  # Define the device where we want run our model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
  ])

  # Load the GTSRB test set
  test_set = torchvision.datasets.GTSRB(
    root = './data',
    split = 'test',
    download = True,
    transform = transforms
  )

  print('Number of test images: {}'.format(len(test_set)))

  # Load data from disk and organize it in batches
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

  # Instantiate model structure
  model = Net()

  # Move the model to the right device
  model = model.to(device)

  # Load the model from file
  model.load_state_dict(torch.load('./model/' + state_dict))

  # Performe i forward passes to uncertainty estimation
  for i in range(100): # 5
    predictions = np.empty((0, 43))
    # Set the model in evaluation mode
    model.eval()
    enable_dropout(model)

    dropout_predictions = np.empty((0, 12630, 43))

    # Evaluate the model
    for idx, batch in enumerate(test_loader):
      x, y = batch

      # Move the data to the right device
      x, y = x.to(device), y.to(device)

      # Perform the forward pass with Net
      out = model(x)

      # Get predictions values
      probs = torch.softmax(out, dim=1)

      predictions = np.vstack((predictions, probs.cpu().detach().numpy()))

    dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))

  with open('uncertainty_2.npy', 'wb') as f:
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
    np.save(f, mean)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)
    np.save(f, variance)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)
    np.save(f, entropy)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon), axis=-1), axis=0) # shape (n_samples,)
    np.save(f, mutual_info)

  print(mean)
  print(variance)
  print(entropy)
  print(mutual_info)


# %%
