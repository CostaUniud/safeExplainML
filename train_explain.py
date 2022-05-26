# %%
import torch
from model_explain import Net
import torch.nn.functional as F

import numpy as np

# Hyper-parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 40

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

  train_set = torch.utils.data.TensorDataset(torch.FloatTensor(np.load('data_def.npy')), torch.LongTensor(np.load('labels_def.npy')))

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
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

  print('Number of training images: {}'.format(len(train_set)))

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
      x = torch.permute(x, (0, 3, 1, 2)) # 129x32x32

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
    torch.save(model.state_dict(), './model2/' + model_file)

  print('Training done!')




# %%
