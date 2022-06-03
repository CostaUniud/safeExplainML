# %%

# Train second model with explainability data
import torch
import torchvision
import torch.nn.functional as F
from model_explain import Net_explain
import matplotlib.pyplot as plt

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

  # Normalize images to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
  normalize = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((-0.0073, -0.0098, -0.0057, -0.0090, -0.0105, -0.0064, -0.0107, -0.0124,
        -0.0082, -0.0089, -0.0113, -0.0087, -0.0092, -0.0125, -0.0068, -0.0076,
        -0.0097, -0.0062, -0.0013, -0.0150, -0.0080, -0.0074, -0.0129, -0.0056,
        -0.0088, -0.0104, -0.0056, -0.0087, -0.0107, -0.0054, -0.0078, -0.0128,
        -0.0048, -0.0020, -0.0144, -0.0055, -0.0033, -0.0100, -0.0073, -0.0091,
        -0.0102, -0.0073, -0.0088, -0.0097, -0.0060, -0.0122, -0.0076, -0.0080,
        -0.0041, -0.0123, -0.0051, -0.0085, -0.0094, -0.0043, -0.0077, -0.0109,
        -0.0076, -0.0070, -0.0122, -0.0047, -0.0049, -0.0103, -0.0048, -0.0061,
        -0.0147, -0.0047, -0.0100, -0.0072, -0.0070, -0.0045, -0.0119, -0.0044,
        -0.0076, -0.0102, -0.0066, -0.0047, -0.0097, -0.0065, -0.0064, -0.0095,
        -0.0061, -0.0057, -0.0131, -0.0061, -0.0056, -0.0085, -0.0067, -0.0091,
        -0.0074, -0.0055, -0.0032, -0.0098, -0.0053, -0.0086, -0.0128, -0.0063,
        -0.0044, -0.0113, -0.0083, -0.0050, -0.0103, -0.0085, -0.0012, -0.0096,
        -0.0087, -0.0028, -0.0093, -0.0102, -0.0049, -0.0109, -0.0099, -0.0040,
        -0.0119, -0.0086, -0.0039, -0.0128, -0.0083, -0.0060, -0.0127, -0.0086,
        -0.0017, -0.0130, -0.0082, -0.0019, -0.0124, -0.0068, -0.0024, -0.0144,
        -0.0064), (0.1827, 0.1772, 0.1277, 0.1950, 0.1836, 0.1337, 0.1908, 0.1847, 0.1347,
        0.1970, 0.1949, 0.1425, 0.2406, 0.2327, 0.1734, 0.1865, 0.1742, 0.1262,
        0.1869, 0.1958, 0.1314, 0.2123, 0.1990, 0.1444, 0.1983, 0.1886, 0.1423,
        0.1933, 0.1770, 0.1229, 0.1914, 0.2008, 0.1296, 0.1757, 0.1970, 0.1158,
        0.2035, 0.1479, 0.1340, 0.1710, 0.1514, 0.1127, 0.1672, 0.1597, 0.1171,
        0.1852, 0.1867, 0.1225, 0.1964, 0.1933, 0.1300, 0.1572, 0.1464, 0.1055,
        0.1798, 0.1817, 0.1315, 0.1804, 0.1794, 0.1198, 0.1673, 0.1558, 0.1140,
        0.1948, 0.2064, 0.1290, 0.1904, 0.1995, 0.1246, 0.1806, 0.1685, 0.1204,
        0.2087, 0.1997, 0.1409, 0.1684, 0.1609, 0.1167, 0.2106, 0.1859, 0.1331,
        0.2077, 0.2128, 0.1449, 0.1753, 0.1704, 0.1197, 0.1805, 0.1740, 0.1149,
        0.1820, 0.1601, 0.1124, 0.1939, 0.1941, 0.1331, 0.1706, 0.1737, 0.1232,
        0.1707, 0.1653, 0.1156, 0.1657, 0.1606, 0.1129, 0.1745, 0.1548, 0.1129,
        0.1558, 0.1506, 0.1100, 0.1883, 0.1858, 0.1268, 0.1643, 0.1623, 0.1110,
        0.1856, 0.1816, 0.1271, 0.1558, 0.1645, 0.1090, 0.1741, 0.1760, 0.1194,
        0.1971, 0.2069, 0.1302))
  ])

  train_set = torch.utils.data.TensorDataset(torch.load('data_train_def_15000.pt'), torch.load('labels_train_def_15000.pt'))

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
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

  print('Number of training images: {}'.format(len(train_set)))

  # Instantiate model structure
  model = Net_explain()

  # Move the model to the right device
  model = model.to(device)

  # Define the optimizer (Adam)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

  # Set the model in training mode
  model.train()

  # Set loss and acc values arrays for visualization purpose
  loss_vals = []
  acc_vals = []

  # Train the model
  for e in range(EPOCHS):
    epoch_loss = []
    epoch_acc = []
    for i, batch in enumerate(train_loader):
      x, y = batch
      x = torch.permute(x, (0, 3, 1, 2)) # 129x32x32
      x = normalize(x)

      # Move the data to the right device
      x, y = x.to(device), y.to(device)

      # Perform the forward pass with Net
      out = model(x)

      # Define the loss function
      loss = F.nll_loss(out, y)

      # Perfom the update of the model's parameter using the optimizer
      optimizer.zero_grad() # Clean previous gradients
      loss.backward()
      epoch_loss.append(loss.item())
      optimizer.step()

      # Obtain probabilities over the logits
      a = accuracy(torch.softmax(out, dim=1), y)
      epoch_acc.append(a)

      # Print information about the training
      if i % 100 == 0:
        print('Loss: {:.05f} - Accuracy {:.05f}'.format(loss.item(), a))

    loss_vals.append(sum(epoch_loss)/len(epoch_loss))
    acc_vals.append(sum(epoch_acc)/len(epoch_acc))
    print('Epoch {} done!'.format(e))

  # Save the model
  model_file = 'model2.pth'
  torch.save(model.state_dict(), './model2/' + model_file)

  print('Training done!')

  # Plot accuracy and loss values for each epoch
  plt.figure(figsize=(10,5))
  plt.title("Training loss and accuracy")
  plt.plot(loss_vals, label="loss")
  plt.plot(acc_vals, label="accuray")
  plt.xlabel("Epochs")
  plt.ylabel("Loss / Accuracy")
  plt.legend()
  plt.show()




# %%
