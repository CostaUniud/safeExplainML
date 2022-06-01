# %%

# Train second model with explainability data
import torch
import torchvision
import torch.nn.functional as F
from model_explain import Net
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
    torchvision.transforms.Normalize((-0.0048, -0.0074, -0.0061, -0.0052, -0.0066, -0.0054, -0.0050, -0.0065,
        -0.0058, -0.0045, -0.0054, -0.0054, -0.0051, -0.0067, -0.0044, -0.0043,
        -0.0057, -0.0056, -0.0011, -0.0083, -0.0067, -0.0051, -0.0077, -0.0060,
        -0.0054, -0.0071, -0.0063, -0.0056, -0.0078, -0.0060, -0.0047, -0.0062,
        -0.0042, -0.0028, -0.0071, -0.0066, -0.0047, -0.0083, -0.0077, -0.0069,
        -0.0073, -0.0060, -0.0059, -0.0078, -0.0074, -0.0079, -0.0072, -0.0060,
        -0.0025, -0.0086, -0.0060, -0.0051, -0.0084, -0.0066, -0.0055, -0.0076,
        -0.0064, -0.0041, -0.0078, -0.0056, -0.0037, -0.0072, -0.0062, -0.0034,
        -0.0076, -0.0049, -0.0065, -0.0074, -0.0065, -0.0029, -0.0077, -0.0055,
        -0.0053, -0.0081, -0.0070, -0.0043, -0.0061, -0.0059, -0.0054, -0.0076,
        -0.0062, -0.0042, -0.0079, -0.0059, -0.0041, -0.0068, -0.0060, -0.0056,
        -0.0056, -0.0043, -0.0027, -0.0071, -0.0064, -0.0047, -0.0072, -0.0045,
        -0.0042, -0.0087, -0.0062, -0.0013, -0.0070, -0.0078, -0.0017, -0.0068,
        -0.0083, -0.0026, -0.0076, -0.0088, -0.0029, -0.0077, -0.0084, -0.0010,
        -0.0080, -0.0081, -0.0022, -0.0078, -0.0088, -0.0010, -0.0079, -0.0079,
        -0.0006, -0.0078, -0.0080, -0.0027, -0.0095, -0.0067, -0.0025, -0.0089,
        -0.0070), (0.1674, 0.1696, 0.1243, 0.1689, 0.1613, 0.1186, 0.1573, 0.1539, 0.1161,
        0.1560, 0.1605, 0.1198, 0.1643, 0.1586, 0.1165, 0.1635, 0.1566, 0.1184,
        0.1710, 0.1735, 0.1249, 0.1838, 0.1765, 0.1331, 0.1917, 0.1842, 0.1370,
        0.1660, 0.1598, 0.1188, 0.1418, 0.1441, 0.1023, 0.1445, 0.1460, 0.1063,
        0.1695, 0.1533, 0.1202, 0.1447, 0.1397, 0.1062, 0.1663, 0.1719, 0.1246,
        0.1682, 0.1701, 0.1204, 0.1803, 0.1731, 0.1268, 0.1436, 0.1475, 0.1080,
        0.1632, 0.1564, 0.1144, 0.1406, 0.1353, 0.1044, 0.1335, 0.1276, 0.0989,
        0.1537, 0.1481, 0.1082, 0.1546, 0.1517, 0.1119, 0.1410, 0.1360, 0.1047,
        0.1591, 0.1598, 0.1200, 0.1404, 0.1261, 0.0964, 0.1665, 0.1535, 0.1126,
        0.1758, 0.1674, 0.1202, 0.1494, 0.1432, 0.1113, 0.1322, 0.1292, 0.0998,
        0.1371, 0.1360, 0.1011, 0.1453, 0.1378, 0.1030, 0.1414, 0.1422, 0.1059,
        0.1291, 0.1376, 0.1010, 0.1389, 0.1394, 0.1055, 0.1550, 0.1495, 0.1137,
        0.1373, 0.1341, 0.1025, 0.1425, 0.1455, 0.1079, 0.1462, 0.1402, 0.1056,
        0.1450, 0.1403, 0.1044, 0.1444, 0.1464, 0.1115, 0.1591, 0.1541, 0.1174,
        0.1699, 0.1693, 0.1254))
  ])

  train_set = torch.utils.data.TensorDataset(torch.load('data_train_def.pt'), torch.load('labels_train_def.pt'))

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
  model = Net()

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
