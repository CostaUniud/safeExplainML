# %%

# Train second model with explainability data
import torch
import torchvision
import torch.nn.functional as F
from model_explain import Net_explain
# import matplotlib.pyplot as plt

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
    torchvision.transforms.Normalize((-6.1252e-03, -9.2014e-03, -6.5125e-03, -6.9699e-03, -9.1594e-03,
        -6.7314e-03, -8.6459e-03, -1.1104e-02, -8.0948e-03, -7.0646e-03,
        -1.1371e-02, -9.8521e-03, -8.4179e-03, -1.0891e-02, -6.5474e-03,
        -5.6363e-03, -7.8393e-03, -6.2083e-03,  6.2250e-05, -1.4476e-02,
        -8.9999e-03, -5.5913e-03, -1.1191e-02, -5.0998e-03, -7.4369e-03,
        -1.0318e-02, -6.2087e-03, -8.0595e-03, -1.0829e-02, -6.2457e-03,
        -7.3045e-03, -1.1525e-02, -5.1274e-03, -2.4338e-03, -1.4372e-02,
        -7.3189e-03, -6.0360e-03, -1.0727e-02, -8.3029e-03, -8.6634e-03,
        -1.1569e-02, -8.4311e-03, -9.3450e-03, -9.4304e-03, -7.1391e-03,
        -1.1910e-02, -7.1874e-03, -8.5546e-03, -3.0863e-03, -1.1668e-02,
        -6.3716e-03, -7.6904e-03, -9.9651e-03, -6.0123e-03, -7.5910e-03,
        -1.1480e-02, -1.0464e-02, -6.5038e-03, -1.2236e-02, -5.8235e-03,
        -4.2946e-03, -1.1727e-02, -6.9553e-03, -5.8615e-03, -1.4544e-02,
        -6.7010e-03, -1.0057e-02, -9.1149e-03, -9.0361e-03, -4.1150e-03,
        -1.1568e-02, -6.0661e-03, -7.1686e-03, -1.0636e-02, -8.7129e-03,
        -5.4148e-03, -1.0815e-02, -7.9999e-03, -7.9161e-03, -1.0438e-02,
        -7.5843e-03, -5.9259e-03, -1.3451e-02, -8.8806e-03, -5.1014e-03,
        -9.2165e-03, -9.1340e-03, -9.2023e-03, -7.5926e-03, -6.9537e-03,
        -3.4532e-03, -1.0616e-02, -7.3571e-03, -7.6865e-03, -1.2973e-02,
        -7.6857e-03, -4.1548e-03, -1.2481e-02, -9.8652e-03, -4.3122e-03,
        -1.0617e-02, -9.7941e-03, -1.7318e-03, -1.0647e-02, -1.1069e-02,
        -3.1921e-03, -1.0526e-02, -1.2143e-02, -5.4204e-03, -1.2055e-02,
        -1.1770e-02, -4.0685e-03, -1.1923e-02, -1.0502e-02, -3.9076e-03,
        -1.3562e-02, -1.0640e-02, -4.5721e-03, -1.3447e-02, -9.7179e-03,
        -7.2479e-04, -1.3003e-02, -9.9620e-03, -2.2424e-03, -1.3187e-02,
        -8.9082e-03, -1.8168e-03, -1.3100e-02, -7.4005e-03), (0.2139, 0.2164, 0.1484, 0.2226, 0.2158, 0.1550, 0.2146, 0.2128, 0.1523,
        0.2439, 0.2507, 0.1821, 0.2892, 0.2883, 0.2100, 0.2131, 0.2082, 0.1511,
        0.2152, 0.2302, 0.1564, 0.2438, 0.2346, 0.1735, 0.2469, 0.2500, 0.1728,
        0.2213, 0.2122, 0.1488, 0.2173, 0.2319, 0.1520, 0.2007, 0.2221, 0.1390,
        0.2474, 0.1948, 0.1650, 0.2006, 0.1855, 0.1332, 0.2004, 0.1946, 0.1373,
        0.2136, 0.2207, 0.1446, 0.2243, 0.2306, 0.1562, 0.1763, 0.1726, 0.1203,
        0.2115, 0.2191, 0.1528, 0.1994, 0.2040, 0.1381, 0.1962, 0.1950, 0.1366,
        0.2152, 0.2241, 0.1442, 0.2263, 0.2337, 0.1463, 0.2008, 0.1896, 0.1391,
        0.2315, 0.2278, 0.1605, 0.2040, 0.2007, 0.1457, 0.2488, 0.2281, 0.1622,
        0.2351, 0.2483, 0.1691, 0.2064, 0.2103, 0.1469, 0.2069, 0.2015, 0.1356,
        0.2089, 0.1900, 0.1358, 0.2164, 0.2152, 0.1480, 0.2118, 0.2206, 0.1522,
        0.1917, 0.1908, 0.1359, 0.1979, 0.1964, 0.1387, 0.2110, 0.1984, 0.1446,
        0.1926, 0.1904, 0.1356, 0.2137, 0.2133, 0.1450, 0.1927, 0.1955, 0.1315,
        0.2068, 0.2083, 0.1414, 0.1811, 0.1964, 0.1314, 0.2026, 0.2100, 0.1432,
        0.2305, 0.2445, 0.1605))
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
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
  model_file = 'modelDef.pth'
  torch.save(model.state_dict(), model_file)

  print('Training done!')

  # Plot accuracy and loss values for each epoch
  # plt.figure(figsize=(10,5))
  # plt.title("Training loss and accuracy")
  # plt.plot(loss_vals, label="loss")
  # plt.plot(acc_vals, label="accuray")
  # plt.xlabel("Epochs")
  # plt.ylabel("Loss / Accuracy")
  # plt.legend()
  # plt.show()




# %%
