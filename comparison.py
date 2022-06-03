#%%

# Compute input feature importance with Captum library
import torch
import torchvision
from model import Net
from model_explain import Net_explain
from captum.attr import IntegratedGradients

# Model file to evaluate
state_dict = 'model/model1.pth'
state_dict2 = 'model2/model2.pth'

# Classes (43) which images belong to
classes = ('Limit 20km', 'Limit 30km', 'Limit 50km', 'Limit 60km', 'Limit 70km', 'Limit 80km', 
        'End limit 80km', 'Limit 100km', 'Limit 120km', 'No overtaking', 'No overtaking of heavy vehicles', 
        'Intersection with right of way', 'Right of way', 'Give right to pass', 'Stop', 'Transit prohibition', 
        'Prohibition of heavy vehicles transit', 'Wrong way', 'Generic danger', 'Dangerous left curve', 'Dangerous right curve', 
        'Double curve', 'Danger of bumps', 'Danger of slipping', 'Asymmetrical bottleneck', 'Men at work', 'Traffic light', 
        'Pedestrian crossing', 'School', 'Cycle crossing', 'Snow', 'Wild animals', 'Go ahead', 'Right turn mandatory', 
        'Left turn mandatory', 'Mandatory direction straight', 'Directions right and straight', 'Directions left and straight', 
        'Mandatory step to the right', 'Mandatory step to the left', 'Roundabout', 'End of no overtaking', 'End of no overtaking of heavy vehicles')

# Show an image
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

# Calling attribute on attribution algorithm defined in input
def attribute_image_features(algorithm, input, target, **kwargs):
  model.zero_grad()
  tensor_attributions = algorithm.attribute(input, target=target, **kwargs)
  
  return tensor_attributions

# Main
if __name__ == '__main__':
  # Define the device where we want run our model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
  normalize = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
  ])

  normalize2 = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((-0.0059, -0.0086, -0.0058, -0.0067, -0.0085, -0.0060, -0.0082, -0.0108,
        -0.0073, -0.0067, -0.0113, -0.0092, -0.0083, -0.0103, -0.0058, -0.0052,
        -0.0076, -0.0055,  0.0003, -0.0141, -0.0082, -0.0055, -0.0108, -0.0042,
        -0.0070, -0.0101, -0.0054, -0.0075, -0.0106, -0.0055, -0.0065, -0.0116,
        -0.0045, -0.0024, -0.0142, -0.0066, -0.0055, -0.0101, -0.0079, -0.0080,
        -0.0111, -0.0080, -0.0089, -0.0088, -0.0065, -0.0114, -0.0067, -0.0079,
        -0.0028, -0.0114, -0.0056, -0.0074, -0.0094, -0.0053, -0.0075, -0.0114,
        -0.0101, -0.0063, -0.0120, -0.0051, -0.0040, -0.0115, -0.0064, -0.0059,
        -0.0143, -0.0060, -0.0096, -0.0086, -0.0084, -0.0037, -0.0114, -0.0054,
        -0.0073, -0.0102, -0.0080, -0.0051, -0.0104, -0.0074, -0.0076, -0.0103,
        -0.0071, -0.0062, -0.0132, -0.0082, -0.0050, -0.0090, -0.0086, -0.0090,
        -0.0071, -0.0063, -0.0034, -0.0102, -0.0067, -0.0075, -0.0128, -0.0069,
        -0.0038, -0.0118, -0.0092, -0.0041, -0.0104, -0.0094, -0.0013, -0.0103,
        -0.0107, -0.0030, -0.0103, -0.0118, -0.0051, -0.0118, -0.0112, -0.0037,
        -0.0118, -0.0101, -0.0033, -0.0134, -0.0103, -0.0042, -0.0132, -0.0092,
        -0.0004, -0.0130, -0.0095, -0.0019, -0.0128, -0.0083, -0.0016, -0.0128,
        -0.0065), (0.2061, 0.2102, 0.1431, 0.2142, 0.2091, 0.1479, 0.2116, 0.2122, 0.1491,
        0.2390, 0.2480, 0.1768, 0.2862, 0.2885, 0.2095, 0.2079, 0.2070, 0.1462,
        0.2077, 0.2241, 0.1488, 0.2365, 0.2317, 0.1704, 0.2418, 0.2460, 0.1695,
        0.2160, 0.2124, 0.1466, 0.2097, 0.2303, 0.1481, 0.1915, 0.2156, 0.1317,
        0.2317, 0.1835, 0.1549, 0.1927, 0.1816, 0.1289, 0.1887, 0.1854, 0.1293,
        0.2039, 0.2175, 0.1399, 0.2191, 0.2272, 0.1512, 0.1666, 0.1650, 0.1141,
        0.2047, 0.2137, 0.1477, 0.1942, 0.2022, 0.1349, 0.1890, 0.1892, 0.1307,
        0.2067, 0.2204, 0.1384, 0.2136, 0.2266, 0.1385, 0.1982, 0.1912, 0.1368,
        0.2235, 0.2233, 0.1557, 0.1957, 0.1957, 0.1400, 0.2404, 0.2231, 0.1576,
        0.2260, 0.2412, 0.1627, 0.1993, 0.2061, 0.1409, 0.1954, 0.1972, 0.1295,
        0.2001, 0.1842, 0.1298, 0.2066, 0.2084, 0.1410, 0.2023, 0.2114, 0.1432,
        0.1861, 0.1892, 0.1330, 0.1921, 0.1935, 0.1338, 0.2035, 0.1951, 0.1395,
        0.1853, 0.1851, 0.1294, 0.2066, 0.2092, 0.1408, 0.1870, 0.1894, 0.1254,
        0.1998, 0.2037, 0.1367, 0.1763, 0.1915, 0.1270, 0.1970, 0.2037, 0.1379,
        0.2218, 0.2386, 0.1537))
  ])

  # Load the GTSRB test set
  test_set = torchvision.datasets.GTSRB(
    root = './data',
    split = 'test',
    download = True,
    transform = normalize
  )

  # Load data from disk and organize it in batches
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

  print('Number of test images: {}'.format(len(test_loader)))

  # Instantiate model structure
  model = Net(inplace_mode = False)
  model2 = Net_explain()

  # Move the model to the right device
  model = model.to(device)
  model2 = model2.to(device)

  # Load the model from file
  model.load_state_dict(torch.load(state_dict))
  model2.load_state_dict(torch.load(state_dict2))

  # Set the model 2 in evaluation mode
  model2.eval()

  test_accuracy = 0.0
  test_accuracy2 = 0.0

  # Open a CSV file
  output_file = open('comparison2.csv', 'w')
  output_file.write('Filename,ClassId,Pred_1,Conf_1,Pred_2,Conf_2\n')

  # Evaluate the model
  for idx, batch in enumerate(test_loader):
    print(idx)
    x, y = batch

    # Move the data to the right device
    x, y = x.to(device), y.to(device)

    # Perform the forward pass with Net
    out = model(x)

    # Get predictions values model 1
    probs = torch.softmax(out, dim=1)
    conf, pred_idxs = torch.max(probs.data, 1)

    # Check for incorrect predictions
    if (pred_idxs != y):
      x.requires_grad = True

      # Set the model in evaluation mode
      model.eval()

      b = None

      # For each class
      for id, c in enumerate(classes):

        # Applies integrated gradients attribution algorithm on test image
        ig = IntegratedGradients(model)
        attr_ig, delta = attribute_image_features(ig, x, id, baselines=x * 0, return_convergence_delta=True)
        attr_ig = attr_ig.squeeze().float().to(device)
        # print(attr_ig.size())
        if b is None:
          b = attr_ig
        else:
          b = torch.cat((b, attr_ig))
      
      # print(b.size())
      b = normalize2(b).unsqueeze(0)
      out2 = model2(b)

      # Get predictions values model 2
      probs2 = torch.softmax(out2, dim=1)
      conf2, pred_idxs2 = torch.max(probs2.data, 1)

      # Check for incorrect predictions
      if (pred_idxs2 == y):
        # Write info about predction on CSV file
        output_file.write("%d,%s,%s,%f,%s,%f\n" % (idx, classes[y], classes[pred_idxs], conf, classes[pred_idxs2], conf2))
    
    # Get the accuracy of one logit and sum with it to that of the others
    test_accuracy += accuracy(probs, y)
    # Get the accuracy of one logit and sum with it to that of the others
    test_accuracy2 += accuracy(probs2, y)

  # Compute accuracy
  test_accuracy /= len(test_set)
  print('Test accuracy: {:.05f}'.format(test_accuracy))
  output_file.write('Test accuracy: {:.05f}\n'.format(test_accuracy))

  # Compute accuracy model 2
  test_accuracy2 /= len(test_set)
  print('Test accuracy model 2: {:.05f}'.format(test_accuracy2))
  output_file.write('Test accuracy model 2: {:.05f}'.format(test_accuracy2))

  # Close CSV
  output_file.close()







# %%
