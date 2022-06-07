#%%

# Evaluate performance model
import torch
import torchvision
from model_explain import Net_explain
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
import numpy as np

# Model file path
state_dict = './model2/model2.pth'

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

# Main
if __name__ == '__main__':
  # Define the device where we want run our model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Normalize images to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
  normalize = torchvision.transforms.Compose([
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

  test_set = torch.utils.data.TensorDataset(torch.load('data_test_def.pt'), torch.load('labels_test_def.pt'))

  print('Number of test images: {}'.format(len(test_set)))

  # Load data from disk and organize it in batches
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

  # Instantiate model structure
  model = Net_explain()

  # Move the model to the right device
  model = model.to(device)

  # Load the model from file
  model.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')))

  # Set themodel in evaluation mode
  model.eval()

  # Inizializate variables to compute evaluation metrics
  test_accuracy = 0.0
  target = []
  pred = np.empty((0,43), float)

  # Open a CSV file
  output_file = open('wrong_explain.csv', 'w')
  output_file.write('Filename,ClassId,Pred,Conf\n')

  # Evaluate the model
  for idx, batch in enumerate(test_loader):
    x, y = batch
    x = torch.permute(x, (0, 3, 1, 2)) # 129x32x32
    x = normalize(x)

    # Move the data to the right device
    x, y = x.to(device), y.to(device)

    # Perform the forward pass with Net
    out = model(x)

    # Get predictions values
    probs = torch.softmax(out, dim=1)
    conf, pred_idxs = torch.max(probs.data, 1)

    # Append current label and prediction values
    target = np.append(target, y.item())
    pred = np.append(pred, probs.cpu().detach().numpy(), axis=0)

    # Check for incorrect predictions
    if (pred_idxs != y):
      # Write info about predction on CSV file
      output_file.write("%d,%s,%s,%f\n" % (idx, classes[y], classes[pred_idxs], conf))

    # Get the accuracy of one logit and sum with it to that of the others
    test_accuracy += accuracy(probs, y)

  # Compute accuracy
  test_accuracy /= len(test_set)
  print('Test accuracy: {:.05f}'.format(test_accuracy))
  output_file.write('Test accuracy: {:.05f}'.format(test_accuracy))

  # Compute ROC AUC
  macro_roc_auc_ovo = roc_auc_score(target, pred, multi_class="ovo", average="macro")
  weighted_roc_auc_ovo = roc_auc_score(
    target, pred, multi_class="ovo", average="weighted"
  )
  macro_roc_auc_ovr = roc_auc_score(target, pred, multi_class="ovr", average="macro")
  weighted_roc_auc_ovr = roc_auc_score(
    target, pred, multi_class="ovr", average="weighted"
  )
  print(
    "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
  )
  print(
    "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
  )
  output_file.write(
    "\nOne-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
  )
  output_file.write(
    "\nOne-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
  )

  # Compute and plot precision-recall curve
  precision = dict()
  recall = dict()
  average_precision = dict()

  target = label_binarize(target, classes=np.arange(43))
  
  # For each class
  for i in range(43):
    precision[i], recall[i], _ = precision_recall_curve(target[:, i], pred[:, i])
    average_precision[i] = average_precision_score(target[:, i], pred[:, i])

  # A "micro-average": quantifying score on all classes jointly
  precision["micro"], recall["micro"], _ = precision_recall_curve(
    target.ravel(), pred.ravel()
  )
  average_precision["micro"] = average_precision_score(target, pred, average="micro")

  display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
  )
  display.plot()
  _ = display.ax_.set_title("Micro-averaged over all classes")

  # Close CSV
  output_file.close()


# %%
