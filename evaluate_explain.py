#%%

# Evaluate performance model
import torch
import torchvision
from model_explain import Net
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

  test_set = torch.utils.data.TensorDataset(torch.load('data_test_def.pt'), torch.load('labels_test_def.pt'))

  print('Number of test images: {}'.format(len(test_set)))

  # Load data from disk and organize it in batches
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

  # Instantiate model structure
  model = Net()

  # Move the model to the right device
  model = model.to(device)

  # Load the model from file
  model.load_state_dict(torch.load(state_dict))

  # Set themodel in evaluation mode
  model.eval()

  # Inizializate variables to compute evaluation metrics
  test_accuracy = 0.0
  target = []
  pred = np.empty((0,43), float)

  # Open a CSV file
  output_file = open('pred_explain.csv', 'w')
  output_file.write('Filename,ClassId,Pred,Conf\n')

  # Evaluate the model
  for idx, batch in enumerate(test_loader):
    x, y = batch

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
    # if (pred_idxs != y):
    
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
