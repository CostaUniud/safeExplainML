#%%

# Evaluate performance model
import torch
import torchvision
from model import Net
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
import numpy as np

# Model file path
state_dict = './model/model1.pth'

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
  model.load_state_dict(torch.load(state_dict))

  # Set themodel in evaluation mode
  model.eval()

  # Inizializate variables to compute evaluation metrics
  test_accuracy = 0.0
  target = []
  pred = np.empty((0,43), float)

  # Open a CSV file
  output_file = open('wrong1.csv', 'w')
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
