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

  # Unnormalize transforms
  unnormalize = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((0., 0., 0.), (1/0.2672, 1/0.2564, 1/0.2629)),
    torchvision.transforms.Normalize((-0.3337, -0.3064, -0.3171), (1., 1., 1.)),
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
  model2 = model.to(device)

  # Load the model from file
  model.load_state_dict(torch.load(state_dict))
  model2.load_state_dict(torch.load(state_dict2))

  # Set the model 2 in evaluation mode
  model2.eval()

  # Open a CSV file
  output_file = open('comparison.csv', 'w')
  output_file.write('Filename,ClassId,Pred_1,Conf_1,Pred_2,Conf_2\n')

  # Evaluate the model
  for idx, batch in enumerate(test_loader):
    x, y = batch

    # Move the data to the right device
    x, y = x.to(device), y.to(device)

    # Perform the forward pass with Net
    out = model(x)

    # Get predictions values model 1
    probs = torch.softmax(out, dim=1)
    conf, pred_idxs = torch.max(probs.data, 1)

    x.requires_grad = True

    # Set the model in evaluation mode
    model.eval()

    b = None

    # For each class
    for id, c in enumerate(classes):

      # Applies integrated gradients attribution algorithm on test image
      ig = IntegratedGradients(model)
      attr_ig, delta = attribute_image_features(ig, x, id, baselines=x * 0, return_convergence_delta=True)
      attr_ig = attr_ig.squeeze().cpu().detach()

      if b is None:
        b = attr_ig.to(device)
      else:
        b = torch.cat((b, attr_ig))

    out2 = model2(b)
    
    # Get predictions values model 2
    probs2 = torch.softmax(out2, dim=1)
    conf2, pred_idxs2 = torch.max(probs2.data, 1)

  # Write info about predction on CSV file
  output_file.write("%d,%s,%s,%f,%s,%f\n" % (idx, classes[y], classes[pred_idxs], conf, classes[pred_idxs2], conf2))

  # Close CSV
  output_file.close()







# %%
