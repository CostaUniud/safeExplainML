#%%

import torch
import torchvision
from model import Net
import numpy as np
import os

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

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

  test_set2 = torchvision.datasets.GTSRB(
    root = './data',
    split = 'test',
    download = False
  )

  print('Number of test images: {}'.format(len(test_set)))

  # Load data from disk and organize it in batches
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

  # Instantiate model structure
  model = Net(False)

  # Move the model to the right device
  model = model.to(device)

  # Load the model from file
  model.load_state_dict(torch.load('./model/' + state_dict))

  # Set themodel in evaluation mode
  model.eval()

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

    input = x
    input.requires_grad = True

    model.eval()

    original_image = torch.permute(unnormalize(input[0].cpu().detach()), (1, 2, 0)).numpy()

    # Create folder to save captum results
    directory = 'sample_' + str(idx)
    parent_dir = 'explainability/'
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    # Save captum results on npy files
    with open(path + '/data' + str(idx) + '.npy', 'wb') as f:
      # For each class
      for id, c in enumerate(classes):

        # Create sub-folder to save captum results
        sub_directory = 'class_' + str(id)
        parent_dir = path
        sub_path = os.path.join(parent_dir, sub_directory)
        os.mkdir(sub_path)

        # Computes gradients with respect to class ind and transposes them for visualization purposes
        # saliency = Saliency(model)
        # grads = saliency.attribute(input, target = id)
        # grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        # np.save(f, grads)

        # Applies integrated gradients attribution algorithm on test image
        ig = IntegratedGradients(model)
        attr_ig, delta = attribute_image_features(ig, input, id, baselines=input * 0, return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        np.save(f, attr_ig)

        # Use integrated gradients and noise tunnel with smoothgrad square option on the test image
        # ig = IntegratedGradients(model)
        # nt = NoiseTunnel(ig)
        # attr_ig_nt = attribute_image_features(nt, input, id, baselines=input * 0, nt_type='smoothgrad_sq', nt_samples=100, stdevs=0.2)
        # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # np.save(f, attr_ig_nt)

        # # Applies DeepLift on test image
        # dl = DeepLift(model)
        # attr_dl = attribute_image_features(dl, input, id, baselines=input * 0)
        # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # np.save(f, attr_dl)

        # Visualize and save the original image and the attributions for Saliency Maps, DeepLift, Integrated Gradients and Integrated Gradients with SmoothGrad
        # _ = viz.visualize_image_attr(None, np.array(test_set2[idx][0]), method="original_image", title="Original Image")
        # _[0].savefig(sub_path + '/original_image.png')
        # _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
        #                         show_colorbar=True, title="Overlayed Gradient Magnitudes")
        # _[0].savefig(sub_path + '/overlayed_gradient_magnitudes.png')
        _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all",
                                show_colorbar=True, title="Overlayed Integrated Gradients")
        _[0].savefig(sub_path + '/overlayed_integrated_gradients.png')
        # _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
        #                             outlier_perc=10, show_colorbar=True, 
        #                             title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
        # _[0].savefig(sub_path + '/overlayed_integrated_gradients_with_smoothGrad_squared.png')
        # _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all", show_colorbar=True, 
        #                         title="Overlayed DeepLift")
        # _[0].savefig(sub_path + '/overlayed_deepLift')

# %%
