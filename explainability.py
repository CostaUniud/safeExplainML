#%%
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

from model import Net

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

# Model file
state_dict = 'model_2.pth'

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
def imshow(img, transpose = True):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Calling attribute on attribution algorithm defined in input
def attribute_image_features(algorithm, input, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=labels[ind], **kwargs)
    
    return tensor_attributions

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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

    # Instantiate model structure
    model = Net(False)

    # Move the model to the right device
    model = model.to(device)

    # Load the model from file
    model.load_state_dict(torch.load('./model/' + state_dict))

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # Show some images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Perform the forward pass with Net
    out = model(images)

    # Get predictions values
    _, predicted = torch.max(out, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # Choose a test image at index ind
    ind = 0
    input = images[ind].unsqueeze(0)
    input.requires_grad = True

    # Sets model to eval mode for interpretation purposes
    model.eval()

    # Computes gradients with respect to class ind and transposes them for visualization purposes
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=labels[ind].item())
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # Applies integrated gradients attribution algorithm on test image
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    # Use integrated gradients and noise tunnel with smoothgrad square option on the test image
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq', nt_samples=100, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # Applies DeepLift on test image
    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # Visualize the attributions for Saliency Maps, DeepLift, Integrated Gradients and Integrated Gradients with SmoothGrad
    print('Original Image')
    print('Predicted:', classes[predicted[ind]], ' Probability:', torch.max(F.softmax(out, 1)).item())

    original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    _ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")

    _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                            show_colorbar=True, title="Overlayed Gradient Magnitudes")

    _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                            show_colorbar=True, title="Overlayed Integrated Gradients")

    _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
                                outlier_perc=10, show_colorbar=True, 
                                title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

    _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                            title="Overlayed DeepLift")
                            
# %%
