# %%

# Prepare the data to train the second model
import numpy as np
import torch
import os

# Classes (43) which images belong to
classes = ('Limit 20km', 'Limit 30km', 'Limit 50km', 'Limit 60km', 'Limit 70km', 'Limit 80km', 
        'End limit 80km', 'Limit 100km', 'Limit 120km', 'No overtaking', 'No overtaking of heavy vehicles', 
        'Intersection with right of way', 'Right of way', 'Give right to pass', 'Stop', 'Transit prohibition', 
        'Prohibition of heavy vehicles transit', 'Wrong way', 'Generic danger', 'Dangerous left curve', 'Dangerous right curve', 
        'Double curve', 'Danger of bumps', 'Danger of slipping', 'Asymmetrical bottleneck', 'Men at work', 'Traffic light', 
        'Pedestrian crossing', 'School', 'Cycle crossing', 'Snow', 'Wild animals', 'Go ahead', 'Right turn mandatory', 
        'Left turn mandatory', 'Mandatory direction straight', 'Directions right and straight', 'Directions left and straight', 
        'Mandatory step to the right', 'Mandatory step to the left', 'Roundabout', 'End of no overtaking', 'End of no overtaking of heavy vehicles')
# Main
if __name__ == '__main__':
    # Define the device where we want run our model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # A support variable x to start the concatenation for data
    x = None

    # A support variable y to start the concatenation for labels
    y = None

    # Load np array with labels that need to be re-ordered depending on the order in which we concatenate the samples 
    z = np.load('../labels_train.npy', allow_pickle=True)
    z = torch.from_numpy(z).long().to(device)

    # Concatenate np arrays of captum results and put labels in correct order
    for id, filename in enumerate(os.listdir('../explainability_train')):
        print(filename)
        if filename.endswith('.npy'):
            # Open npy file with data to concatenate
            with open('../explainability_train/' + filename, 'rb') as g:
                # A support variable b to start the concatenation
                b = None
                # Concatenate the 43 arrays (32,32,3) on axis=2 --> (32,32,43x3)
                for classe in classes:
                    a = np.load(g, allow_pickle=True)
                    a = torch.from_numpy(a).float().to(device)
                    if b is None:
                        b = a.to(device)
                    else:
                        b = torch.cat((b, a), axis=2)
                        # print(b.shape)
                # Concatenate in a global array x with initial shape 1x32x32x129
                b = b.unsqueeze(0)
                if x is None:
                    x = b
                else:
                    x = torch.vstack((x, b))
                print(x.shape)
                # Concatenate the new ordered labels
                if y is None:
                    y = torch.FloatTensor([z[int(filename[4:-4])]], device=device)
                else:
                    y = torch.cat((y, torch.FloatTensor([z[int(filename[4:-4])]], device=device)))
                print(y.shape)
                # Save intermediate tensors
                if id != 0 and (id % 5000) == 0:
                    torch.save(x, '../data_train_def_' + str(id) + '.pt')
                    torch.save(y, '../labels_train_def_' + str(id) + '.pt')

    # Save final tensors
    torch.save(x, '../data_train_def.pt') # final x.shape = 12630x32x32x129 (num_samples, height, width, channels)
    torch.save(y, '../labels_train_def.pt') # final y.shape = 12630

# %%
