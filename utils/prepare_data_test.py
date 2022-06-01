# %%

# Prepare the data to train the second model
import numpy as np
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

# A support np array x to start the concatenation
x = np.empty((0, 32, 32, 129), float)

# A support np array y to start the concatenation for labels
y = []

# Load np array with labels that need to be re-ordered depending on the order in which we concatenate the samples 
z = np.load('labels.npy', allow_pickle=True)

# DIRECTORY STRUCTURE
# explainability (root) -> sample_0 (directory) -> data0.npy (file with 43 np arrays)
#                       -> sample_1 (directory) -> data1.npy (file)
#                       -> sample_i (directory) -> datai.npy (file)
#                       -> sample_12659 (directory) -> data12659.npy (file)

# Concatenate np arrays of captum results and put labels in correct order
for root, dirs, files in os.walk('.\explainability_test\\'):  
    for filename in files:
        if filename.endswith('.npy'):
            # Open npy file
            with open(root + '\\' + filename, 'rb') as g:
                first_time = True
                # Concatenate the 43 arrays (32,32,3) on axis=2 --> (32,32,43x3)
                for id, classe in enumerate(classes):
                    a = np.load(g, allow_pickle=True)
                    if first_time:
                        b = np.copy(a)
                        first_time = False
                    else:
                        b = np.concatenate((b, a), axis = 2) 
                # Concatenate in a global array x with initial shape 1x32x32x129
                x = np.concatenate((x, [b]))
                print(x.shape)
                # Concatenate the new ordered labels
                y = np.append(y, np.array(z[int(filename[4:-4])]))
                print(y.shape)

# Save final arrays
np.save('data_test.npy', x) # final x.shape = 12630x32x32x129 (num_samples, height, width, channels)
np.save('labels_test.npy', y) # final y.shape = 12630

# %%
