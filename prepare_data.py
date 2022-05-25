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

# Save in a NPY file captum attributes
with open('data.npy', 'wb') as f:
    for subdirs, dirs, files in os.walk('.\explainability\\'):
        for filename in files:
            if filename.endswith('.npy'):
                # print('filename', filename)
                print(subdirs + '\\' + filename)
                with open(subdirs + '\\' + filename, 'rb') as g:
                    for c in classes:
                        a = np.load(g, allow_pickle=True)
                        np.save(f, a)
                        # print(a[16,16,-1])