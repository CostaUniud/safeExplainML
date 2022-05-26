# %%
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

c = np.arange(132096) # DA ELIMINARE ALLA FINE PRIMA DEL TRAINING
c = c.reshape((1, 32, 32, 129))
c = np.zeros_like(c)

l = np.array([0]) # DA ELIMINARE ALLA FINE PRIMA DEL TRAINING

# Save in a NPY file captum attributes
with open('data.npy', 'wb') as f:
    for root, dirs, files in os.walk('.\explainability\\'):
        for filename in files:
            if filename.endswith('.npy'):
                # print('filename', filename)
                # print(root + '\\' + filename)
                with open(root + '\\' + filename, 'rb') as g:
                    first_time = True
                    for id, classe in enumerate(classes):
                        a = np.load(g, allow_pickle=True)
                        if first_time:
                            b = np.copy(a)
                            first_time = False
                        else:
                            b = np.concatenate((b, a), axis=2)
                    # print(b.shape)
                    c = np.concatenate((c, [b]))
                    print(c.shape)
                    x = np.load('labels_new.npy', allow_pickle=True)
                    # print(x)
                    l = np.concatenate((l, np.array([x[int(filename[4:-4])]])))
                    print(l.shape)
                    # print(l)
    np.save(f, c)
np.save('labels.npy', l)

# %%
