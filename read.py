import numpy as np

# with open('test.npy', 'wb') as f:
#     np.save(f, np.array([1, 2]))
#     np.save(f, np.array([1, 3]))

# with open('test.npy', 'rb') as f:
#     a = np.load(f)
#     b = np.load(f)

# print(a, b)

with open('.\explainability\sample_0\data0.npy', 'rb') as f:
    a = np.load(f, allow_pickle=True)
    b = np.load(f, allow_pickle=True)
    c = np.load(f, allow_pickle=True)
    d = np.load(f, allow_pickle=True)
print(a, b, c, d)