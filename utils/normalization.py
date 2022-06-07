# %%
# Calculate mean and standard devation on a dataset
import torch
from torch.utils.data import DataLoader
import numpy as np

def get_mean_and_std(dataloader):
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0
  for data, _ in dataloader:
    data = torch.permute(data, (0, 3, 1, 2)) # 129x32x32
    # Mean over batch, height and width, but not over the channels
    channels_sum += torch.mean(data, dim=[0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1
  
  mean = channels_sum / num_batches

  # std = sqrt(E[X^2] - (E[X])^2)
  std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

  return mean, std

train_set = torch.utils.data.TensorDataset(torch.load('../data_train_def_5000.pt'), torch.load('../labels_train_def_5000.pt'))

train_dataloader = DataLoader(dataset=train_set, batch_size=64)

print(get_mean_and_std(train_dataloader))
# %%
