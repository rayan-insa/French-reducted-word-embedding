import os
import fasttext.util
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder
from dataSamplization import DataSamplization
from autoEncoderTrainingLoop import autoEncoderTraining
from dataBatching import dataBatching

modelBinPath = 'bin/modelsSavedLocally/cc.fr.300.bin'
sampleNum = 10000

# Data samplization
dataSamplization = DataSamplization(sampleNum=sampleNum, modelBinPath=modelBinPath)
sample_vocabulary = dataSamplization.mostUsedDataSample()
embedding_matrix = np.array([dataSamplization.fastTextBaseModel.get_word_vector(word) for word in sample_vocabulary])

# Normalization and fitting to pytorch of the embedding matrix
scaler = StandardScaler()
embedding_matrix_normalized = scaler.fit_transform(embedding_matrix)
train_data, test_data = train_test_split(embedding_matrix_normalized, test_size=0.2, random_state=42)
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# Batching
batch_size = 32


train_dataloader, test_dataloader = dataBatching(train_tensor, test_tensor, batch_size)

# Model, loss function, and optimizer
autoencoder = AutoEncoder(input_dim=300, hidden_dim1=256, hidden_dim2=128, bottleneck_dim=30)

# Training parameters
num_epochs = 100
# Early stopping parameters
patience = 10
min_delta = 0.001


autoEncoderTraining(num_epochs=100, patience=10, min_delta=0.001, lr=0.001, autoEncoder=autoencoder, train_dataloader=train_dataloader, test_dataloader=test_dataloader)