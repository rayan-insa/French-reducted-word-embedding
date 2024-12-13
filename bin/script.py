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

modelBinPath = 'bin/modelsSavedLocally/cc.fr.300.bin'

def loadFastTextBaseModel():
    """
    Loads the FastText model for French language.
    If a saved model exists, it loads the model from the saved file.
    Otherwise, it downloads the original FastText model, saves it for future use, and then loads it.
    Returns:
        fasttext.FastText._FastText: The loaded FastText model.
    """
    if os.path.exists(modelBinPath):
        print("Loading saved FastText model...")
    else:
        print("Loading original FastText model...")
        fasttext.util.download_model('fr', if_exists='ignore')
        print("Saving model for future use...")
    
    model = fasttext.load_model(modelBinPath)
    return model

fastTextBaseModel = loadFastTextBaseModel()
entire_vocabulary = fastTextBaseModel.get_words()

# Selection of 10000 most-used words from the vocabulary
sample_vocabulary = entire_vocabulary[:10000]
embedding_matrix = np.array([fastTextBaseModel.get_word_vector(word) for word in sample_vocabulary])

# Normalization and fitting to pytorch of the embedding matrix
scaler = StandardScaler()
embedding_matrix_normalized = scaler.fit_transform(embedding_matrix)
train_data, test_data = train_test_split(embedding_matrix_normalized, test_size=0.2, random_state=42)
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# Batching
batch_size = 32
train_dataset = TensorDataset(train_tensor)
test_dataset = TensorDataset(test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, and optimizer
autoencoder = AutoEncoder(input_dim=300, hidden_dim1=256, hidden_dim2=128, bottleneck_dim=30)

def autoEncoderTraining() :
    print("Training the AutoEncoder...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

    # Early stopping parameters
    patience = 10
    min_delta = 0.001
    best_loss = float('inf')
    early_stop_counter = 0

    # Training loop
    num_epochs = 200
    train_losses = []
    test_losses = []

    def evaluate(model, dataloader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_data = batch[0]
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_data)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_train_loss = 0
        for batch in train_dataloader:
            batch_data = batch[0]

            optimizer.zero_grad()
            reconstructed = autoencoder(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_dataloader))
        test_loss = evaluate(autoencoder, test_dataloader, criterion)
        test_losses.append(test_loss)

        # Early stopping logic
        if test_loss < best_loss - min_delta:
            best_loss = test_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")

    # Plot loss curves
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(test_losses)), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print("AutoEncoder training completed.")

autoEncoderTraining()