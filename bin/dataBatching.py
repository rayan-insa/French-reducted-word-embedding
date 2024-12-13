from torch.utils.data import DataLoader, TensorDataset

def dataBatching(train_data, test_data, batch_size):
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader