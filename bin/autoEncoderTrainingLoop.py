def autoEncoderTraining(num_epochs, patience, min_delta, lr, autoEncoder, train_dataloader, test_dataloader):
    import torch.nn as nn
    import torch.optim as optim
    import torch

    print("Training the AutoEncoder...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoEncoder.parameters(), lr)

    best_loss = float('inf')
    early_stop_counter = 0

    # Training and evaluation metrics
    train_losses = []
    test_losses = []
    bottleneck_outputs = None

    # Evaluation function
    def evaluate(model, dataloader, criterion):
        model.eval()
        total_loss = 0
        outputs = []
        with torch.no_grad():
            for batch in dataloader:
                batch_data = batch[0]
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_data)
                total_loss += loss.item()
                
                # Save bottleneck outputs
                outputs.append(model.bottleneck_output)
        return total_loss / len(dataloader), torch.cat(outputs)

    # Training loop
    for epoch in range(num_epochs):
        autoEncoder.train()
        epoch_train_loss = 0
        for batch in train_dataloader:
            batch_data = batch[0]

            optimizer.zero_grad()
            reconstructed = autoEncoder(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_dataloader))
        test_loss, bottleneck_outputs_current = evaluate(autoEncoder, test_dataloader, criterion)
        test_losses.append(test_loss)

        # Update bottleneck outputs
        bottleneck_outputs = bottleneck_outputs_current

        # Early stopping logic
        if test_loss < best_loss - min_delta:
            best_loss = test_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")

    print("AutoEncoder training completed.")
    return train_losses, test_losses, bottleneck_outputs
