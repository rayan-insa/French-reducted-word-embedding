import os
import fasttext.util
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

class DataSamplization():
    def __init__(self, sampleNum=10000):
        """
        Initializes the DataSamplization class.
        Args:
            sampleNum (int): Number of most-used words to sample from the vocabulary (default: 10000).
            modelBinPath (str): Path to the FastText model binary file (default: 'bin/modelsSavedLocally/cc.fr.300.bin').
        """
        os.makedirs("data/modelsSavedLocally", exist_ok=True)
        self.modelBinPath = 'data/modelsSavedLocally'
        self.sampleNum = sampleNum
        self.csvSaveFile = 'data/vocab'
        self.fastTextBaseModel = self.loadFastTextBaseModel()

    def getWordId(self, word):
        """
        Returns the ID of the word in the vocabulary.
        Args:
            word (str): The word to get the ID of.
        Returns:
            int: The ID of the word.
        """
        return self.fastTextBaseModel.get_word_id(word)


    def loadFastTextBaseModel(self):
        """
        Loads the FastText model for French language.
        If a saved model exists, it loads the model from the saved file.
        Otherwise, it downloads the original FastText model, saves it for future use, and then loads it.
        Returns:
            fasttext.FastText._FastText: The loaded FastText model.
        """
        
        if os.path.exists(self.modelBinPath+"/cc.fr.300.bin"):
            print("Loading saved FastText model...")
        else:
            print("Loading original FastText model...")
            fasttext.util.download_model('fr', if_exists='ignore')
            print("Saving model for future use...")
        
        model = fasttext.load_model(self.modelBinPath+"/cc.fr.300.bin")
        return model
    
    def entireVocabulary(self):
        """
        Returns the entire vocabulary of the FastText model.
        """
        self.entireVocab = self.fastTextBaseModel.get_words()
        fileName = "entire_vocabulary.csv"
        self.dataSampleIntoCSV(sampleVocabulary=self.entireVocab, fileName=fileName)
        return self.entireVocab
    
    def mostUsedDataSample(self, sampleNum):
        """
        Selects sampleNum most-used words from the vocabulary.
        Returns:
            list: Sample of most-used words from the vocabulary.
        """
        self.sampleNum = sampleNum
        self.entire_vocabulary = self.fastTextBaseModel.get_words()
        # Excluding the 150 first most-used words because most of them are simple characters such as ()[]{},"';:/" etc
        self.sample_vocabulary = self.entire_vocabulary[150:self.sampleNum+150]
        fileName = "most_used_words.csv"
        self.dataSampleIntoCSV(sampleVocabulary=self.sample_vocabulary, fileName=fileName)
        return self.sample_vocabulary
    
    def randomWordsDataSample(self, sampleNum):
        """
        Selects sampleNum random words from the vocabulary.
        Returns:
            list: Sample of random words from the vocabulary.
        """
        self.sampleNum = sampleNum
        self.entire_vocabulary = self.fastTextBaseModel.get_words()
        self.sample_vocabulary = random.sample(self.entire_vocabulary, self.sampleNum)
        fileName = "random_words.csv"
        self.dataSampleIntoCSV(sampleVocabulary=self.sample_vocabulary, fileName=fileName)
        return self.sample_vocabulary
    
    
    def dataSampleIntoCSV(self, sampleVocabulary, fileName):
        """
        Creates a CSV file with a sample of sampleNum most-used words from the vocabulary.
        """
        os.makedirs(self.csvSaveFile, exist_ok=True)
        if not os.path.exists(self.csvSaveFile+fileName):
            with open(self.csvSaveFile+fileName, 'w') as f:
                f.write("id,word\n")
                for index, word in enumerate(sampleVocabulary):
                    f.write(f"{index},{word}\n")
            print(f"Vocabulary sample saved into {self.csvSaveFile+fileName}")
        else:
            print(f"{self.csvSaveFile+fileName} already exists.")
    
    def getWordsEmbeddings(self, words):
        """
        Returns the exact same words as the input words.
        Args:
            words (list(str)): The words to find the exact same words of.
        Returns:
            list(str): The embeddings of the exact same words as the input words.
        """
        sameWordsEmbeddings = []
        alreadySeen = []
        for word in words:
            if word not in alreadySeen:
                alreadySeen.append(word)
                sameWordsEmbeddings.append(self.fastTextBaseModel.get_word_vector(word))
        return sameWordsEmbeddings

    def getWordsIds(self, words):
        """
        Returns the IDs of the words given.
        Args:
            words (list(str)): The words to get the IDs of.
        Returns:
            list(int): The IDs of the words
        """

        IdsList = []
        for word in words:
            IdsList.append(self.fastTextBaseModel.get_word_id(word))
        return IdsList
    
    def getWordsIdsWihtoutRepeat(self, words):
        """
        Returns the IDs of the words given without repeating the same word.
        Args:
            words (list(str)): The words to get the IDs of.
        Returns:
            list(int): The IDs of the words
        """

        IdsList = []
        alreadySeen = []
        for word in words:
            if word not in alreadySeen:
                alreadySeen.append(word)
                IdsList.append(self.fastTextBaseModel.get_word_id(word))
        return IdsList
    

def dataBatching(train_data, test_data, batch_size):
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def autoEncoderTraining(num_epochs, patience, min_delta, lr, autoEncoder, train_dataloader, test_dataloader):

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
        if min_delta is not None and patience is not None:
            if test_loss < best_loss - min_delta:
                best_loss = test_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")

    print("AutoEncoder training completed.")
    return train_losses, test_losses, bottleneck_outputs


# Example training loop
def word2vecFineTuning(model, dataloader, epochs=150, lr=0.001):
    print("\nFine-tuning the Embeddings... please wait...")
    patience=5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for target_ids, context_ids in dataloader:
            target_ids = target_ids.long()
            context_ids = context_ids.long()
            
            optimizer.zero_grad()
            scores = model(target_ids)
            
            loss = criterion(scores, context_ids)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        if epoch >= 50 and (best_loss - avg_loss) < 0.4:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    print("Fine-tuning completed.\n")
    return model


def evaluate_word_similarity(embedding_dict, test_pairs):
    """
    Evaluate cosine similarity for test word pairs.
    :param embedding_dict: Dict of word -> embedding.
    :param test_pairs: List of (word1, word2) tuples.
    :return: List of cosine similarity scores.
    """
    similarities = []
    for word1, word2 in test_pairs:
        if word1 in embedding_dict and word2 in embedding_dict:
            vec1 = embedding_dict[word1]
            vec2 = embedding_dict[word2]
            sim = cosine_similarity([vec1], [vec2])[0][0]
            similarities.append(sim)
        else:
            similarities.append(None)  # Handle missing words gracefully
            print(f"Missing word pair: {word1}, {word2}")
    return similarities

def PCA_visualize_embeddings(embedding_dicts, words, titles=None, withLabels=True):
    """
    Visualize embeddings in 2D using PCA for multiple embedding dictionaries.
    :param embedding_dicts: List of dictionaries, each containing word -> embedding.
    :param words: List of words to visualize.
    :param titles: List of titles for each embedding_dict (optional).
    """
    # If no titles provided, generate default titles
    if titles is None:
        titles = [f"Embedding {i+1}" for i in range(len(embedding_dicts))]

    # Create colors for different embedding dictionaries
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    plt.figure(figsize=(10, 8))
    
    for i, embedding_dict in enumerate(embedding_dicts):
        selected_embeddings = np.array([embedding_dict[word] for word in words if word in embedding_dict])
        selected_words = [word for word in words if word in embedding_dict]
        
        # Perform PCA reduction to 2D
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(selected_embeddings)
        
        # Plot the reduced embeddings with a unique color for each embedding_dict
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color=colors[i % len(colors)], label=titles[i])
        
        # Add labels for each word
        if withLabels:
            for j, word in enumerate(selected_words):
                plt.text(reduced_embeddings[j, 0], reduced_embeddings[j, 1], word, fontsize=12)

    plt.title("PCA Visualization for Multiple Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.legend()
    plt.show()


def plot_similarity_comparison(similarities_list, labels):
    """
    Plot a comparison of average similarities across configurations.
    :param similarities_list: List of similarity scores for each configuration.
    :param labels: Configuration names.
    """
    # Filtrer les listes pour exclure les valeurs None
    cleaned_similarities_list = [
        [sim for sim in similarities if sim is not None]
        for similarities in similarities_list
    ]
    
    avg_similarities = [np.nanmean(similarities) for similarities in cleaned_similarities_list]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_similarities, color='skyblue')
    plt.xlabel('Configurations')
    plt.ylabel('Average Similarity')
    plt.title('Comparison of Average Similarity Scores')
    plt.show()



def tSNE_visualize_embeddings(embedding_dicts, words, titles=None, perplexity=30, n_iter=1000, withLabels=True):
    """
    Visualize embeddings in 2D using t-SNE for multiple embedding dictionaries.
    :param embedding_dicts: List of dictionaries, each containing word -> embedding.
    :param words: List of words to visualize.
    :param titles: List of titles for each embedding_dict (optional).
    :param perplexity: Perplexity parameter for t-SNE.
    :param n_iter: Number of iterations for optimization.
    """
    # If no titles provided, generate default titles
    if titles is None:
        titles = [f"Embedding {i+1}" for i in range(len(embedding_dicts))]

    # Create colors for different embedding dictionaries
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    plt.figure(figsize=(10, 8))
    
    for i, embedding_dict in enumerate(embedding_dicts):
        selected_embeddings = np.array([embedding_dict[word] for word in words if word in embedding_dict])
        selected_words = [word for word in words if word in embedding_dict]

        # Adjust perplexity to avoid errors
        adjusted_perplexity = min(perplexity, len(selected_words) - 1)
        
        # Perform t-SNE reduction to 2D
        tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, n_iter=n_iter, random_state=42)
        reduced_embeddings = tsne.fit_transform(selected_embeddings)
        
        # Plot the reduced embeddings with a unique color for each embedding_dict
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color=colors[i % len(colors)], label=titles[i])
        
        if withLabels:
            # Add labels for each word
            for j, word in enumerate(selected_words):
                plt.text(reduced_embeddings[j, 0], reduced_embeddings[j, 1], word, fontsize=12)

    plt.title("t-SNE Visualization for Multiple Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid()
    plt.legend()
    plt.show()