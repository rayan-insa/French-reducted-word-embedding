import os
import fasttext.util
import random

class DataSamplization():
    def __init__(self, sampleNum=10000, modelBinPath='bin/modelsSavedLocally/cc.fr.300.bin', csvSavePath='bin/data/'):
        """
        Initializes the DataSamplization class.
        Args:
            sampleNum (int): Number of most-used words to sample from the vocabulary (default: 10000).
            modelBinPath (str): Path to the FastText model binary file (default: 'bin/modelsSavedLocally/cc.fr.300.bin').
        """

        self.modelBinPath = modelBinPath
        self.sampleNum = sampleNum
        self.csvSaveFile = csvSavePath
        self.fastTextBaseModel = self.loadFastTextBaseModel()


    def loadFastTextBaseModel(self):
        """
        Loads the FastText model for French language.
        If a saved model exists, it loads the model from the saved file.
        Otherwise, it downloads the original FastText model, saves it for future use, and then loads it.
        Returns:
            fasttext.FastText._FastText: The loaded FastText model.
        """
        if os.path.exists(self.modelBinPath):
            print("Loading saved FastText model...")
        else:
            print("Loading original FastText model...")
            fasttext.util.download_model('fr', if_exists='ignore')
            print("Saving model for future use...")
        
        model = fasttext.load_model(self.modelBinPath)
        return model
    
    def entireVocabulary(self):
        """
        Returns the entire vocabulary of the FastText model.
        """

        return self.fastTextBaseModel.get_words()
    
    def mostUsedDataSample(self):
        """
        Selects sampleNum most-used words from the vocabulary.
        Returns:
            list: Sample of most-used words from the vocabulary.
        """

        self.entire_vocabulary = self.entireVocabulary()
        # Excluding the 150 first most-used words because most of them are simple characters such as ()[]{},"';:/" etc
        self.sample_vocabulary = self.entire_vocabulary[150:self.sampleNum+150]
        self.csvSaveFile = self.csvSaveFile
        fileName = "most_used_words.csv"
        self.dataSampleIntoCSV(sampleVocabulary=self.sample_vocabulary, fileName=fileName)
        return self.sample_vocabulary
    
    def randomWordsDataSample(self):
        """
        Selects sampleNum random words from the vocabulary.
        Returns:
            list: Sample of random words from the vocabulary.
        """

        self.entire_vocabulary = self.entireVocabulary()
        self.sample_vocabulary = random.sample(self.entire_vocabulary, self.sampleNum)
        self.csvSaveFile = self.csvSaveFile
        fileName = "random_words.csv"
        self.dataSampleIntoCSV(sampleVocabulary=self.sample_vocabulary, fileName=fileName)
        return self.sample_vocabulary
    
    
    def dataSampleIntoCSV(self, sampleVocabulary, fileName):
        """
        Creates a CSV file with a sample of sampleNum most-used words from the vocabulary.
        """

        with open(self.csvSaveFile+fileName, 'w') as f:
            f.write("id,word\n")
            for index, word in enumerate(sampleVocabulary):
                f.write(f"{index},{word}\n")
        print("Vocabulary sample saved into vocabulary_sample.csv")
