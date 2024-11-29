# Name this file assignment4.py when you submit

import math
import os

class bag_of_words_model:

    def __init__(self, directory):
        self.directory = directory 

    def tf_idf(self, document_filepath):
       
        vocabulary = set()
        for file_name in os.listdir(self.directory):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.directory, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    words = f.read().strip().lower().split()
                    vocabulary.update(words)
        vocabulary = sorted(vocabulary)

       
        total_docs = len([file for file in os.listdir(self.directory) if file.endswith(".txt")])
        idf = {}
        for word in vocabulary:
            doc_count = 0
            for file_name in os.listdir(self.directory):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(self.directory, file_name)
                    with open(file_path, 'r') as f:
                        if word in f.read().lower().split():
                            doc_count += 1
            idf[word] = math.log2(total_docs / doc_count) if doc_count > 0 else 0

        with open(document_filepath, 'r') as f:
            words = f.read().strip().lower().split()
        tf = {}
        total_words = len(words)
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        for word in tf:
            tf[word] /= total_words

        # Calculate TF-IDF vector
        tf_idf_vector = [
            tf.get(word, 0) * idf.get(word, 0) for word in vocabulary
        ]
        return tf_idf_vector

    def predict(self, document_filepath, weights_filepath):
       
        tf_idf_vector = self.tf_idf(document_filepath)

        
        with open(weights_filepath, 'r', encoding='utf-8') as f:
            weights = [float(w) for w in f.read().strip().split(",")]

        z = sum(tf_idf_vector[i] * weights[i] for i in range(len(weights)))

        prediction = 1 / (1 + math.exp(-z))
        return prediction




def main():
    training_directory = r"C:\Users\haroo\Desktop\AIA4\Examples_files\Examples\Example2\training_documents"
    test_document_path = r"C:\Users\haroo\Desktop\AIA4\Examples_files\Examples\Example2\test_document.txt"
    weights_filepath = r"C:\Users\haroo\Desktop\AIA4\Examples_files\Examples\Example2\weights.txt"
    model = bag_of_words_model(training_directory)

    tf_idf_vector = model.tf_idf(test_document_path)
    print("TF-IDF Vector:", tf_idf_vector)

    prediction = model.predict(test_document_path, weights_filepath)
    print("Final Prediction:", prediction)
main()