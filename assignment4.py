

import math
import os

class bag_of_words_model:

    def __init__(self, directory):
        self.directory = directory 

    def tf_idf(self, document_filepath):
       # put all the unqiue words in a  set
        givenwords = set()
        for i in os.listdir(self.directory):
            if ".txt" in i[-4:]:
                file_path = os.path.join(self.directory, i)
                with open(file_path, 'r') as f:
                    words = f.read().strip().lower().split()
                    givenwords.update(words)
        givenwords = sorted(givenwords)
    #check if given file has txt
        total_txt_files = 0
        for i in os.listdir(self.directory):
            if ".txt" in i[-4:]:
                total_txt_files += 1
        total_docs = total_txt_files
    #count the total number of text files
        idf = {}
        for phrase in givenwords:
            occurance = 0
            for i in os.listdir(self.directory):
                if ".txt" in i[-4:]:
                    file_path = os.path.join(self.directory, i)
                    with open(file_path, 'r') as f:
                        if phrase in f.read().lower().split():
                            occurance += 1
            #idf formula
            idf[phrase] = math.log2(total_docs / occurance) 

 
        with open(document_filepath, 'r') as f:
            file= f.read()
            file = file.strip()
            file =  file.split()
            words = file

        termfreq = {}
    #get term frequency for the words in teh doc
        for phrase in words:
            termfreq[phrase] = termfreq.get(phrase, 0) + 1
        for phrase in termfreq:
            termfreq[phrase] /= len(words)

        v = []
        for phrase in givenwords:
            tfvalue = termfreq.get(phrase, 0)  
            idfvalue = idf.get(phrase, 0)  
            v.append(tfvalue * idfvalue)  

        return v

    def predict(self, document_filepath, weights):
       
        v = self.tf_idf(document_filepath)
        with open(weights, 'r') as f:
            file_content = f.read()
            file_content = file_content.strip()
            file_content = file_content.split(",")
            weights = []
            for i in file_content:
                weights.append(float(i))
        counter = 0
        for i in range(len(weights)):
            tf_idf_value = v[i]
            weight_value = weights[i]
            
            counter += tf_idf_value * weight_value

        prediction = 1 / (1 + math.exp(-counter))
        return prediction
    
# USED LLM FOR GENERAL CONCEPTS ON tf_idf PYTHON CODE and how to loop through an os as permission given in course outline.



def main():
    training_directory = r"C:\Users\haroo\Desktop\AIA4\Examples_files\Examples\Example2\training_documents"
    test_document_path = r"C:\Users\haroo\Desktop\AIA4\Examples_files\Examples\Example2\test_document.txt"
    weights_filepath = r"C:\Users\haroo\Desktop\AIA4\Examples_files\Examples\Example2\weights.txt"
    model = bag_of_words_model(training_directory)

    v = model.tf_idf(test_document_path)
    print("V:", v)

    prediction = model.predict(test_document_path, weights_filepath)
    print("Prediction:", prediction)
main()
