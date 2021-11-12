from transformers import BertTokenizer, BertModel
import fasttext
import fasttext.util
import numpy as np


class BERT2rep:
    def __init__(self, model_name):
        self.model_name = model_name
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).cuda()
        self.bert.eval()

    def generate_rep(self, texts):
        encoded_input = self.bert_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to('cuda:0')
        output = self.bert(**encoded_input)
        return output[0][:,0,:]


class fastText2rep:
    def __init__(self, model_name):
        self.model_name = model_name
        self.ft = fasttext.load_model(model_name)

    def generate_rep(self, texts):
        if len(texts) == 0:
            return np.zeros(300)
        reps = []
        for text in texts:
            reps.append(self.ft.get_word_vector(text))
        return np.mean(reps, axis=0)
