# coding=utf-8
import artm
import subprocess
import pandas as pd
import re
from nltk.corpus import stopwords
import pymorphy2


def preprocessing_for_artm(number_of_docs=10):
    data = pd.read_csv("../data/lenta_ru.csv")
    texts = data["text"]
    doc_text = ""
    morph = pymorphy2.MorphAnalyzer()
    for i in range(number_of_docs):
        text = texts[i]
        doc_text += " |text "
        text = str(text).decode('utf-8')
        text = re.sub("[0-9!@#$%^&*()\[\],\.<>;:\"{}/~`\-+=«»—\|?\^']+", '', text)
        list_of_words = re.sub(ur"(u?)\w+", ' ', text, ).split(" ")
        filtered_list_of_word = [morph.parse(w.lower())[0].normal_form
                               for w in list_of_words if w not in stopwords.words("russian")]
        filtered_text = u" ".join(filtered_list_of_word).encode('utf-8').strip()
        doc_text += filtered_text
        doc_text += "\n"
    f = open("../data/lenta.txt", "w")
    f.write(doc_text)
    f.close()


print 'BigARTM version ', artm.version(), '\n\n\n'
preprocessing_for_artm(3)
subprocess.call(['./clear.sh'])
