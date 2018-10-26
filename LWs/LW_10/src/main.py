from lib.data_analysis.texts.Preprocessing import Preprocessing
from lib.data_analysis.texts.StopWords import StopWords
from lib.data_analysis.texts.Corpus import Corpus
import pandas as pd

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.loadStopWordsFromFile()
# print(sw.getStopWords())

data = pd.read_csv(r"/home/sokolov/lr2/myPackage/data/lenta_ru.csv")
documents = data["text"].tolist()
tags = data["tags"].tolist()
# print(documents, tags)

corpus = Corpus()
corpus.loadCorpusFromList(documents, tags)
# print(corpus.getDocuments()[99].getText(), corpus.getDocuments()[99].getTag())

# try:
#     print(corpus.getDocumentByIndex(10))
#     print(corpus.getDocumentByIndex(150))
# except IndexError:
#     print("Index Error")

# doc_50 = corpus.getDocumentByIndex(50)
# doc_50.convertTextToListOfWords()
# print(doc_50.getTextAsListOfWords())

# doc_60 = corpus.getDocumentByIndex(60)
# print(Preprocessing.convertDocumentToListOfWords(doc_60))

