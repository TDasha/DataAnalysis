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
# print(corpus.getDocuments()[100].getText(), corpus.getDocuments()[100].getTag())

