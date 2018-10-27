from lib.data_analysis.algorithms.texts.PLSA import PLSA
from lib.data_analysis.texts.Preprocessing import Preprocessing
from lib.data_analysis.texts.StopWords import StopWords
from lib.data_analysis.texts.Corpus import Corpus
from os import getcwd
from os import mknod
import pandas as pd

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.loadStopWordsFromFile()
# print(sw.getStopWords())

data = pd.read_csv("/home/sokolov/PycharmProjects/t/DataAnalysis/LWs/LW_10/data/lenta_ru.csv")
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

# doc_10 = corpus.getDocumentByIndex(10)
# print(doc_10.getText(), end="\n\n")
# normalized_list = Preprocessing.convertListOfWordsToNormalForms(Preprocessing.convertDocumentToListOfWords(doc_10))
# print(normalized_list, end="\n\n")
# segList = Preprocessing.removeStopWordsFromListOfWords(sw.getStopWords(), normalized_list)
# print(segList, end="\n\n")

K = 10    # number of topic
maxIteration = 30
threshold = 10.0
topicWordsNum = 10
docTopicDist = 'docTopicDistribution.txt'
docTopicDist = getcwd() + "/../results/" + docTopicDist
topicWordDist = 'topicWordDistribution.txt'
topicWordDist = getcwd() + "/../results/" + topicWordDist
dictionary = 'dictionary.dic'
dictionary = getcwd() + "/../results/" + dictionary
topicWords = 'topics.txt'
topicWords = getcwd() + "/../results/" + topicWords

plsa = PLSA(corpus, sw, K, maxIteration, threshold, topicWordsNum, docTopicDist, topicWordDist, dictionary, topicWords)