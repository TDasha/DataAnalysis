from lib.data_analysis.algorithms.texts.PLSA import PLSA
from lib.data_analysis.texts.StopWords import StopWords
from lib.data_analysis.texts.Corpus import Corpus
from os import getcwd
import pandas as pd

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.loadStopWordsFromFile()

data = pd.read_csv("/home/sokolov/PycharmProjects/t/DataAnalysis/LWs/LW_10/data/lenta_ru.csv")
documents = data["text"].tolist()
tags = data["tags"].tolist()

corpus = Corpus()
corpus.loadCorpusFromList(documents, tags)

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
plsa.EM_Algo()
