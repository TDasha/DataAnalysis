from lib.data_analysis.algorithms.texts.plsa import PLSA
from lib.data_analysis.texts.stop_words import StopWords
from lib.data_analysis.texts.corpus import Corpus
from os import getcwd
import pandas as pd

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.load_stop_words_from_file()

data = pd.read_csv("/home/sokolov/PycharmProjects/t/DataAnalysis/LWs/LW_10/data/lenta_ru.csv")
documents = data["text"].tolist()
tags = data["tags"].tolist()

corpus = Corpus()
corpus.load_corpus_from_list(documents, tags)

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
plsa.em_algorithm()
