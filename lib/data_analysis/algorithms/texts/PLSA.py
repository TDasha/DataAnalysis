import re
from numpy import int8
from numpy import zeros
from pylab import random
from ...texts.StopWords import StopWords
from ...texts.Corpus import Corpus
from ...texts.Preprocessing import Preprocessing

class PLSA:

    def __init__(self, corpus: Corpus, stopWords: StopWords, K: int, maxIteration: int, threshold: float,
                 topicWordsNum: int, docTopicDist: str, topicWordsDist: str, dictionary: str, topicWords: str) -> None:
        self._corpus = corpus
        self._stopWords = stopWords
        self._K = K  # number of topic
        self._maxIteration = maxIteration
        self._threshold = threshold
        self._topicWordsNum = topicWordsNum
        self._docTopicDist = docTopicDist
        self._topicWordDist = topicWordsDist
        self._dictionary = dictionary
        self._topicWords = topicWords

        self._N = len(corpus.getDocuments())
        self._wordCounts = []
        self._word2id = {}
        self._id2word = {}
        self._currentId = 0

        for document in corpus.getDocuments():
            normalized_list = Preprocessing.convertListOfWordsToNormalForms(Preprocessing
                                                                            .convertDocumentToListOfWords(document))
            self._segList = Preprocessing.removeStopWordsFromListOfWords(self._stopWords.getStopWords(),
                                                                         normalized_list)
            self._wordCount = {}
            for word in self._segList:
                word = word.lower().strip()
                if len(word) > 1 and not re.search('[0-9]', word) and word not in self._stopWords.getStopWords():
                    if word not in self._word2id.keys():
                        self._word2id[word] = self._currentId
                        self._id2word[self._currentId] = word
                        self._currentId += 1
                    if word in self._wordCount:
                        self._wordCount[word] += 1
                    else:
                        self._wordCount[word] = 1
            self._wordCounts.append(self._wordCount)

        # length of dictionary
        self._M = len(self._word2id)

        # generate the document-word matrix
        self._X = zeros([self._N, self._M], int8)
        for word in self._word2id.keys():
            j = self._word2id[word]
            for i in range(0, self._N):
                if word in self._wordCounts[i]:
                    self._X[i, j] = self._wordCounts[i][word]

        # lamda[i, j] : p(zj|di)
        self._lamda = random([self._N, self._K])

        # theta[i, j] : p(wj|zi)
        self._theta = random([self._K, self._M])

        # p[i, j, k] : p(zk|di,wj)
        self._p = zeros([self._N, self._M, self._K])

    def initializeParameters(self):
        for i in range(0, self._N):
            self._normalization = sum(self._lamda[i, :])
            for j in range(0, self._K):
                self._lamda[i, j] /= self._normalization;

        for i in range(0, self._K):
            self._normalization = sum(self._theta[i, :])
            for j in range(0, self._M):
                self._theta[i, j] /= self._normalization;