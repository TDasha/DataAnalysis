from lib.data_analysis.texts.StopWords import StopWords

sw = StopWords("/home/sokolov/PycharmProjects/plsa/stopwords.dic")
sw.loadStopWordsFromFile()
print(sw.getStopWords())