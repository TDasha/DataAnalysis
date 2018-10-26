from lib.data_analysis.texts.StopWords import StopWords

sw = StopWords("/home/sokolov/PycharmProjects/t/DataAnalysis/lib/data_analysis/texts/stopwords.dic")
sw.loadStopWordsFromFile()
print(sw.getStopWords())