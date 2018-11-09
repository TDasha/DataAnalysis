import codecs

import nltk


class StopWords:

    def __init__(self, pathToFileWithStopWords: str) -> None:
        self._stopWords = []
        self.__pathToFileWithStopWords = pathToFileWithStopWords

    def get_stop_words(self) -> list:
        return self._stopWords

    def load_stop_words_from_file(self) -> None:
        file = codecs.open(self.__pathToFileWithStopWords, 'r', 'utf-8')
        stopWords = [line.strip() for line in file]
        file.close()
        self._stopWords = stopWords

    def load_stop_word_from_nltk_lib(self) -> None:
        stopwords_list = nltk.corpus.stopwords.words('russian')
        self._stopWords.append(list(set(stopwords_list)))

    def is_stop_word(self, stopWord: str):
        return stopWord in self._stopWords
