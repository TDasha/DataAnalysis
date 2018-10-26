import re
from lib.data_analysis.texts import Document


class Preprocessing:

    @staticmethod
    def convertTextToListOfWords(text: str) -> list:
        return re.sub("[^\w]", " ", text).split()

    @staticmethod
    def convertDocumentToListOfWords(document: Document):
        return re.sub("[^\w]", " ",  document.getText()).split()

    # @staticmethod
    # def removeStopWords(stop_words: list) -> list:
        # return stop_words.r