from .Document import Document


class Corpus:

    def __init__(self) -> None:
        self.__pathToFile = ''
        self.__documents = []

    def loadCorpusFromFile(self):
        pass

    def loadCorpusFromList(self, documents: list, tags: list = []):
        for index in range(len(documents)):
            try:
                self.__documents.append(Document(documents[index], tags[index]))
            except IndexError:
                self.__documents.append(Document(documents[index], ''))

    def getDocuments(self):
        return self.__documents

    def getDocumentByIndex(self):
        pass
