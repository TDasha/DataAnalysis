from .Document import Document


class Corpus:

    def __init__(self) -> None:
        self.__pathToFile = ''
        self.__documents = []

    def load_corpus_from_file(self) -> None:
        pass

    def load_corpus_from_list(self, documents: list, tags: list = []) -> None:
        for index in range(len(documents)):
            try:
                self.__documents.append(Document(documents[index], tags[index]))
            except IndexError:
                self.__documents.append(Document(documents[index], ''))

    def get_documents(self) -> list:
        return self.__documents

    def get_document_by_index(self, index: int) -> Document:
        try:
            return self.__documents[index]
        except IndexError:
            raise IndexError
