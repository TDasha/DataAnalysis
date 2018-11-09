from .Preprocessing import Preprocessing


class Document:

    def __init__(self, text: str, tag: str = '') -> None:
        self._text = text.strip()
        self._text_as_list = []
        self._tag = tag.strip()

    def get_text(self) -> str:
        return self._text

    def get_tag(self) -> str:
        return self._tag

    def set_text(self, text: str) -> None:
        self._text = text.strip()
        self._text_as_list = []

    def set_tag(self, tag: str = '') -> None:
        self._tag = tag.strip()

    def convert_text_to_list_of_words(self) -> None:
        self._text_as_list = Preprocessing.convert_text_to_list_of_words(self._text)

    def get_text_as_list_of_words(self) -> list:
        return self._text_as_list
