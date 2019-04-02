import string
import random
import numpy as np
from typing import Iterable, Dict, Tuple, List, Union


class TransformCompose:
    """
        Transform text and annotation for name entity recognition task.

        Parameters
        ----------
        transforms : list
            List of transformations.

    """

    def __init__(self, transforms: Iterable[BasicTransform]):
        self.transforms = transforms

    def apply(self, text: str, annotation: Dict[str, List[Tuple[int]]]) -> Union[str, Dict[str, List[Tuple[int]]]]:
        """
        Perform transformation text, annotation.

        ----------
        text : str
            Input text.
        annotation : dict
            Annoation begin, end list for each entity.

        Returns
        -------
            Transformed text, annotation.
        """

        for t in self.transforms:
            text, annotation = t(text, annotation)

        return text, annotation


class BasicTransform:
    """
    Base class for text, annotation transformation.
    """

    def apply(self, text: str, annotation: Dict[str, List[Tuple[int]]]):
        raise NotImplementedError

    def __update_text_annotation(self):
        raise NotImplementedError


class AnnotationMisprint(BasicTransform):
    """
    Add random misprint to Annotation without length changing.

    Parameters
    ----------
    n_letters: int, default: 1
        Number of letter for changing in each word.

    repeat_patience: int, default: 10
        Patience for looking for correct replace letter.

    same_case: bool, default: False
        Ue the same case for replaced letter.

    p: float [0, 1], default: 0.5
        Probability for word choosing.

    """

    def __init__(self, n_letters: int=1, repeat_patience: int=10, same_case: bool=False, p: float=0.5):
        super(AnnotationMisprint, self).__init__()

        self.n_letters = n_letters
        self.repeat_patience = repeat_patience
        self.same_case = same_case
        self.p = p
        # Russian letters
        self.cyrillic_lowercase = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.cyrillic_uppercase = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

    def apply(self, text: str, annotation: Dict[str, List[Tuple[int]]]) -> Union[str, Dict[str, List[Tuple[int]]]]:
        """
        Perform transformation text, annotation.

        ----------
        text : str
            Input text.
        annotation : dict
            Annoation begin, end list for each entity.

        Returns
        -------
            Transformed text, annotation.
        """

        text_list = [ch for ch in text]
        for entity, pos_lt in annotation.items():
            for pos in pos_lt:
                if random.random() > self.p:
                    word = text[pos[0]: pos[1]]
                    new_word = self.__change_letters(word, self.n_letters)
                    # Change word in text
                    for i_ch, ch in enumerate(new_word):
                        text_list[pos[0] + i_ch] = ch

        new_text = ''.join(text_list)
        return new_text, annotation

    def __change_letters(self, word: str, n_letters: int) -> str:
        """
        Replace letters in the word

        Parameters
        ----------
        word: str
            Input word.

        n_letters: int
            Number of letters for changing.

        Returns
        -------
            New word.

        """
        
        repeat_counter = self.repeat_patience
        possible_position = [i for i, _ in enumerate(word)]
        word_list = [ch for ch in word]
        while n_letters > 0 and repeat_counter > 0 and len(possible_position) > 0:
            pos = random.choice(possible_position)
            possible_position.remove(pos)
            ch = word_list[pos]
            new_ch = self.__get_new_ch(ch)

            if new_ch != ch:
                word_list[pos] = new_ch
                n_letters -= 1

            repeat_counter -= 1

        return ''.join(word_list)

    def __get_new_ch(self, ch: str) -> str:
        """
        Selects a letter depending on the input letter.

        Parameters
        ----------
        ch: str
            input letter.

        Returns
        -------
            Letter for replace.

        """

        short_list = None
        # Digit
        if ch in string.digits:
            short_list = [s for s in string.digits if s != ch]
        # Latin letters
        elif ch in string.ascii_lowercase and self.same_case:
            short_list = [s for s in string.ascii_lowercase if s != ch]
        elif ch in string.ascii_uppercase and self.same_case:
            short_list = [s for s in string.ascii_uppercase if s != ch]
        elif ch in string.ascii_letters:
            short_list = [s for s in string.ascii_letters if s != ch]
        # Cyrillic letters
        elif ch in self.cyrillic_lowercase and self.same_case:
            short_list = [s for s in self.cyrillic_lowercase if s != ch]
        elif ch in self.cyrillic_uppercase and self.same_case:
            short_list = [s for s in self.cyrillic_uppercase if s != ch]
        elif ch in self.cyrillic_lowercase + self.cyrillic_uppercase:
            short_list = [s for s in self.cyrillic_lowercase + self.cyrillic_uppercase if s != ch]

        if short_list is not None:
            new_ch = random.choice(short_list)
        else:
            new_ch = ch

        return new_ch


class TextMisprint(BasicTransform):
    """
    Add random misprint to Text without length changing.

    Parameters
    ----------
    n_letters: int, default: 10
       Number of letter for changing in whole text.

    repeat_patience: int, default: 10
       Patience for looking for correct replace letter.

    same_case: bool, default: False
       Ue the same case for replaced letter.

    p: float [0, 1], default: 0.5
       Probability for word choosing.

    """

    def __init__(self, n_letters: int=10, repeat_patience: int=10, same_case: bool=False, p: float=0.5):
        super(TextMisprint, self).__init__()

        self.n_letters = n_letters
        self.repeat_patience = repeat_patience
        self.same_case = same_case
        self.p = p
        self.cyrillic_lowercase = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.cyrillic_uppercase = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

    def apply(self, text: str, annotation: Dict[str, List[Tuple[int]]]) -> Union[str, Dict[str, List[Tuple[int]]]]:
        """
        Perform transformation text, annotation.

        ----------
        text : str
           Input text.
        annotation : dict
           Annoation begin, end list for each entity.

        Returns
        -------
           Transformed text, annotation.
        """

        text_list = [ch for ch in text]
        possible_position = [i for i, _ in enumerate(text)]
        letter_counter = self.n_letters
        repeat_counter = self.repeat_patience
        while letter_counter > 0 and repeat_counter > 0 and len(possible_position) > 0:
            pos = random.choice(possible_position)
            possible_position.remove(pos)
            ch = text_list[pos]
            new_ch = self.__get_new_ch(ch)

            if new_ch != ch:
                text_list[pos] = new_ch
                letter_counter -= 1
                repeat_counter = self.repeat_patience

            repeat_counter -= 1

        new_text = ''.join(text_list)
        return new_text, annotation

    def __get_new_ch(self, ch: str) -> str:
        """
        Selects a letter depending on the input letter.

        Parameters
        ----------
        ch: str
            input letter.

        Returns
        -------
            Letter for replace.

        """

        short_list = None
        # Digit
        if ch in string.digits:
            short_list = [s for s in string.digits if s != ch]
        # Latin letters
        elif ch in string.ascii_lowercase and self.same_case:
            short_list = [s for s in string.ascii_lowercase if s != ch]
        elif ch in string.ascii_uppercase and self.same_case:
            short_list = [s for s in string.ascii_uppercase if s != ch]
        elif ch in string.ascii_letters:
            short_list = [s for s in string.ascii_letters if s != ch]
        # Cyrillic letters
        elif ch in self.cyrillic_lowercase and self.same_case:
            short_list = [s for s in self.cyrillic_lowercase if s != ch]
        elif ch in self.cyrillic_uppercase and self.same_case:
            short_list = [s for s in self.cyrillic_uppercase if s != ch]
        elif ch in self.cyrillic_lowercase + self.cyrillic_uppercase:
            short_list = [s for s in self.cyrillic_lowercase + self.cyrillic_uppercase if s != ch]

        if short_list is not None:
            new_ch = random.choice(short_list)
        else:
            new_ch = ch

        return new_ch
