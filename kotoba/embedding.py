from abc import ABCMeta, abstractmethod
from ._utils import uniquify, flatten, create_directory
from enum import Enum


class SpecialTokens(Enum):
    PAD = '<PAD>'
    UNKNOWN = '<UNK>'


class TokenEmbedding(metaclass=ABCMeta):

    @abstractmethod
    def token_to_id(self, token):
        pass

    @abstractmethod
    def id_to_token(self, id_):
        pass

    @abstractmethod
    def token_coverage(self, tokens):
        pass

    @abstractmethod
    def id_coverage(self, ids):
        pass

    @abstractmethod
    def export_token_list(self, path):
        pass


class BasicEmbedding(TokenEmbedding):
    def __init__(self, token_list, add_pad=False):
        index_to_token = self._create_index_token_list(token_list, add_pad)
        self._index_to_token = tuple(index_to_token)
        self._token_to_index = self._create_token_index_dict(index_to_token)
        self._token_set = frozenset(index_to_token)

    # noinspection PyMethodMayBeStatic
    def _check_special_chars(self, x):
        try:
            SpecialTokens(x)
        except ValueError:
            return True
        raise ValueError('{} is a reserved keyword'.format(x))

    # noinspection PyMethodMayBeStatic
    def _create_index_token_list(self, token_list, add_pad):
        token_list = uniquify(flatten(token_list))
        token_list = filter(self._check_special_chars, token_list)
        index_to_token = []
        if add_pad:
            index_to_token.append(SpecialTokens.PAD.value)
        index_to_token.extend(token_list)
        return index_to_token

    # noinspection PyMethodMayBeStatic
    def _create_token_index_dict(self, index_token_list):
        count = len(index_token_list)
        return dict(zip(index_token_list, range(count)))

    def token_to_id(self, token):
        default = len(self._token_to_index) + 1
        return self._token_to_index.get(token, default)

    def id_to_token(self, id_):
        if id_ < 0:
            return SpecialTokens.UNKNOWN.value
        else:
            try:
                return self._index_to_token[id_]
            except IndexError:
                return SpecialTokens.UNKNOWN.value

    # noinspection PyMethodMayBeStatic
    def _get_coverage(self, base_set, target_list):
        target_set = set(flatten(target_list))
        common_entries = base_set & target_set
        return len(common_entries) / len(target_set)

    def token_coverage(self, tokens):
        return self._get_coverage(self._token_set, tokens)

    def id_coverage(self, ids):
        ids_set = set(range(len(self._token_set)))
        return self._get_coverage(ids_set, ids)

    def export_token_list(self, path, encoding='utf-8'):
        create_directory(path)
        with open(path, 'w', encoding=encoding) as file:
            file.writelines((l + '\n' for l in self._index_to_token))

    @property
    def token_size(self):
        return len(self._index_to_token)

    def save(self, path, encoding='utf-8'):
        return self.export_token_list(path, encoding)

    @classmethod
    def load(cls, path, encoding='utf-8', add_pad=False):
        with open(path, 'r', encoding=encoding) as file:
            return cls([l.strip() for l in file], add_pad)
