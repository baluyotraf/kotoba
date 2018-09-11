from abc import ABCMeta, abstractmethod
from ._utils import uniquify, flatten, create_directory
from .preprocess import FunctionPreprocessor


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


class Embedding(TokenEmbedding):
    def __init__(self, token_list, special_tokens=None, unk_idx=None):
        special_tokens = special_tokens or []
        self._unk = (None if unk_idx is None
                     else special_tokens[unk_idx])
        index_to_token = self._create_index_token_list(token_list, special_tokens)
        self._index_to_token = tuple(index_to_token)
        self._token_to_index = self._create_token_index_dict(index_to_token)
        self._token_set = frozenset(index_to_token)
        self._unk_id = (len(self._index_to_token) + 1
                        if self._unk is None
                        else unk_idx)

    # noinspection PyMethodMayBeStatic
    def _create_index_token_list(self, token_list, special_tokens):
        token_list = uniquify(flatten(token_list))
        index_to_token = special_tokens.copy()
        index_to_token.extend(token_list)
        return index_to_token

    # noinspection PyMethodMayBeStatic
    def _create_token_index_dict(self, index_token_list):
        count = len(index_token_list)
        return dict(zip(index_token_list, range(count)))

    def token_to_id(self, token):
        return self._token_to_index.get(token, self._unk_id)

    def id_to_token(self, id_):
        if id_ < 0:
            return self._unk
        else:
            try:
                return self._index_to_token[id_]
            except IndexError:
                return self._unk

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

    @classmethod
    def from_export_file(cls, path, special_tokens=None, unk_idx=None, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as file:
            return cls((l.strip() for l in file), special_tokens, unk_idx)

    @classmethod
    def from_glove_file(cls, path, special_tokens=None, unk_idx=None, encoding='utf-8'):
        with open(path, 'r', encoding=encoding) as file:
            parsed_file = (l.strip().split(' ')[0] for l in file)
            return cls(parsed_file, special_tokens, unk_idx)


class EmbedTokenToID(FunctionPreprocessor):

    def __init__(self, token_embedding):
        super().__init__(token_embedding.token_to_id)


class EmbedIDToToken(FunctionPreprocessor):

    def __init__(self, token_embedding):
        super().__init__(token_embedding.id_to_token)
