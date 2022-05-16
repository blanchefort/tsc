import re
from typing import Tuple, List
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import torch


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


def get_lemmas(text: str = '', pos: Tuple = ('NOUN', 'VERB', 'INFN',)) -> List[str]:
    """Преобразуем текст в леммы-униграммы

    Args:
        text ([type]): Входящий русскоязычный текст
        pos (Tuple, optional): Части речи, которые нужно использовать. 
            Defaults to ('NOUN', 'VERB', 'INFN',).

    Returns:
        List[str]: Список лемм текста.
    """
    if len(text) == 0:
        return []
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return [re.sub('[^A-Za-zА-Яа-я]+', ' ', t.lemma) for t in doc.tokens if t.pos in pos]


def centroid_mean(cluster_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """Получение арифметического среднего между двумя тензорами.

    Args:
        cluster_emb (torch.Tensor): Тензор центроида кластера.
        doc_emb (torch.Tensor): Тензор документа.

    Returns:
        torch.Tensor: Усьтреднённый тензор.
    """
    return (cluster_emb + doc_emb) / 2
