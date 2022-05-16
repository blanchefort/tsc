from typing import List
from datetime import datetime
from collections import Counter
from pydantic import BaseModel, Field
import torch


class ClusterItem(BaseModel):
    """Модель кластера.

    text_ids: List[int] - идентификаторы текстов, входящих к кластер.
    doc_tokens: List[str] - мешок слов кластера.
    dates: List[str] - даты всех текстов кластера.
    centroid: - центроид кластера.
    """
    text_ids: List[int] = []
    doc_tokens: List[str] = []
    dates: List[str] = []
    centroid: torch.Tensor = torch.zeros(1, 1)

    class Config:
        arbitrary_types_allowed = True

    def add_text(self, idx: int, bow: List[str], date: datetime) -> None:
        """Добавление нового документа в кластер.

        Args:
            idx (int): Идентификатор документа.
            bow (List[str]): Мешок слов документа.
            date (datetime): Дата создания документа.
        """
        self.text_ids.append(idx)
        self.doc_tokens += bow
        self.dates.append(date)

    def get_name_by_doc_tokens(self, top_k: int = 5) -> List[str]:
        """Возвращает топ-N токенов кластера для формирования условного заголовка кластера.

        Args:
            top_k (int, optional): [description]. Defaults to 5.

        Returns:
            List[str]: Список топ-токенов.
        """
        return [w[0] for w in Counter(self.doc_tokens).most_common(top_k)]

    def get_start_date(self) -> datetime:
        """Возвращает дату создания кластера (по самому старому документу).
        """
        return min(self.dates)

    def get_end_date(self) -> datetime:
        """Возвращает дату самого последнего документа.
        """
        return max(self.dates)

    def set_centroid(self, centroid: torch.Tensor) -> None:
        """Задать центроид кластера.

        Args:
            centroid (torch.Tensor): Тензор центроида.
        """
        self.centroid = centroid

    def get_centroid(self) -> torch.Tensor:
        """Возвращает центроид кластера.

        Returns:
            torch.Tensor: Тензор центроида.
        """
        return self.centroid


class DocItem(BaseModel):
    """Модель входящего документа.

    idx: int - уникальный идентиифкатор документа.
    text: str - текст документа.
    date: - дата создания документа.

    """
    idx: int
    text: str
    date: datetime = Field(default_factory=datetime.utcnow)
