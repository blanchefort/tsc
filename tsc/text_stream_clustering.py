from typing import List, Tuple, Dict
from datetime import datetime
import logging
import pickle
import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from tsc.dtypes import DocItem, ClusterItem
from tsc.util import get_lemmas, centroid_mean

logger = logging.getLogger('TSC')
logger.setLevel(logging.INFO)


class TextStreamClustering:
    """Кластеризация потоковых текстовых данных.
    """
    def __init__(
        self,
        threshold: float = .85,
        subcluster_threshold: float = .95,
        model_name: str = 'sberbank-ai/sbert_large_nlu_ru',
        device: str = 'cpu'
    ) -> None:
        """Инициализация кластеризатора.

        Args:
            threshold (float, optional): 
                Отсечка для основной кластеризации. 
                Для каждой нейросети должна подбираться индивидуально,
                но обычно для Трансформеров она должна быть более .75.
                Defaults to .85.
            subcluster_threshold (float, optional): 
                Отсечка для подкластреизации: 
                разбиения кластера на субкластеры. Defaults to .95.
            model_name (str, optional): Имя sentence-transformers, 
                имя модели из Huggingface, 
                либо путь до собственной предобученной модели. 
                Defaults to 'sberbank-ai/sbert_large_nlu_ru'.
            device (str, optional): тип устройства, 
                на котором проводить вычисления. ЦПУ / ГПУ. Defaults to 'cpu'.
        """
        super().__init__()
        logger.info('Инициализация')
        self.threshold = threshold
        self.subcluster_threshold = subcluster_threshold
        self.device = device
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)

        self.doc_texts: List = []
        self.doc_ids: List = []
        self.doc_dates: List = []
        self.doc_bows: List = []

        self.embeddings: List = []

        self.clusters: List = []
        self.cluster_indexes: List = []
        self.cluster_indexes_ready: List = []
        self.cluster_dates: List = []

        self.loaded_clusters_bow: Dict = {}

    def load_docs(self, docs: List[DocItem] = [], pos: Tuple = ('NOUN', 'VERB', 'INFN',)) -> None:
        """Загрузка предобработанных документов для кластеризации

        Args:
            docs (List[DocItem], optional): Документы. Defaults to [].
            pos (Tuple, optional): Части речи,
                которые следует использовать для генерации мешка слов из текстов.
                Defaults to ('NOUN', 'VERB', 'INFN',).

        """
        if len(docs) == 0:
            msg = 'Слишком мало документов для кластеризации'
            logger.error(msg)
            raise ValueError(msg)

        logger.info('Загружаем и обрабатываем документы')
        for doc in tqdm.tqdm(docs, desc='Предобработка документов'):
            if type(doc) != DocItem:
                doc = DocItem(**doc)
            if doc.idx in self.doc_ids:
                msg = 'Идентификаторы текстов должны быть уникальными.'
                logger.error(msg)
                raise KeyError(msg)
            self.doc_texts.append(doc.text)
            self.doc_ids.append(doc.idx)
            self.doc_dates.append(doc.date)
            self.doc_bows.append(get_lemmas(text=doc.text, pos=pos))

        logger.info(f'Загружено {len(self.doc_texts)} документов.')

        logger.info('Создаём эмбеддинги текстов')
        if len(self.embeddings) == 0:
            self.embeddings = self.model.encode(
                sentences=self.doc_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True)
        else:
            texts = []
            for doc in docs:
                if type(doc) != DocItem:
                    doc = DocItem(**doc)
                    texts.append(doc.text)
            ebeddings = self.model.encode(
                sentences=texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True)
            self.embeddings = torch.cat([self.embeddings, ebeddings], dim=0)

    def load_clusters(self, filepath: str, date_from: datetime) -> None:
        """Загрузка мешков слов и центроидов для ранее полученных кластеров.

        Args:
            filepath (str): Путь к файлу с сохранёнными кластерами.
            date_from (datetime): Временная отсечка. 
                Берутся только новые кластера, равные или новее этой даты.
        """
        with open(filepath, 'rb') as fp:
            clusters = pickle.load(fp)
        added_count = 0
        for cluster in clusters:
            if cluster.get_end_date() >= date_from:
                self.clusters.append(cluster.centroid)
                self.cluster_dates.append(cluster.dates)
                cid = len(self.clusters) - 1
                self.loaded_clusters_bow.update({
                    cid: cluster.doc_tokens
                })
                self.cluster_indexes.append([])
                added_count += 1
        logger.info(f'Загружено {added_count} кластеров.')

    def clusters_from_texts(self, texts):
        pass

    def save_clusters(self, filepath: str) -> None:
        """Сохранение кластеров.

        Args:
            filepath (str): Путь к файлу.
        """
        clusters = self.get_clusters_extended()
        if len(clusters) == 0:
            logger.warning('Нет кластеров. Нечего сохранять.')
        else:
            if not filepath.endswith('.clusters'):
                filepath += '.clusters'
            with open(filepath, 'wb') as fp:
                pickle.dump(clusters, fp, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'{len(clusters)} кластеров успешно сохранено: `{filepath}`')

    def clusterize(self) -> None:
        """Основная кластеризация.
        """
        logger.info('Начинаем кластеризацию')
        if len(self.clusters) == 0:
            self.cluster_indexes = util.community_detection(
                self.embeddings,
                threshold=self.threshold,
                min_community_size=2,
                init_max_size=len(self.embeddings))
            for cid in range(len(self.cluster_indexes)):
                emb = [self.embeddings[idx] for idx in self.cluster_indexes[cid]]
                emb = torch.stack(emb)
                self.clusters.append(torch.mean(emb, dim=0))
        cluster_embeddings = torch.stack(self.clusters)
        cos_scores = util.cos_sim(self.embeddings, cluster_embeddings)
        max_results = torch.max(cos_scores, dim=-1)
        for idx, (value, cid) in enumerate(zip(max_results.values, max_results.indices)):
            if value >= self.threshold:
                if idx not in self.cluster_indexes[cid]:
                    self.cluster_indexes[cid].append(idx)
                    self.clusters[cid] = centroid_mean(self.clusters[cid], self.embeddings[idx])
            else:
                self.clusters.append(self.embeddings[idx])
                self.cluster_indexes.append([idx])
        self.cluster_indexes_ready = [c for c in self.cluster_indexes if len(c) > 1]
        logger.info('Кластеризация успешно завершена')

    def get_clusters(self) -> List[List[int]]:
        """Возвращает Кластеры со списком id документов"""
        clusters = []
        for inner_cluster in self.cluster_indexes_ready:
            cluster = []
            for inner_idx in inner_cluster:
                cluster.append(self.doc_ids[inner_idx])
            clusters.append(cluster)
        return clusters

    def get_clusters_extended(self) -> List[ClusterItem]:
        """Возвращает кластеры в виде контейнеров, в которых содержится вся информация о кластере.
        """
        clusters = []
        for cid, inner_cluster in enumerate(self.cluster_indexes_ready):
            cluster = ClusterItem(centroid=self.clusters[cid])
            for inner_idx in inner_cluster:
                cluster.add_text(
                    idx=self.doc_ids[inner_idx],
                    bow=self.doc_bows[inner_idx],
                    date=self.doc_dates[inner_idx]
                )

            if cid in self.loaded_clusters_bow.keys():
                cluster.doc_tokens += self.loaded_clusters_bow[cid]
            clusters.append(cluster)
        return clusters

    def get_subcluster(self, doc_ids: List[int]) -> List[List[int]]:
        """Разбиение большого кластера на подкластеры

        Args:
            doc_ids (List[int]): Идентификаторы документов.

        Returns:
            List[int]: Список с идентификаторами документов, разбитых на подкластеры.
        """
        if len(doc_ids) < 4:
            msg = 'Слишком мало документов для кластеризации'
            logger.error(msg)
            raise ValueError(msg)
        embeddings = []
        for idx in doc_ids:
            idx = self.doc_ids.index(idx)
            embeddings.append(self.embeddings[idx])
        embeddings = torch.stack(embeddings)
        cluster_indexes = util.community_detection(
                embeddings,
                threshold=self.subcluster_threshold,
                min_community_size=2,
                init_max_size=len(embeddings))
        cluster_indexes_ = []
        for cluster in cluster_indexes:
            doc_indexes = []
            for idx in cluster:
                doc_indexes.append(doc_ids[idx])
            cluster_indexes_.append(doc_indexes)
        return cluster_indexes_
