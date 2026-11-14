from typing import Iterable, List
from fastembed import TextEmbedding


class LocalBatchEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        embeddings = self.model.passage_embed(text_list, batch_size=self.batch_size)
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        vec = next(self.model.query_embed([query]))
        return vec.tolist()
