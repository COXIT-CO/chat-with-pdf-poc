from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Iterable, List, Iterator

import numpy as np
from pymilvus import MilvusClient, DataType

from core.logger import info


@dataclass
class VectorItem:
    par_id: str
    vector: np.array
    text: str
    file_name: str
    file_name_orig: Optional[str] = None
    idx: Optional[int] = None
    page_n: Optional[int] = None
    paragraph_box: Optional[Iterable[float]] = None


@dataclass
class SearchResult:
    id: int
    distance: float
    par_id: str
    text: str
    file_name: str
    file_name_orig: Optional[str] = None
    idx: Optional[int] = None
    page_n: Optional[int] = None
    paragraph_box: Optional[Iterable[float]] = None



def collection_from_file_name(file_name: str) -> str:
    return "_" + Path(file_name).stem


class MilvusRepository:
    def __init__(
            self,
            file_path: Path
    ):
        self._client = MilvusClient(str(file_path))

    @property
    def client(self) -> MilvusClient:
        return self._client

    def prepare_collection(self, collection_name: str, dimension: int) -> None:
        """
        Creates a new Milvus collection with the specified name and vector dimension.

        This method sets up a collection with a schema that includes fields for:
        - id: Auto-generated primary key
        - par_id: String identifier for paragraphs
        - vector: Float vector field with the specified dimension
        - text: Text content of the vector item
        - file_name: Source file name
        - file_name_orig: Original file name (optional)
        - idx: Index value (optional)
        - page_n: Page number (optional)
        - paragraph_box: Array of 4 float values representing paragraph coordinates (optional)

        After creating the collection, it also creates a HNSW_SQ index on the vector field
        using COSINE similarity for efficient similarity search.

        Args:
            collection_name (str): Name of the collection to create
            dimension (int): Dimension of the vector field

        Returns:
            None
        """
        schema = MilvusClient.create_schema()

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="par_id", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4196)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="file_name_orig", datatype=DataType.VARCHAR, nullable=True, max_length=128)
        schema.add_field(field_name="idx", datatype=DataType.INT32, nullable=True)
        schema.add_field(field_name="page_n", datatype=DataType.INT32, nullable=True)
        schema.add_field(
            field_name="paragraph_box", datatype=DataType.ARRAY, element_type=DataType.FLOAT,
            max_capacity=4, array_capacity=4,
            nullable=True
        )

        self._client.create_collection(
            collection_name, schema=schema
        )
        info(f"created collection {collection_name}")

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="FLAT",
            index_name="vector_index",
        )

        self._client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            sync=True,
        )
        info(f"created index 'vector_index' for collection: {collection_name}")

    def list_collections(self):
        return self._client.list_collections()

    def collection_info(self, collection_name: str):
        return self._client.describe_collection(collection_name)

    def drop_collection(self, collection_name: str):
        return self._client.drop_collection(collection_name=collection_name)

    def insert(self, collection_name: str, data: Iterable[VectorItem]):
        data = [asdict(d) for d in data]
        res = self._client.insert(
            collection_name=collection_name,
            data=data
        )
        info(f"inserted {len(data)} vectors")
        return res


    def search(
            self,
            collection_name: str,
            vector: np.array,
            limit: int = 10,
            filter: str = ""
    ) -> Iterator[SearchResult]:
        results = self._client.search(
            collection_name=collection_name,
            anns_field="vector",
            data=[vector],
            filter=filter,
            limit=limit,
            output_fields=["par_id", "text", "file_name", "file_name_orig", "idx", "page_n", "paragraph_box"]
        )
        for r in results[0]:
            yield SearchResult(
                id=r["id"],
                distance=r["distance"],
                par_id=r["entity"]["par_id"],
                text=r["entity"]["text"],
                file_name=r["entity"]["file_name"],
                file_name_orig=r["entity"].get("file_name_orig"),
                idx=r["entity"].get("idx"),
                page_n=r["entity"].get("page_n"),
                paragraph_box=r["entity"].get("paragraph_box")
            )

    def get_all_vector_par_ids(self, collection_name: str) -> List[str]:
        # todo use cursor for pagination for large collections
        """
        Retrieves all vector IDs from the specified collection.

        This method queries the collection and returns a list of all primary key IDs.

        Args:
            collection_name (str): Name of the collection to query

        Returns:
            List[int]: List of all vector IDs in the collection
        """
        # First ensure the collection is loaded for search
        self._client.load_collection(collection_name)
        stats = self._client.get_collection_stats(collection_name)
        row_count = int(stats["row_count"])

        # Query all IDs from the collection
        result = self._client.query(
            collection_name=collection_name,
            filter="",  # Empty filter to get all records
            output_fields=["par_id"],
            limit=row_count
        )

        # Extract IDs from the result
        ids = [item["par_id"] for item in result]

        info(f"Retrieved {len(ids)} vector IDs from collection: {collection_name}")
        return ids
