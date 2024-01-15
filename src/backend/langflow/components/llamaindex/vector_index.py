"""Vector Index."""

from typing import List, cast
from langflow import CustomComponent
from llama_index.schema import TextNode
from llama_index import VectorStoreIndex
from langflow.field_typing import Object


class VectorIndexComponent(CustomComponent):
    display_name: str = "Vector Index"
    description: str = "Indexes text into a vector store"

    def build_config(self):
        return {
            "documents": {
                "display_name": "Documents",
                "info": "The documents to ingest",
            }
        }

    def build(
        self,
        documents: Object,
    ) -> Object:
        """Build."""
        documents = cast(List[TextNode], documents)
        return VectorStoreIndex(documents)
