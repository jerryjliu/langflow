"""Query Chain."""

from typing import Any, Callable, Union

from langflow import CustomComponent
from langflow.field_typing import BasePromptTemplate, Chain, Object


class QueryChainComponent(CustomComponent):
    display_name: str = "Query Chain"
    description: str = "Synthesizes an answer from a query engine."

    def build_config(self):
        return {
            "query_engine": {
                "display_name": "Query Engine",
                "info": "The query engine to use",
            },
            "prompt": {
                "display_name": "Prompt",
                "info": "The prompt to use",
            },
        }

    def build(
        self,
        query_engine: Object,
        prompt: BasePromptTemplate,
    ) -> Union[Chain, Callable]:
        """Build."""

        class QueryChain:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                fmt_prompt = prompt.format(**kwargs)
                return str(query_engine.query(fmt_prompt))

        query_chain = QueryChain()

        query_chain.input_keys = prompt.input_variables
        query_chain.prompt = prompt

        return query_chain
