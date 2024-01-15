"""Query Chain."""

from typing import Any, Callable, Union, List, Dict, Optional

from langflow import CustomComponent
from langflow.field_typing import BasePromptTemplate, Chain, Object
from llama_index.query_engine import BaseQueryEngine


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

        class QueryChain(Chain):
            """Query chain as a subclass of langchain chain.

            TODO: this is mostly to get the current runnable to work.
            
            """

            query_engine: BaseQueryEngine
            prompt: BasePromptTemplate
            
            # def __init__(
            #     self, 
            #     query_engine: BaseQueryEngine,
            #     prompt: BasePromptTemplate,
            # ) -> None:
            #     self.query_engine = query_engine
            #     self.prompt = prompt

            @property
            def input_keys(self) -> List[str]:
                return self.prompt.input_variables

            @property
            def output_keys(self) -> List[str]:
                return ["output"]

            def _call(
                self,
                inputs: Dict[str, Any],
                run_manager: Optional[Any] = None,
            ) -> Dict[str, Any]:
                """Execute the chain."""
                return {
                    "output": self.query_engine.query(self.prompt.format(**inputs))
                }

        # class QueryChain:
        #     def __call__(self, *args: Any, **kwargs: Any) -> Any:
        #         fmt_prompt = prompt.format(**kwargs)
        #         return str(query_engine.query(fmt_prompt))
        query_chain = QueryChain(query_engine=query_engine, prompt=prompt)

        # query_chain.input_keys = prompt.input_variables
        # query_chain.prompt = prompt

        return query_chain
