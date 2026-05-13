import re
from typing import Any, Dict, List

import requests


class RetrievalAgent:
    def retrieve(self, query: str, topk: int = 5) -> List[str]:
        raise NotImplementedError("Please implement your retrieval tool here.")


class HttpRetrievalAgent(RetrievalAgent):
    def __init__(self, host: str = "127.0.0.1", port: int = 8008, timeout: int = 10):
        self.endpoint = f"http://{host}:{port}/retrieve"
        self.timeout = timeout

    @staticmethod
    def _parse_passages(retrieval_result: List[Dict[str, Any]]) -> List[str]:
        docs = []
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            lines = content.split("\n")
            title = lines[0]
            body = "\n".join(lines[1:]).strip()
            docs.append(f"Doc {idx+1} (Title: {title})\n{body}")
        return docs

    def retrieve(self, query: str, topk: int = 5) -> List[str]:
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True,
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        results = response.json()["result"]
        return self._parse_passages(results[0])


class MockRetrievalAgent(RetrievalAgent):
    def __init__(self):
        self.corpus = [
            "Inception is a 2010 science fiction film directed by Christopher Nolan.",
            "Christopher Nolan was born on 30 July 1970 in London.",
            "Jurassic Park is a 1993 science fiction adventure film directed by Steven Spielberg.",
            "Steven Spielberg was born on 18 December 1946 in Cincinnati, Ohio.",
            "The Dark Knight was directed by Christopher Nolan.",
            "E.T. the Extra-Terrestrial was directed by Steven Spielberg.",
        ]

    def retrieve(self, query: str, topk: int = 5) -> List[str]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for doc in self.corpus:
            doc_terms = set(re.findall(r"\w+", doc.lower()))
            score = len(query_terms & doc_terms)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:topk] if score > 0]
