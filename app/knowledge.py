from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    title: str
    content: str
    source: str


DOCUMENTS = [
    KnowledgeDocument(
        "kb-001",
        "注射类项目风险提示",
        "注射类项目应由专业医师面诊评估。孕期、哺乳期或正在使用抗凝药物的用户，"
        "应主动告知医师，不应仅依据在线问答决定是否接受项目。",
        "示例知识库/注射类项目说明",
    ),
    KnowledgeDocument(
        "kb-002",
        "透明质酸项目术后护理",
        "透明质酸注射后应按医嘱护理，短期内避免按压治疗区域、剧烈运动和高温环境。"
        "若出现持续疼痛、明显颜色变化或其他异常，应及时联系医疗机构。",
        "示例知识库/术后护理说明",
    ),
    KnowledgeDocument(
        "kb-003",
        "激光类项目前信息确认",
        "激光类项目前应确认皮肤状态、近期暴晒情况、既往治疗经历及是否正在使用光敏性药物。"
        "存在不确定风险时，应转交医师进一步评估。",
        "示例知识库/激光类项目说明",
    ),
    KnowledgeDocument(
        "kb-004",
        "智能顾问服务边界",
        "智能顾问仅用于一般知识科普和风险提示，不能替代执业医师面诊、诊断或治疗方案。"
        "涉及个人病史、用药、过敏或异常症状时，应进入人工审核或线下面诊流程。",
        "示例知识库/服务边界",
    ),
]


def _tokens(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    words.extend(chinese)
    words.extend(chinese[index : index + 2] for index in range(len(chinese) - 1))
    return words


class LocalHybridRetriever:
    """BM25-like score + character-bigram overlap.

    This keeps the project runnable without a model download. The class is an
    adapter boundary that can later be replaced by BGE-M3 + Milvus.
    """

    def __init__(self) -> None:
        self.documents = DOCUMENTS
        self.tokenized = [_tokens(f"{doc.title} {doc.content}") for doc in DOCUMENTS]
        self.avg_length = sum(map(len, self.tokenized)) / len(self.tokenized)
        self.document_frequency = Counter()
        for tokens in self.tokenized:
            self.document_frequency.update(set(tokens))

    def _bm25(self, query: list[str], document: list[str]) -> float:
        frequencies = Counter(document)
        score = 0.0
        for token in set(query):
            df = self.document_frequency[token]
            if not df:
                continue
            tf = frequencies[token]
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            denominator = tf + 1.5 * (0.25 + 0.75 * len(document) / self.avg_length)
            score += idf * tf * 2.5 / denominator if denominator else 0.0
        return score

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_tokens = _tokens(query)
        query_set = set(query_tokens)
        results = []
        for document, tokens in zip(self.documents, self.tokenized, strict=True):
            union = query_set | set(tokens)
            overlap = len(query_set & set(tokens)) / len(union) if union else 0.0
            score = self._bm25(query_tokens, tokens) + overlap * 3
            results.append((score, document))
        results.sort(key=lambda item: item[0], reverse=True)
        return [
            {**asdict(document), "score": round(score, 4)}
            for score, document in results[:top_k]
            if score > 0
        ]


retriever = LocalHybridRetriever()
