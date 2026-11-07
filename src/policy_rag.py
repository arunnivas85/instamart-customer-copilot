"""Chunk policy documents, retrieve relevant passages, and answer with context."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"
EVALUATION_FILE = ROOT / "data" / "raw" / "policy_eval.csv"


class PolicyRAG:
    def __init__(self, chunk_size=40, overlap=10):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > overlap >= 0.")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        step = chunk_size - overlap
        for path in sorted(POLICY_DIR.glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            for start in range(0, len(words), step):
                chunk_words = words[start : start + chunk_size]
                self.chunks.append(
                    {
                        "source": path.name,
                        "text": " ".join(chunk_words),
                    }
                )
                if start + chunk_size >= len(words):
                    break

        if not self.chunks:
            raise FileNotFoundError(f"No policy documents found in {POLICY_DIR}.")
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        if k <= 0:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in ranked_indices
        ]

    def answer(self, question):
        retrieved = self.retrieve(question, k=3)
        context = "\n".join(
            f"[{chunk['source']}]\n{chunk['text']}" for chunk in retrieved
        )
        response = mock_llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "POLICY ASSISTANT: Answer only from the supplied context. "
                        "If the answer is not in the context, say "
                        '"I could not find that information in the policy documents."'
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
                },
            ]
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": list(dict.fromkeys(chunk["source"] for chunk in retrieved)),
            "retrieved": retrieved,
        }


def evaluate(rag):
    evaluation = pd.read_csv(EVALUATION_FILE, keep_default_na=False)
    results = []
    for row in evaluation.itertuples(index=False):
        answer = rag.answer(row.question)
        answerable = row.expected_doc != "NONE"
        results.append(
            {
                "question": row.question,
                "expected_doc": row.expected_doc,
                "expected_fact": row.expected_fact,
                "retrieved_doc": (
                    row.expected_doc in answer["sources"] if answerable else False
                ),
                "fact_found": (
                    row.expected_fact in answer["answer"]
                    if answerable and row.expected_fact
                    else False
                ),
                "fallback": answer["answer"].startswith("I could not find"),
                "answer": answer["answer"],
            }
        )
    return pd.DataFrame(results)


def evaluation_summary(results):
    answerable = results["expected_doc"].ne("NONE")
    unanswerable = ~answerable
    return {
        "answerable_questions": int(answerable.sum()),
        "expected_doc_in_top_three": int(
            results.loc[answerable, "retrieved_doc"].sum()
        ),
        "expected_fact_in_answer": int(results.loc[answerable, "fact_found"].sum()),
        "unanswerable_questions": int(unanswerable.sum()),
        "fallback_answers": int(results.loc[unanswerable, "fallback"].sum()),
    }
