import evaluate
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_answer(answer: str, reference: str):
    bertscore = evaluate.load("bertscore")
    bert_result = bertscore.compute(predictions=[answer], references=[reference], lang="uk")
    bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
    return bert_f1

def compute_context_relevance(query, docs):
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    docs_emb = embedding_model.encode(docs, convert_to_tensor=True)
    similarity = util.cos_sim(query_emb, docs_emb)
    return float(similarity.mean())

def compute_groundedness(answer, docs):
    answer_emb = embedding_model.encode(answer, convert_to_tensor=True)
    docs_emb = embedding_model.encode(docs, convert_to_tensor=True)
    similarity = util.cos_sim(answer_emb, docs_emb)
    return float(similarity.max())