"""RAGAS Evaluation — faithfulness, answer_relevancy, context_recall"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

def run_evaluation(pipeline, qa_pairs):
    rows = []
    for item in qa_pairs:
        q = item["question"]
        contexts = [d.page_content for d in pipeline.vectorstore.similarity_search(q, k=5)]
        answer = pipeline.query(q)
        rows.append({"question": q, "answer": answer, "contexts": contexts, "ground_truth": item["ground_truth"]})
    dataset = Dataset.from_list(rows)
    return evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

if __name__ == "__main__":
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline("./data")
    pipeline.ingest()
    qa = [{"question": "What is RAG?", "ground_truth": "Retrieval Augmented Generation."}]
    print(run_evaluation(pipeline, qa))
