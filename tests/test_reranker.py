from retrieval.reranker import Reranker


def test_reranker_scores_and_sorts_passages():
    reranker = Reranker()
    candidates = [
        {"chunk_id": "c1", "text": "Throw-in is taken from the touchline where the ball went out."},
        {"chunk_id": "c2", "text": "Penalty kick is awarded if a player commits a direct free kick offence inside their penalty area."},
        {"chunk_id": "c3", "text": "A player is in an offside position if they are nearer to the opponents goal line than both the ball and the second-last opponent."},
    ]

    results = reranker.rerank("When is a penalty awarded?", candidates, top_k=2)

    assert len(results) == 2
    # C2 (nói về penalty) phải đứng đầu
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["score"] > results[1]["score"]
