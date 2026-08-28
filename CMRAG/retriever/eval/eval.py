import numpy as np
from typing import List, Dict


def ensure_list(x):
    """ Ensures the input is a list. If it's not a list, it wraps it in a list. """
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        x = x.replace("[", "").replace("]", "").replace(" ", "").split(",")
        if len(x) == 1 and x[0] == "":
            x = []
        else:
            x = [int(x) for x in x]
        return x
    else:
        return [x]



def eval_retrieval(similarities: List[List[float]],
                   doc_page_ids: List[List[int]],
                   gt_page_ids: List[List[int]],
                   topk: int = 100,
                   ks: List[int] = None) -> Dict[str, float]:
    """
    similarities : outer list = queries, inner list = scores for every candidate page
    doc_page_ids : parallel nested list with the real page-id that belongs to each score
    gt_page_ids  : ground-truth page ids for every query (list of ints)
    topk         : cut-off used to build the ranking (must be ≥ max(ks))
    ks           : which cut-offs to report (default [1,3,5,10])
    returns      : dict with keys  HIT@k  NDCG@k  MRR@k
    """
    if ks is None:
        ks = [1, 3, 5, 10]
    if max(ks) > topk:
        raise ValueError("ks cannot be larger than topk")

    hits = {f"HIT@{k}": 0.0 for k in ks}
    ndcgs = {f"NDCG@{k}": 0.0 for k in ks}
    mrrs = {f"MRR@{k}": 0.0 for k in ks}
    recall = {f"Recall@{k}": 0.0 for k in ks}  # multiple relevant documents
    n_queries = len(similarities)

    for scores, page_ids, gt_ids in zip(similarities, doc_page_ids, gt_page_ids):
        scores = np.asarray(scores, dtype=np.float32)
        page_ids = np.asarray(page_ids, dtype=np.int32)
        gt_ids = ensure_list(gt_ids)


        # ----- ranking -----
        # sort DESCENDING
        rank_idx = np.argsort(-scores)[:topk]
        ranked_page_ids = page_ids[rank_idx]

        # ----- relevance vector (1 = relevant, 0 = not) -----
        rel = np.isin(ranked_page_ids, gt_ids).astype(float)

        # ----- metrics per k -----
        for k in ks:
            rel_k = rel[:k]

            # 1. HIT@k
            hits[f"HIT@{k}"] += float(rel_k.sum() > 0)

            # 2. NDCG@k
            if rel_k.sum() == 0:
                ndcgs[f"NDCG@{k}"] += 0.0
            else:
                # DCG
                dcg = (rel_k / np.log2(np.arange(2, rel_k.size + 2))).sum()
                # IDCG = best possible DCG@k
                ideal = np.sort(rel)[::-1][:k]
                idcg = (ideal / np.log2(np.arange(2, ideal.size + 2))).sum()
                ndcgs[f"NDCG@{k}"] += float(dcg / idcg)

            # 3. MRR@k
            first_hit = np.where(rel_k == 1.0)[0]
            if first_hit.size > 0:
                mrrs[f"MRR@{k}"] += float(1.0 / (first_hit[0] + 1.0))
            
            # 4. recall@k (multiple relevant documents)
            if len(gt_ids) > 0:
                recall[f"Recall@{k}"] += float(rel_k.sum() / len(gt_ids))

    # ----- average -----
    for k in ks:
        hits[f"HIT@{k}"] /= n_queries
        ndcgs[f"NDCG@{k}"] /= n_queries
        mrrs[f"MRR@{k}"] /= n_queries
        recall[f"Recall@{k}"] /= n_queries

    # flatten into one dict
    return {**hits, **ndcgs, **mrrs, **recall}