import numpy as np

def evaluate_graph(true_graph, learned_graph):
    true = true_graph.astype(int)
    pred = learned_graph.astype(int)

    TP = np.sum((true == 1) & (pred == 1))
    FP = np.sum((true == 0) & (pred == 1))
    FN = np.sum((true == 1) & (pred == 0))
    TN = np.sum((true == 0) & (pred == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }