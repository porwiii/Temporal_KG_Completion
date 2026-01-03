## Overview of Temporal Graph Models
https://github.com/jiapuwang/Awesome-TKGC

## Inductive Models repositories and papers:
### 1. Inductive Representation Learning on Temporal Graphs (ICLR 2020)   
Repo: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs 

Papers:
* https://openreview.net/pdf?id=rJeW1yHYwH
* Follow-up work: https://arxiv.org/abs/2103.15213
* Predecessor work: https://arxiv.org/abs/1911.12864
   
  
### 2. Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks
Repo: https://github.com/snap-stanford/CAW
   
Paper: https://arxiv.org/abs/2101.05974v5

---

## Experiments

### TCompoundE: Evaluation Summary

In this part of the project, we evaluated **TCompoundE**, an embedding-based temporal knowledge graph completion (TKGC) model, on the **tkgl-smallpedia** dataset. The dataset is characterized by a strong temporal split and a high *surprise index*, with many entities appearing only in the validation or test sets.

Since TCompoundE assumes a **closed-world setting**, where all entities must be observed during training, the original dataset could not be used directly. To enable evaluation, we filtered out validation and test triples containing entities unseen during training. While this preprocessing step avoids undefined embeddings, it substantially reduces the difficulty of the task by lowering the surprise index.

After preprocessing, the model was trained and evaluated using standard ranking metrics. The results reveal a clear generalization gap: TCompoundE achieves an MRR of approximately **0.53** on the training set, but only around **0.20–0.23** on validation and test data. This indicates limited generalization ability even after restricting evaluation to known entities.

These results should be interpreted as an **upper bound** on TCompoundE’s performance for this dataset. In the original high-surprise setting, performance would likely degrade further or evaluation would fail altogether. Overall, this experiment highlights the limitations of embedding-based temporal KG completion models when applied to datasets with evolving entity sets, motivating the exploration of more inductive approaches.

---

### TGAT: Applicability to Temporal Knowledge Graph Completion

In this part of the project, we investigated whether **TGAT**, a temporal graph neural network designed for link prediction in interaction graphs, could be applied to the Temporal Knowledge Graph Completion (TKGC) task on the **Smallpedia** dataset.

TGAT is formulated to solve a **binary temporal link prediction** problem: given two nodes and a timestamp, predict whether an edge exists between them. In contrast, TKGC requires ranking candidate tail entities given a **(head, relation, timestamp)** query, which is a relational and multi-class ranking problem.

This mismatch leads to several fundamental incompatibilities. TGAT does not condition predictions on relation identity, treats relations only as auxiliary edge features, and is trained using binary labels with randomly sampled negatives. Smallpedia, however, consists solely of positive relational facts and requires filtered ranking-based evaluation using metrics such as MRR and Hits@k, which TGAT does not support.

Empirically, attempts to train TGAT on Smallpedia resulted in chance-level performance across all metrics (loss ≈ 1.386, accuracy/AUC/AP ≈ 0.5), indicating random guessing. This behavior is not due to implementation errors but reflects a misalignment between TGAT’s learning objective and the TKGC task.

In summary, although TGAT can be executed on the dataset, its architecture and training objective are unsuitable for Temporal Knowledge Graph Completion. This experiment further underscores the need for models explicitly designed for relational, time-aware ranking tasks when working with temporal knowledge graphs.

 
