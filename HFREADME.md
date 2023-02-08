---
annotations_creators:
- expert-generated
language:
- da
language_creators:
- expert-generated
license:
- afl-3.0
multilinguality:
- monolingual
pretty_name: DDisco
size_categories:
- 1K<n<10K
source_datasets: []
tags:
- discourse
- coherence
task_categories:
- text-classification
task_ids: []
---

# Dataset Card for DDisco

## Dataset Description

The DDisco dataset is a dataset which can be used to train models to classify levels of coherence in _danish_ discourse. Each entry in the dataset is annotated with a discourse coherence label (rating from 1 to 3):

1: low coherence (difficult to understand, unorganized, contained unnecessary details and can not be summarized briefly and easily)
2: medium coherence
3: high coherence (easy to understand, well organized, only contain details that support the main point and can be summarized briefly and easily).
Grammatical and typing errors are ignored (i.e. they do not affect the coherency score) and the coherence of a text is considered within its own domain.

### Additional Information

[DDisCo: A Discourse Coherence Dataset for Danish](https://aclanthology.org/2022.lrec-1.260.pdf)

### Contributions

[@ajders](https://github.com/ajders)