# NER system

## Task description
Using [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) data build a classifier that will recognise Named Entities and their types in english texts. This is a multiclass classification task.

## Methods
CRF - Conditional Random Fields algorithm - was used for solving this problem. The realisation of the algorithm was taken from sklearn_crfsuite package.

## Procedure
1. Given features from a [tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-use-conll-2002-data-to-build-a-ner-system), build the CRF model to see how it performs on CoNLL-2003 data.
2. Analyze model performance and the impact of features chosen.
3. Based on the analysis made, implement additional features and measure the performance a new-built model again.

## Files
* NER.py - the script for training and testing the models.
* crf_model_baseline and crf_model_new_features - baseline model from step 1 in the section above and new-built model with additional features accordingly.
* baseline_info - performance of the baseline model according to F1-score and top of features according to its impact + analysis of models' performance.
* new_features_info - performance of the new-built model according to F1-score and top of features according to its impact.
* CoNLL-2003 - train and test data.

## Results
The most important result is the rise of total F1-score from 0.844 (baseline) to 0.866 (new-built) which is not really remarkable, but progress.