# Finnish propbank dataset assignment

## the goal
to predict semantic roles for words in finnish: multilabel classification task;

## data
[conll-u Finnish propbank](https://github.com/TurkuNLP/Finnish_PropBank/tree/data)

## features and target variable
id of a word in a sentence, its POS, its head and its deprel, POS and deprel were encoded by LabelEncoder; target variable are deprels of a word and they were encoded by MultiLabelBinarizer because it is a multilabel classification task;

## quality
accuracy measure is roughly 0.687, though it is clear that it is achieved due to the fact that the majority of words has no semantic role assigned; it means that this method can be used as a baseline only and needs further elaboration;
