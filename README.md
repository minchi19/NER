# Named Entity Recognition (NER)

# Introduction
NER is the task of identifying useful information (entities) in text and categorising them into categories like names of persons, organizations, locations, time, quantities, monetary values and percentage. Entity can be a word or sequence of words in a sentence. Named-entity recognition is also known as entity identification, entity chunking and entity/information extraction. With the application of NER, a high level overview of any document can be obtained.

# Objective:
Given a sentence as input(list of words), the objective here is to predict NER tag for each word in the given sentence into pre defined classes of entities.

# EDA
The dataset has the following columns or features -

Index - Index numbers for each word [Numeric type]
Sentence # - The number of sentences in the dataset (We will find the number of sentences below) [Numeric type]
Word - The words in the sentence [Character type]
POS - Parts Of Speech tags, these are tags given to the type of words as per the Penn TreeBank Tagset [Categorical type]
Tag -The tags given to each word based on the IOB tagging system described above (Target variable) [Categorical type]

### Key Insights
1. Most of the words are tagged as outside of any chunk(O). These words can be considered as fillers and their presence might impact the classifier performance as well.
2. dataset mostly contains words related to geographical locations, geopolitical entities and person names.


# Performance metrics  
Performance metrics we will use the following metrics to evaluate the models -

Precision
Recall
F1 score
The metrics mentioned above are calculated using True/False positives and True/False negatives respectively.
Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
Precision=TP/TP+FP
Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.
Recall=TP/TP+FN

F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is the harmonic mean of the both Precision and Recall
F1−Score=2∗(Recall∗Precision)/(Recall+Precision)

# Modelling

1. Baseline Model: Conditional Random Field (CRF)
A Conditional Random Field (CRF) is a powerful sequence prediction model that works by modeling the conditional probability of a sequence of labels (tags) given the sequence of input observations (words). CRFs are particularly useful for structured prediction tasks like Named Entity Recognition (NER) because they consider the dependencies between labels, improving prediction accuracy for sequential data.

Architecture:

Input: Sequence of words (tokens) in a sentence.
Output: Sequence of predicted tags for each word (e.g., B-person, I-location, O).
CRF Layer: Predicts the tag sequence by modeling the joint distribution of the sequence of tags, conditioned on the input sentence.

Shortcomings of the CRF Baseline:

Class Imbalance:
Issue: The large imbalance between the classes (especially with the O class dominating) can make the model overly biased towards predicting the majority class (O), resulting in poor recall for rare entities like I-art, I-nat, etc.
Solution: This can be mitigated by applying class weights to penalize incorrect predictions on rare entities, or using oversampling techniques for underrepresented classes.
Low Recall for Rare Entities:
Issue: For classes like I-art, B-eve, I-nat, and others, the recall is low due to their underrepresentation in the training data.
Solution: Using techniques like data augmentation, semi-supervised learning, or synthetic data generation (such as back-translation) can help improve recall for these classes. Transfer learning from pre-trained models can also assist in improving performance on rare entities.
Lack of Contextual Understanding:
Issue: CRF models on their own do not capture the deep contextual information between words because they treat each word's prediction separately based on features like word embeddings.
Solution: Combining CRF with a deep learning layer like LSTM or BiLSTM can enhance contextual understanding by allowing the model to capture dependencies between words over long sequences.

CRF Result
  precision    recall  f1-score   support

       B-art       0.39      0.09      0.15       330
       B-eve       0.55      0.35      0.43       234
       B-geo       0.86      0.88      0.87     30163
       B-gpe       0.96      0.93      0.95     12674
       B-nat       0.67      0.38      0.48       161
       B-org       0.77      0.64      0.70     16181
       B-per       0.84      0.78      0.81     13620
       B-tim       0.92      0.88      0.90     16347
       I-art       0.11      0.03      0.04       254
       I-eve       0.32      0.21      0.25       184
       I-geo       0.79      0.75      0.77      5928
       I-gpe       0.96      0.58      0.72       149
       I-nat       0.67      0.31      0.43        32
       I-org       0.72      0.64      0.68     13473
       I-per       0.84      0.88      0.86     13914
       I-tim       0.83      0.75      0.79      5204
           O       0.98      0.99      0.99    710413

    accuracy                           0.96    839261
   macro avg       0.72      0.59      0.64    839261
weighted avg       0.96      0.96      0.96    839261

# Improvement Model: LSTM

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies in sequential data. Unlike standard RNNs, LSTMs have specialized gates (input, forget, and output gates) that help mitigate the vanishing gradient problem, making them more effective for learning from longer sequences.

### LSTM Model Architecture
Embedding Layer: Converts words into dense vector representations (128-dimensional).
LSTM Layers: Captures long-term dependencies and context in the sequence. Multiple LSTM layers (64, 128 units) with dropout to prevent overfitting.
TimeDistributed Layer: Outputs predictions (entity tags) for each word in the sequence.
Dropout Layer: Helps prevent overfitting.

How It Addresses Shortcomings
Class Imbalance: The model  use class weights to give more importance to rare classes during training.
Low Recall for Rare Entities: LSTMs capture context, improving performance on rare entities.

# Advanced Model: BiLSTM

What is BiLSTM?

Bidirectional LSTM (BiLSTM) is an extension of the LSTM where the model processes the sequence of words in both forward and reverse directions, making it more powerful at capturing context from both the past and future.

### BILSTM Model Architecture

Embedding Layer: Converts words into dense vector representations.
Bidirectional LSTM Layer: Processes the sequence in both directions (forward and backward) to capture richer context.
TimeDistributed Layer: Outputs predictions (entity tags) for each word in the sequence.
Dropout Layer: Helps prevent overfitting.

# Future Scope: Fine-Tuning and Optimizations

1. Fine-Tuning LSTM/ BiLSTM Models:

Hyperparameter Optimization: Experiment with tuning hyperparameters like the number of layers, hidden units, and dropout rates to find the optimal configuration.
Learning Rate Scheduling: Use adaptive learning rates or learning rate decay to prevent overfitting and ensure better convergence.
Longer Sequences: For certain datasets, increasing the sequence length could improve context capture. This would require adjusting model architecture and padding strategies.

2. Embedding + LSTM:

Pre-trained Embeddings: Incorporate word embeddings like GloVe or Word2Vec to initialize the embedding layer with rich, pre-trained word representations that provide semantic meaning.
Character-level Features: Combine word-level and character-level embeddings to capture morphological information, which is especially useful for rare or out-of-vocabulary words.
3. Transfer Learning:

Pre-trained Language Models like BERT or RoBERTa can be fine-tuned on the NER task, providing a significant performance boost. These models have been pre-trained on large datasets and have learned contextual representations of words, making them powerful for NER tasks.


# Conclusion

CRF Model: Effective for sequential prediction but limited by class imbalance and lack of deep contextual understanding.

LSTM/BiLSTM Models: Enhance the ability to capture long-term dependencies and improve recall for rare entities.

Future Improvements: Fine-tuning models, using transfer learning, and leveraging pre-trained embeddings can significantly improve NER performance.
