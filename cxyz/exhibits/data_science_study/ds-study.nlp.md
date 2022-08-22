## models

- Discriminative vs. generative

## language models

A probability distribution over sequences of words.

Used in speech recognition, machine translation, part-of-speech tagging, information retrieval etc.

Unigram model is the bag of words model.

https://en.wikipedia.org/wiki/Language_model

Related:

- Bag of words
- tf-idf

## concepts

language phenomena:

- compositionality
- polysemy
- anaphora
- long-term dependencies
- agreement
- negation

http://ruder.io/nlp-imagenet/

## taxonomy vs ontology

https://pediaa.com/wp-content/uploads/2018/01/Difference-Between-Taxonomy-and-Ontology-Comparison-Summary.png

## embeddings

### Contextual embeddings

- Glove

### Non-contextual embeddings

- Flair
- ELMO
- BERT

## metrics

- BLEU
- ROUGE
- Edit/Levenshtein
- Word movers distance
- Sentence movers distance

## problems (ruder's taxonomy)

-

## problems

Classification

- Precision/recall/f1

Language modeling

- Predicting words given words that came so far
- Component of speech recognition and machine translation

Image captioning

- Generating textual description for a given image

Machine translation

- Translating from one language to another
- BLEU score

Question/answering

- Answered @
  ...

Speech recognition

- Transform audio of spoken language into human readable text
- Precision/recall

Document summarization

- Creating short meaningful descriptions of larger documents
- Related to natural language generation

Textual entailment

...

Sentiment analysis

- Precision/recall

Ranking

- "Learning to rank"
- ...

entity recognition

- annotation methods: markup, IOB, BIOES
- https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/