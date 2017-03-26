Spec
==
For ~75 character input, be able to determine the programming language with >90% precision.


Steps
==
Raw
    plang.csv <- Run bigquery query for each of the programming languages and output them to csv grouped by type (determined via path extension).

Get a human sample for you to test and play against (1000 samples)


Model Character (read on karpathy's rnn generator)
==


Model Trigram
==
    Transform raw csv content into trigram representation.

    Combine each of the csv file content into a giant trigram and train glove word embedding vectors on it.

    Verify embedding makes sense (???)
