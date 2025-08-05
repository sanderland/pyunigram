# Unigram implementation in Python


## Sources

* [Kudo, 2018](https://aclanthology.org/P18-1007.pdf)
    * [SentencePiece implementation](https://raw.githubusercontent.com/google/sentencepiece/refs/heads/master/src/unigram_model_trainer.cc) which differs significantly.
    * [Talk linked therein](https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf)




--------------------------------------------------
SentencePiece Unigram        2,699,621         4.04
Hugging Face Unigram         3,188,064         3.42
Human Unigram                5,044,255         2.16


Tokenizer                       Tokens  Compression
--------------------------------------------------
Human Unigram                   39,473         1.59


Original: The quick brown fox jumps over the lazy dog
Length: 43 chars, 43 bytes

+---------------+------------------------------------------------------------+------------+---------------+
| Tokenizer     | Tokens                                                     |   # Tokens |   Compression |
+===============+============================================================+============+===============+
| Human Unigram | T h e ▁ q u i ck ▁ b r o w n ▁fo x ▁ j u mp s ▁over ▁t h e |         32 |          1.34 |
|               | ▁la z y ▁ d o g                                            |            |               |
+---------------+------------------------------------------------------------+------------+---------------+

Token counts:
  Human Unigram: 32 tokens


======================================== Example 2 ========================================

Original: I love programming in Python
Length: 28 chars, 28 bytes

+---------------+----------------------------------------------------+------------+---------------+
| Tokenizer     | Tokens                                             |   # Tokens |   Compression |
+===============+====================================================+============+===============+
| Human Unigram | I ▁ l o v e ▁pro g r a m min g ▁ i n ▁ P y t h o n |         23 |          1.22 |
+---------------+----------------------------------------------------+------------+---------------+

Token counts:
  Human Unigram: 23 tokens


======================================== Example 3 ========================================

Original: Taylor Alison Swift is an American singer songwriter
Length: 52 chars, 52 bytes

+---------------+--------------------------------------------------------------+------------+---------------+
| Tokenizer     | Tokens                                                       |   # Tokens |   Compression |
+===============+==============================================================+============+===============+
| Human Unigram | Taylor ▁Al i s o n ▁ S w if t ▁ i s ▁ a n ▁American ▁sin ger |         23 |          2.26 |
|               | ▁song write r                                                |            |               |
+---------------+--------------------------------------------------------------+------------+---------------+



--- after fix ---






