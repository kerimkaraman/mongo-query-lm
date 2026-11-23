# Bigram model: what it is and why it comes first

## Next token prediction

Every language model, from a tiny bigram to GPT or Claude, answers the same question:

> Given the text so far, what is the next token?

It writes text one token at a time, left to right, always looking back at what is already written, never forward.

In everyday English, after `th` the most likely next letter is `e` (because `the` is so common). A model learns patterns like this from data instead of being told them.

## What a bigram is

A **bigram** is two consecutive tokens. The bigrams of `{"age"` are `{"`, `"a`, `ag`, `ge`, `e"`.

A **counting bigram model** reads the whole dataset once, counts how often each character follows each other character, and turns every row of counts into probabilities. With 54 tokens in our vocabulary, the model is a 54 x 54 table: 2,916 numbers. That table *is* the model.

Real rows from our data:

```
'{' -> '"' 100%
'$' -> 'l' 31%   'g' 31%   'i' 14%   'n' 14%   'o' 9%
'}' -> '\n' 41%  '}' 27%   ',' 26%   ']' 6%
```

- `{` is always followed by `"`, because every JSON object starts with a quoted key. The model never gets this wrong.
- `$` has five options (`$lt`, `$gt`, `$in`, `$ne`, `$or`). The model rolls the dice, but every option leads to a valid query.
- `}` is the problem. Whether it should end the query or close a nested `{...}` depends on characters further back. The bigram cannot see them, so it ends the query after 41% of every `}`, including the ones that only close a nested block.

**Training is counting.** It takes well under a second, and after it the data is no longer needed; only the table is kept.

**Generating** starts from the end token, picks the next character at random according to the table row (sampling), and repeats until the end token appears again.

## Bigram vs a large language model

Both do next token prediction. The difference is how much they look at and how they store what they learned.

| | Bigram | Neural models (MLP, transformer, GPT) |
|---|---|---|
| Looks back at | Only the last token | A few tokens up to hundreds of thousands |
| Stores knowledge in | A table of counts | Learned weights |
| Learns by | Counting, once | Gradient descent, step by step |
| Unseen situations | Cannot predict (count is 0) | Generalizes from similar ones |

In short: the bigram looks only at the probabilities for the next step; an LLM looks at the big picture.

### Why not just a bigger table?

A table that looks further back grows explosively:

| Looks back | Table cells |
|---|---|
| 1 character (bigram) | 54² = 2,916 |
| 2 characters (trigram) | 54³ ≈ 157,000 |
| 8 characters | 54⁹ ≈ 3.9 × 10¹⁵ |

Our dataset has about 300,000 characters, so almost every cell of a large table would stay at 0. Neural networks solve this by learning similarities (`$gt` behaves like `$lt`, digits behave like digits) in a small number of weights, which carry over to combinations never seen in training.

## Why we build it

The bigram is not used to train later models; the transformer never sees this table and learns from the raw data on its own. The bigram is a **ruler** that the later models are measured against:

1. **Baseline.** Every later model must beat it on the same metrics (valid JSON rate, loss). Without a baseline, "the transformer is good" means nothing.
2. **Probability and sampling.** Every later model also ends by producing a probability for each token and sampling from it. Here that mechanism is visible without any neural network.
3. **Measurement code.** The valid JSON check written for the bigram is reused for every later model, so all models are measured the same way.
4. **Proof that training works.** In step 4 a neural network learns the same table by gradient descent instead of counting. If its weights match the counted table, training does what it should.
