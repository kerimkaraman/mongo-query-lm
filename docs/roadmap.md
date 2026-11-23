# Roadmap

Each step teaches one concept and shows it in real numbers. A step is ticked only after it is understood, approved, and committed.

## Part A: a language model from scratch

- [x] **0. Data generator:** define the e-commerce schema and generate thousands of example queries from code.
  Concepts: synthetic data, variety, reproducibility (seeds).
- [x] **1. Tokenizer:** turn query text into numbers and back.
  Concepts: token, vocabulary, encode/decode.
- [ ] **2. Counting bigram:** count which character follows which, then sample new queries.
  Concepts: probability, sampling. First measurement: valid JSON rate.
- [ ] **3. Loss:** score how good a model is.
  Concepts: negative log likelihood; random model vs counting model.
- [ ] **4. Neural bigram:** learn the same table with a neural network.
  Concepts: tensor, weights, gradient, autograd, gradient descent.
- [ ] **5. MLP:** look at the previous few characters.
  Concepts: embedding, hidden layer, train/validation split, overfitting.
- [ ] **6. Self-attention:** let a character look back at a far-away one (a `}` at its `{`).
  Concepts: attention, causal mask.
- [ ] **7. Small transformer:** put the pieces together.
  Concepts: transformer block, `nn.Module`. Measurement: valid query rate on a real MongoDB.

## Part B: question → query

- [ ] **8. Conditional generation:** the generator also writes questions; our transformer learns question → query.
- [ ] **9. Pretrained model, zero-shot:** load a small pretrained model and measure it without training (baseline).
- [ ] **10. Full fine-tuning:** train all of the pretrained model's weights on our pairs.
- [ ] **11. LoRA:** freeze the model, train small adapters instead.
  Concepts: frozen weights, adapter, rank.
- [ ] **12. Evaluation:** compare our model, the LoRA model, and the base model.
  Concepts: execution accuracy, a hand-written test set for real generalization.
- [ ] **13. (optional) Scaling up:** a bigger model, QLoRA, a cloud GPU.
  Concepts: quantization.

## What I learned

Filled in after each approved step.

### 0. Data generator
- A model only learns from examples; with no dataset, I generate one from code. Every generated query is valid by construction and matches the schema.
- One seeded random generator drives every choice, so seed 42 always gives the same 5,000 queries and seed 43 gives different ones.
- Variety is measurable: equality appears ~4,200 times but `$gt` only ~420, and 16% of filters are duplicates. The model will learn frequent patterns best.
- The dataset is small: ~290K characters with 53 distinct characters, and the longest query is 152 characters (useful later for choosing context length).

### 1. Tokenizer
- A model only works with numbers. A tokenizer splits text into tokens and gives each one an id: `encode` turns text into ids, `decode` turns ids back into text.
- Mine is character-level and written from scratch: the vocabulary is every character in the data plus an end token, 54 in total. Sorting it makes the ids reproducible.
- The end token (`\n`, id 0) marks where a query ends, so a model can learn to stop instead of running on forever.
- There is no universal tokenizer. Every model family has its own, and a model only works with the tokenizer it was trained with.
