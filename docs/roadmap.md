# Roadmap

Each step teaches one concept and shows it in real numbers. A step is ticked only after it is understood, approved, and committed.

## Part A: a language model from scratch

- [x] **0. Data generator:** define the e-commerce schema and generate thousands of example queries from code.
  Concepts: synthetic data, variety, reproducibility (seeds).
- [x] **1. Tokenizer:** turn query text into numbers and back.
  Concepts: token, vocabulary, encode/decode.
- [x] **2. Counting bigram:** count which character follows which, then sample new queries.
  Concepts: probability, sampling. First measurement: valid JSON rate.
- [x] **3. Loss:** score how good a model is.
  Concepts: negative log likelihood; random model vs counting model.
- [x] **4. Neural bigram:** learn the same table with a neural network.
  Concepts: tensor, weights, gradient, autograd, gradient descent.
- [ ] **5. MLP:** look at the previous few characters.
  Concepts: embedding, hidden layer, train/validation split, overfitting, smoothing (unseen pairs have zero probability and infinite loss).
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

### 2. Counting bigram
- Every language model does next token prediction. A bigram looks only at the last character; an LLM looks at the whole text. (Like `th` -> `e` in English: the model learns such patterns from data, nobody writes them.)
- A counting bigram is a 54 x 54 table of "which character follows which". Training is counting, and the table *is* the model. Generating means rolling dice on one row at a time (sampling).
- It is a baseline, not a building block: later models never see this table, they have to beat it. Measured: **2.8% valid JSON** over 1,000 samples, and even those 28 are meaningless (`{"$gete": 17.3}`), so valid JSON alone overstates quality.
- Its limit shows at `}`: whether a query should end depends on characters it cannot see, so it closes brackets it never opened. Details: [docs/concepts/bigram.md](concepts/bigram.md).

### 3. Loss
- Valid JSON rate is too coarse to train with: a query is either valid or not. Loss scores every single prediction, so even a small improvement shows up.
- Loss is measured per token on real data: for each position, `-log` of the probability the model gave to the real next character, averaged over all 297,590 predictions.
- Uniform random model: 3.989 (= log 54). Counting bigram: **1.626**. In plain terms, the bigram gives the right character about 20% on average instead of 1.85%, like choosing among ~5 options instead of 54 (perplexity).
- Loss and valid JSON see different things: the bigram struggles most after digits (numbers are random, nobody can predict them), while the `}` mistakes that break whole queries barely move the average loss.

### 4. Neural bigram
- Training is a loop of four lines: compute the loss, clear old gradients, `loss.backward()` to get a gradient for every weight (autograd), then nudge every weight against its gradient. LLMs are trained with the same loop, just bigger.
- Traced one weight (`$` -> `g`, counted 31.2%): the model started at 1.7%, the gradient kept saying "increase", and it reached 31.0%. The gradient shrinks as the answer gets close.
- Plain gradient descent left rarely seen rows behind (loss 1.671, largest gap 37.5%). Adam gives every weight its own step size, so rare characters learn too: loss **1.627**, the counted table recovered with a 0.0% gap, in 50 steps instead of 300+.
- Same table, same knowledge: valid JSON stays at 2.7%. A better model has to look further back.
