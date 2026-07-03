# Study Guide: LLM Evolution, Tokenization & Prompt Engineering

---

# TOPIC 1: GPT, T5, BERT — Comparison & Evolution of LLMs

## 1.1 Evolution of Language Models (Short Timeline)

| Era | Model | Key Idea |
|---|---|---|
| Pre-2017 | RNN, LSTM, GRU | Sequential processing, struggled with long-range context |
| 2017 | **Transformer** (Vaswani et al., "Attention Is All You Need") | Self-attention replaces recurrence — parallel processing |
| 2018 | **BERT** (Google) | Bidirectional encoder, great at *understanding* text |
| 2018–2019 | **GPT / GPT-2** (OpenAI) | Decoder-only, great at *generating* text |
| 2019 | **T5** (Google) | Encoder-decoder, treats every NLP task as "text-to-text" |
| 2020+ | GPT-3, T5-XXL, InstructGPT, ChatGPT etc. | Scale + instruction tuning + RLHF |

**Core shift:** Before Transformers, models read text word-by-word (slow, forgot early context). Transformers use **self-attention** — every word looks at every other word simultaneously, so context is captured instantly and training can be parallelized on GPUs.

---

## 1.2 The Three Architecture Types

Think of Transformers as having two building blocks: an **Encoder** (reads/understands input) and a **Decoder** (generates output). Models differ in which blocks they use.

### A) Encoder-only → **BERT**
- Reads the *entire* sentence at once, both left-to-right and right-to-left ("bidirectional").
- Doesn't generate text — it produces rich contextual representations of tokens.
- Trained using **Masked Language Modeling (MLM)**: random words are hidden (`[MASK]`), and the model predicts them using context from *both* sides.
- Analogy: like filling in a crossword puzzle — you use clues from all directions.

```
Input:  "The [MASK] sat on the mat."
BERT predicts: "cat"  (using both "The" before and "sat on the mat" after)
```

### B) Decoder-only → **GPT**
- Reads text left-to-right only (**causal/autoregressive**) — it can only see previous words, never future ones.
- Trained to predict the **next word**, given all previous words.
- This is exactly what's needed for text generation (writing, chatting, completing).
- Analogy: like writing a story one word at a time, never allowed to peek ahead.

```
Input: "The cat sat on the"
GPT predicts next word: "mat"
```

### C) Encoder-Decoder → **T5**
- Encoder reads and understands the full input; Decoder generates the output based on that understanding.
- T5's big idea: **"Text-to-Text Transfer Transformer"** — every task (translation, summarization, classification, Q&A) is reframed as converting one text string into another.

```
Input:  "translate English to German: The house is wonderful."
Output: "Das Haus ist wunderbar."

Input:  "summarize: <long article>"
Output: "<short summary>"
```

---

## 1.3 Comparison Table (GPT vs BERT vs T5)

| Feature | **GPT** | **BERT** | **T5** |
|---|---|---|---|
| Architecture | Decoder-only | Encoder-only | Encoder-Decoder |
| Direction | Left-to-right (causal) | Bidirectional | Bidirectional (encoder) + causal (decoder) |
| Pretraining Task | Next-token prediction | Masked Language Modeling | Span corruption (mask spans, predict them) → text-to-text |
| Best At | Text generation, chat, completion | Classification, NER, Q&A (extractive), sentence understanding | Any task framed as text-to-text: translation, summarization, Q&A |
| Output Style | Free-form generated text | Labels / embeddings / spans (not fluent generation) | Generated text (like GPT) but task-conditioned |
| Strengths | Fluent generation, few-shot learning | Deep contextual understanding, strong on classification benchmarks (GLUE) | Extremely flexible — one model, many tasks |
| Limitations | Can't "see" future context, weaker at pure understanding tasks | Cannot generate coherent long text | Slightly more complex/expensive (two stacks) |
| Example Use Case | Chatbots, story writing, code generation | Spam detection, sentiment analysis, entity recognition | Summarization, translation, multi-task pipelines |

**Simple rule of thumb:**
- Need to **understand** text → BERT
- Need to **generate** text → GPT
- Need **flexibility across many tasks** in one unified format → T5

---

## 1.4 Slide-by-Slide Plan for Your 5-Slide Presentation

1. **Title Slide** – "GPT vs BERT vs T5: Evolution of LLMs"
2. **Evolution Timeline** – RNN → Transformer → BERT/GPT → T5
3. **Architecture Diagrams** – Encoder-only vs Decoder-only vs Encoder-Decoder (with the analogy examples above)
4. **Comparison Table** – (use the table in 1.3)
5. **Use Cases & Takeaways** – When to use which model + key limitations

---

# TOPIC 2: Tokenization Schemes & Embeddings

## 2.1 Why Tokenization Is Required
Models don't understand raw text — they understand **numbers**. Tokenization is the process of breaking text into smaller units (tokens) and mapping each to a numeric ID.

```
"unhappiness" → ["un", "happi", "ness"] → [1023, 4521, 892]
```

**Why not just split by whole words?**
- Vocabulary would explode (millions of unique words, typos, rare words).
- Can't handle unseen/out-of-vocabulary (OOV) words.

**Why not just split by characters?**
- Sequences become very long, losing meaning-per-token efficiency.

**Subword tokenization** is the middle ground — common words stay whole, rare words get broken into meaningful pieces.

---

## 2.2 Main Tokenization Schemes

### A) BPE — Byte Pair Encoding (used by GPT-2, RoBERTa)
- Starts with individual characters.
- Iteratively merges the **most frequent adjacent pair** of symbols into a new token.
- Repeats until desired vocabulary size is reached.

```
Start: l o w  ,  l o w e r  ,  n e w e s t
Step 1: merge "e"+"s" → "es" (most frequent pair)
Step 2: merge "es"+"t" → "est"
... continues until vocab size target is met
```

### B) WordPiece (used by BERT)
- Similar to BPE, but merges pairs based on **maximizing likelihood** of the training data (not just raw frequency).
- Uses `##` prefix to mark subword continuations.

```
"playing" → ["play", "##ing"]
"unaffordable" → ["un", "##afford", "##able"]
```

### C) SentencePiece (used by T5, XLNet, ALBERT)
- Treats text as a raw stream of Unicode characters (including spaces!) — no assumption of "words" separated by whitespace.
- Language-agnostic — works well for languages without spaces (e.g., Japanese, Chinese).
- Can implement BPE or Unigram Language Model internally.

```
"Hello world" → ["▁Hello", "▁world"]   # ▁ represents a space
```

### Quick Comparison

| Scheme | Used By | Key Trait |
|---|---|---|
| BPE | GPT-2, RoBERTa | Frequency-based merges |
| WordPiece | BERT | Likelihood-based merges, `##` prefix |
| SentencePiece | T5, XLNet | Whitespace-aware, language-agnostic |

---

## 2.3 Tokenization with Hugging Face (Code Example)

```python
from transformers import AutoTokenizer

# BERT (WordPiece)
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(bert_tok.tokenize("unaffordable housing"))
# ['una', '##ffo', '##rda', '##ble', 'housing']

# GPT-2 (BPE)
gpt_tok = AutoTokenizer.from_pretrained("gpt2")
print(gpt_tok.tokenize("unaffordable housing"))
# ['un', 'aff', 'ord', 'able', 'Ġhousing']   # Ġ marks a preceding space

# T5 (SentencePiece)
t5_tok = AutoTokenizer.from_pretrained("t5-small")
print(t5_tok.tokenize("unaffordable housing"))
# ['▁un', 'aff', 'or', 'd', 'able', '▁housing']
```

---

## 2.4 Embeddings & Positional Embeddings

**Token Embeddings:**
Once text is tokenized into IDs, each ID is mapped to a dense vector (e.g., 768 numbers for BERT-base). These vectors capture *meaning* — similar words end up close together in vector space.

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("I love NLP", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# torch.Size([1, 5, 768])  → 5 tokens, each represented by 768 numbers
```

**Why "positional" embeddings are needed:**
Self-attention has no built-in sense of word order — "dog bites man" and "man bites dog" would look identical to plain attention. Positional embeddings inject **order information** by adding a position-specific vector to each token embedding.

- **BERT/GPT (original):** Learned absolute positional embeddings (a vector for position 1, position 2, etc.)
- **Original Transformer paper:** Fixed sinusoidal functions (sin/cos waves of different frequencies)
- **T5:** Uses **relative** positional encoding (distance between tokens, not absolute position)

```
Final Input Embedding = Token Embedding + Positional Embedding (+ Segment Embedding in BERT)
```

---

# TOPIC 3: Prompt Engineering Techniques

## 3.1 Zero-shot, One-shot, Few-shot Prompting

**Zero-shot** — No examples given, just the instruction.
```
Prompt: "Classify the sentiment of this review: 'The food was cold and bland.'"
Output: Negative
```

**One-shot** — One example given before the actual task.
```
Prompt:
Review: "Amazing service, loved it!" → Sentiment: Positive
Review: "The food was cold and bland." → Sentiment:
Output: Negative
```

**Few-shot** — Multiple (2+) examples given to establish a pattern.
```
Prompt:
Review: "Amazing service!" → Positive
Review: "Terrible wait times." → Negative
Review: "It was okay, nothing special." → Neutral
Review: "The food was cold and bland." →
Output: Negative
```

**When to use which:**
- Zero-shot: simple/common tasks the model already "knows"
- Few-shot: tricky formatting, domain-specific tasks, or when you need consistent output structure

---

## 3.2 Role-Based & Instruction Prompting

**Role-based prompting** — Assign the model a persona to shape tone/expertise.
```
"You are a senior data scientist. Explain overfitting to a beginner in 3 sentences."
```

**Instruction prompting** — Give a clear, direct command with explicit constraints.
```
"Summarize the following article in exactly 3 bullet points, each under 15 words."
```

**Combining both (most effective in practice):**
```
"You are an expert technical writer. Summarize the following research abstract 
in 2 sentences, avoiding jargon, for a non-technical audience."
```

---

## 3.3 Prompts for Summarization, Classification, and Q&A

### Summarization
```
"Summarize the following text in 3 sentences, focusing only on key findings:
<text>"
```

### Classification
```
"Classify the following email as 'Spam' or 'Not Spam'. Respond with only one word.
Email: <email text>"
```

### Question Answering
```
"Based only on the context below, answer the question. If the answer isn't 
in the context, say 'Not found.'

Context: <passage>
Question: <question>"
```

---

## 3.4 Comparing Outputs Using Different Prompts

This is about testing how prompt *phrasing* changes output quality. A simple experiment structure:

| Prompt Variant | Example | What to Observe |
|---|---|---|
| Vague | "Tell me about this text." | Often generic, unfocused output |
| Specific + constrained | "List 3 key facts from this text, one sentence each." | More structured, easier to evaluate |
| Zero-shot | Direct question, no examples | Baseline performance |
| Few-shot | Question + 2-3 examples | Usually more consistent formatting |
| Role-based | "As an expert..." | Often more detailed/domain-appropriate tone |

**For your deliverable (Prompt Design Exercise):** pick ONE task (e.g., sentiment classification), run it through zero-shot vs few-shot vs role-based prompts on the same input, and record outputs side-by-side to show how prompt design affects results.

---

# Quick Recap Summary

| Topic | One-Line Takeaway |
|---|---|
| GPT vs BERT vs T5 | Decoder=generate, Encoder=understand, Encoder-Decoder=flexible text-to-text |
| Tokenization | Breaks text into subwords so models can handle any word, even unseen ones |
| Embeddings | Turn tokens into meaningful number vectors; positional embeddings add order |
| Prompt Engineering | The way you phrase a prompt (examples, role, constraints) directly shapes output quality |

