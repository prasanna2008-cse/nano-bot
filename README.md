# Nano GPT — a tiny ChatGPT built from scratch

A complete, small-scale chat model written **from scratch** in plain PyTorch.
No `transformers`, no pretrained weights, no API keys: the tokenizer, the
transformer, the training loop, the CLI and the web chat UI are all in this repo.

It is a college project, so the point is not to beat ChatGPT. The point is that
every part is small enough to read, run on a laptop CPU, and explain in a viva.

```
corpus: ~29,000 characters   vocab: 65 characters
model:  ~826,000 parameters, 4 transformer blocks
train:  ~35 minutes on a laptop CPU (no GPU needed)
```

## What is in the box

| File | What it does |
| --- | --- |
| `model.py` | GPT: token + position embeddings, causal self-attention, MLP blocks, weight tying, sampling |
| `tokenizer.py` | Character-level tokenizer (build, encode, decode, save/load) |
| `data/*.txt` | The conversation corpus used for training |
| `train.py` | Next-character training with AdamW, warmup, cosine decay, train/val loss |
| `prompts.py` | Builds the `User: ... / Bot: ...` prompt and trims the reply |
| `chat.py` | Terminal chat with streaming output |
| `app.py` | Zero-dependency web server + streaming `/api/chat` endpoint |
| `web/` | Chat UI (HTML/CSS/JS), no build step |
| `checkpoints/nano_gpt.pt` | The trained model, committed so the demo runs immediately |

## How the model works

The model is a decoder-only transformer that predicts **one character at a time**.
Everything below is implemented in `model.py` in about 150 lines.

1. **Tokenize.** Each character becomes an integer id. A 65-character vocabulary
   is enough for English conversation.
2. **Embed.** `x = wte[token] + wpe[position]` turns ids into vectors and tells
   the model where each character sits in the sequence.
3. **Attention.** For every position, project to queries, keys and values, then

   ```
   Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + mask) V
   ```

   The causal mask sets every future position to `-inf`, so a character can only
   look at characters before it.

4. **Feed-forward.** A two-layer MLP with GELU expands each vector to 4× width
   and projects it back, with residual connections and LayerNorm around both
   sub-layers (pre-norm).
5. **Predict.** A final linear layer produces one score per vocabulary entry.
   Softmax turns those scores into probabilities for the next character.
6. **Train.** Cross-entropy between predicted and true next character, optimised
   with AdamW. Gradients come from PyTorch autograd.
7. **Chat.** The prompt `User: ...\nBot:` is fed in, and characters are sampled
   (temperature + top-k) until the model writes a blank line or a new `User:`.

Parameter count is kept small with weight tying: the output projection shares
its matrix with the token embedding.

## Quickstart

```bash
pip install -r requirements.txt      # PyTorch only
python app.py                        # open http://127.0.0.1:8000
```

The committed checkpoint is used automatically. To talk in the terminal instead:

```bash
python chat.py
```

## Train it yourself

The corpus is concatenated from every `data/*.txt` file, so you can add your own
conversations in the same format:

```
User: what is a transformer
Bot: A transformer is a neural network that uses attention to look at all positions in a sequence at once.
```

Then retrain from scratch:

```bash
python train.py
```

Useful options:

```bash
python train.py --max-steps 6000          # longer run
python train.py --n-layer 6 --n-embd 192  # bigger model (slower on CPU)
python train.py --out checkpoints/big.pt  # different output path
```

`train.py` prints train/val loss every 250 steps and writes the checkpoint plus
`checkpoints/tokenizer.json` at the end. Everything is deterministic given
`--seed`.

## Project structure

```
nano-gpt/
├── model.py            transformer + sampling
├── tokenizer.py        character tokenizer
├── prompts.py          prompt format and reply cleanup
├── train.py            training loop
├── chat.py             terminal chat
├── app.py              web server (standard library)
├── web/                chat UI
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/               conversation corpus
│   ├── intro.txt
│   ├── coding.txt
│   ├── study.txt
│   └── life.txt
├── checkpoints/        trained weights
├── requirements.txt
└── README.md
```

## Using the web API

`POST /api/chat` streams the reply as plain text:

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Who are you?","temperature":0.8,"top_k":40}'
```

Request fields:

| Field | Meaning |
| --- | --- |
| `message` | the new user message (required) |
| `history` | optional list of `[user, bot]` turns for context |
| `temperature` | randomness, `0.1`–`1.5` (default `0.3`) |
| `top_k` | sample only from the k most likely characters (default `40`) |

## Sample output

Real replies from the committed checkpoint at temperature `0.3`:

| You | Nano GPT |
| --- | --- |
| `hello` | Hello! I am Nano, a tiny language model built from scratch. What would you like to talk about? |
| `what is a transformer` | A transformer is a neural network that uses attention to look at all positions in a sequence at once. |
| `tell me a joke` | Why did the computer go to the doctor? Because it had a virus. |
| `how do i study for exams` | Study in short focused sessions, test yourself often, and review the topics you get wrong. |

Raise the temperature slider and the replies become more creative and more broken;
lower it and they become safer and more repetitive. That trade-off is easy to see
live in the UI, and it is a good thing to demonstrate in a viva.

### Training run

The shipped model was trained for 3,500 steps with a batch size of 24 and a
context of 160 characters. Final losses:

```
train loss 0.147   val loss 2.594
```

The gap is expected: 29k characters is far too little text for a model of this
size, so it largely memorises the corpus and loses accuracy on unseen wording.
Training briefly with more dropout and early stopping gives a lower validation
loss (1.72) but visibly worse replies, because the model has not learned enough
of the language yet. The project ships the memorising checkpoint on purpose, and
the comparison itself is documented here as a lesson about overfitting.

## Limitations

- **It is tiny.** It has under a million parameters and about 29k characters of
  training text, so it often produces imperfect or repetitive sentences.
- **It hallucinates facts.** It has no database and no internet access. It is a
  character predictor, not a knowledge base.
- **No memory.** Each request rebuilds the prompt from the last few turns that
  the browser sends. Nothing is stored on the server.
- **English only**, and only topics that appear in the corpus.

These limits are documented honestly rather than hidden, which is what makes the
project a good study of how large language models actually work.

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Nano GPT: tiny ChatGPT built from scratch in PyTorch"
git branch -M main
git remote add origin https://github.com/<your-username>/nano-gpt.git
git push -u origin main
```

Create the empty repository named `nano-gpt` on GitHub first (no README, no
`.gitignore`), then run the commands above.

## License

MIT — see `LICENSE`.
