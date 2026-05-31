# Tokenizer notes

Background reading for the M3 tokenizer (`pyds4/tokenizer.py`). Top-down:
what the model uses → how the pipeline is wired → each stage in detail.

The implementation is a clean-room port of `bpe_tokenize_text` in `ds4.c`
(lines 15041–15108), with byte-equal parity against `./ds4 --dump-tokens` on
20 oracle cases.

---

## 1. What tokenizer does DeepSeek V4 Flash use?

The model uses **the DeepSeek tokenizer family** (the same line as V2/V3):

- **Algorithm**: byte-level BPE (GPT-2 style).
- **Vocab size**: 129,280 — same as DeepSeek V3.
- **Special tokens**: full-width bracket style — `<｜begin▁of▁sentence｜>`,
  `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`. The `｜` is
  U+FF5C (not ASCII pipe); the `▁` is U+2581 (the SentencePiece underscore).
- **New for V4**: `<think>` / `</think>` reasoning markers, and `｜DSML｜`
  (DeepSeek's structured-markup tag).

DeepSeek's upstream release ships this as a HuggingFace `tokenizers`-library
`tokenizer.json`. The vocab size and special-token shapes line up with that
upstream file; we haven't byte-compared the merge tables directly, but our
parity oracle (`./ds4 --dump-tokens`) catches any disagreement that would
matter in practice.

**About `tokenizer.ggml.pre = "joyai-llm"`**: GGUF stores the pre-tokenizer
choice as a *name*, not a regex — runtimes are expected to recognize the name
and hardcode the matching split logic. `"joyai-llm"` is the identifier
antirez chose in ds4 (`ds4.c:15022` calls it "the JoyAI BPE pre-tokenizer").
Our 9-branch byte walker mirrors that ds4 walker — that's the only thing
we've actually verified. Whether the JoyAI rules are bit-equivalent to
DeepSeek's upstream HuggingFace pre-tokenizer is a separate claim we
haven't checked, and is not load-bearing for this project: our parity
target is ds4, not HF.

### Can we use an existing Python library instead?

Short answer: no, because of the pre-tokenizer.

| Library | BPE core | JoyAI pre-tokenizer |
|---|---|---|
| `tokenizers` (HF, Rust) | yes, fast | no — primitives don't compose to JoyAI's rules |
| `tiktoken` (OpenAI) | yes, fast | no — and can't load arbitrary merges easily |
| `sentencepiece` | different algorithm family | n/a |

You'd end up writing the pre-tokenizer in Python anyway, losing the speed
advantage. The ~330-line hand-rolled version is byte-equal on all oracle
cases and runs the test suite in 0.3s, so there's no perf pressure to swap.

If prefill ever becomes tokenizer-bound (it won't), the merge step is where
a priority-queue replacement would help — see § 4 below.

---

## 2. The pipeline

The four stages, in order:

```
raw text (str)
   │
   │  ① pre-tokenize     (split bytes into pieces)
   ▼
pieces: list[bytes]
   │
   │  ② byte-encode      (bytes → printable codepoints, 1-to-1)
   ▼
symbols: list[str]       per piece, one symbol per starting byte
   │
   │  ③ BPE merge        (greedy by rank, within each piece)
   ▼
merged symbols: list[str]
   │
   │  ④ vocab lookup     (string → integer ID)
   ▼
token IDs: list[int]
```

`encode_raw()` runs all four stages. `encode()` adds an outer wrapper that
scans for the 7 special-token literals and emits them by ID directly,
sending only the in-between text through `encode_raw`.

Stages run **independently per piece** — pre-tokenization is the gatekeeper
that decides which byte sequences are even allowed to consider merging. That
property is what gives BPE its linguistic priors (see § 3).

---

## 3. Pre-tokenizer

The stage that runs **before BPE**. It chops the raw input string into
smaller chunks ("pieces"). BPE then runs independently on each piece;
merges never cross piece boundaries.

### Why bother?

1. **It controls which merges are legal.** BPE greedily fuses adjacent
   symbols by lowest merge rank. If you fed it the whole string, it could
   fuse `"the"` with the leading space and the next word. Pre-tokenization
   forces those into separate pieces so they can't merge.

2. **It encodes linguistic priors cheaply.** "Don't merge across whitespace",
   "keep digit runs ≤3", "isolate CJK", "punct runs stay together" — all
   pre-tokenizer rules. They prevent the merge table from wasting capacity
   on cross-word junk.

3. **It must be exact.** The same merges trained against one pre-tokenizer
   produce different token streams under another. One off-by-one in the
   split logic and IDs diverge. That's why the JoyAI walker has 9 branches
   in a load-bearing order — see `_pretokenize` in `pyds4/tokenizer.py:171`.

### Concrete example

For `"def foo(x):"`:

```
pre-tokenizer pieces:  ["def", " foo", "(x", "):"]
                         │     │       │      │
                         │     │       │      └─ punct run (branch 7)
                         │     │       └─ punct + alpha (branch 3)
                         │     └─ space + letters (branch 5)
                         └─ letter run (branch 4)

BPE on each piece:     [def][ foo][(x][):]   → token IDs
```

BPE never considers merging `"foo"` with `"(x"`, because the pre-tokenizer
already split them.

### JoyAI vs GPT-2

GPT-2's pre-tokenizer is one regex:
`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`.

JoyAI's version is a hand-rolled byte walker (`_pretokenize`) because
regex engines don't run cleanly on raw bytes with custom UTF-8 handling and
because the rules don't fit cleanly into regex anyway.

Branches in order (re-ordering breaks parity):

1. Up to 3 ASCII digits.
2. CJK / Hiragana / Katakana run.
3. Punct followed by ASCII alpha — take the punct + alpha run.
4. Letter-like run (ASCII alpha or any non-ASCII byte).
5. One non-newline non-punct byte followed by letter-like (lets a single
   leading space attach to the next word: `"    int"` → `"   "`, `" int"`).
6. `" "` followed by punct — consume space + punct run + trailing newlines.
7. Bare punct run + trailing newlines.
8. Whitespace run. Newlines snap to end-of-last-newline; otherwise reserve
   a single trailing space for the next word piece.
9. Fallback: skip one UTF-8 char.

---

## 4. Byte-encode

GPT-2's trick to make BPE work on **arbitrary bytes** while still operating
on **printable text**.

### The problem

Classical BPE works on characters. But what character is byte `0x00`? Or
`0x09` (tab)? Or `0xC3` (the first byte of `é` in UTF-8)? You can't print
them, can't store them in a vocab file, can't eyeball them in a debugger.

You also can't just decode bytes as UTF-8 and operate on codepoints, because:

- Arbitrary user input might not be valid UTF-8.
- You want the tokenizer to be **lossless** — every byte sequence has
  exactly one tokenization, and decoding round-trips perfectly.

### The fix (bijective byte ↔ codepoint map)

256 raw bytes ↔ 256 printable Unicode codepoints. Convert bytes to that
codepoint string, run BPE on the string, you're back in "text-character"
land.

The map has two halves (`pyds4/tokenizer.py:33-56`):

1. **188 "printable" bytes map to themselves**: `0x21..0x7E` (94 visible
   ASCII), `0xA1..0xAC` (12), `0xAE..0xFF` (82). These render fine.

2. **The other 68 bytes** (`0x00..0x20`, `0x7F..0xA0`, `0xAD`) get pushed up
   to `U+0100..U+0143` — arbitrary printable codepoints in Latin Extended-A.
   These are the bytes that *don't* print cleanly: control chars, space,
   tab, newline, non-breaking space, etc.

```
byte    in text                  after byte-encode
----    -----------------        -----------------
0x20    ' '   (space)            'Ġ'  (U+0120)
0x0A    '\n'  (newline)          'Ċ'  (U+010A)
0x09    '\t'  (tab)              'ĉ'  (U+0109)
0x41    'A'                      'A'  (self)
0x61    'a'                      'a'  (self)
0xC3    1st byte of 'é'          'Ã'  (self — 0xC3 is in the self-map range)
0xA9    2nd byte of 'é'          '©'  (self)
```

The literal text `" hello"` becomes the string `"Ġhello"` before BPE. The
`Ġ` and `Ċ` you see scattered through GPT-2 / DeepSeek vocab files aren't
the model thinking in weird Unicode — they're the byte-encoded forms of
space and newline. A vocab entry like `"Ġthe"` decodes back to `" the"`;
the leading `Ġ` means "this token starts with a space".

### Why this is clean

- **Lossless**: bijective, so decode trivially recovers the original bytes.
- **Vocab-friendly**: every merges/vocab entry is printable text.
- **Robust input**: works on raw bytes including invalid UTF-8, so malformed
  input can't crash the tokenizer.
- **CJK still works**: `你` is UTF-8 bytes `E4 BD A0`, all in the self-map
  range. BPE has merges for common CJK byte sequences and recombines them.

---

## 5. BPE itself

**Byte Pair Encoding** — originally a 1994 compression algorithm, repurposed
in 2015 for NLP. The idea is dead simple: repeatedly replace the most common
adjacent symbol pair with a new symbol.

Two phases: **training** (one-time, offline — we don't do this; we inherit
DeepSeek's merge table) and **inference** (every encode call).

### Training (offline)

```
1. Initial vocab = single bytes (size 256 for byte-level BPE).
2. Count every adjacent pair of symbols across the corpus.
3. Most frequent pair, say ('t', 'h'), gets added to vocab as 'th'.
   Record the merge: ('t', 'h') → 'th' with RANK 0 (first learned).
4. Rewrite the corpus: every 't' followed by 'h' is now one symbol 'th'.
5. Repeat from step 2. Next merge might be ('th', 'e') → 'the', RANK 1.
6. Stop when vocab size hits N (DeepSeek: 129,280).
```

Output: a list of merges **in the order learned**. That order is the
**merge rank**. Lower rank = learned earlier = was more frequent.

Our GGUF stores this list (`tokenizer.ggml.merges`); we load it into
`merge_rank: dict[(left, right) → int]` (`pyds4/tokenizer.py:318-322`).

### Inference (`_bpe_piece`)

Given one piece — a list of starting symbols, one per byte after
byte-encoding — apply merges greedily, but **not by frequency** or **by
position**. By **rank**.

```python
# pyds4/tokenizer.py:361-371
while len(symbols) >= 2:
    best_i = -1
    best_rank = -1
    for i in range(len(symbols) - 1):
        r = rank.get((symbols[i], symbols[i + 1]))
        if r is not None and (best_i < 0 or r < best_rank):
            best_rank = r
            best_i = i
    if best_i < 0:
        break
    symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
```

Each iteration:

1. Look at every adjacent pair.
2. For each, check if it's a known merge.
3. Among matching pairs, pick the one with the **lowest rank**.
4. Merge it. Repeat. Stop when no adjacent pair matches.

**Why lowest rank wins** — not lowest position, not highest frequency:
the merge learned *first* during training was learned first because it was
the most common. Applying merges in learned-order at inference time
reproduces the training-time symbol stream exactly. That's what makes BPE
deterministic and matches inference distributions to training.

### Worked example

Suppose the merge ranks are:

| rank | merge        |
|------|--------------|
| 0    | (`t`, `h`)   |
| 1    | (`e`, `r`)   |
| 2    | (`th`, `e`)  |
| 3    | (`the`, `r`) |

Encode `"there"` (5 single-char symbols after byte-encoding):

```
Start:     [t, h, e, r, e]

Pairs:     (t,h)=0★  (h,e)=?  (e,r)=1  (r,e)=?
Best: (t,h) rank 0. Merge.

After:     [th, e, r, e]

Pairs:     (th,e)=2  (e,r)=1★  (r,e)=?
Best: (e,r) rank 1 — lower rank wins even though (th,e) is leftmost.

After:     [th, er, e]

Pairs:     (th,er)=?  (er,e)=?
No match. Stop.

Result:    [th, er, e]
```

If we'd been greedy left-to-right instead — merge `(th,e)` first because
it's leftmost — we'd get `[the, r, e]`, then `[ther, e]`, then `[there]`.
A totally different token stream. Hence the algorithm scans *all positions*
each iteration and picks by rank, not position.

### Why store merges instead of a flat vocab

The merge sequence is **constructive** — it tells you how to handle any
string, including ones never seen at training. A novel word like
`"thertheresther"` tokenizes cleanly because BPE only needs the merges,
not the words.

### Complexity

| Implementation | Complexity | Notes |
|---|---|---|
| Naive (our `_bpe_piece`) | O(n²) per piece | Scan all n−1 pairs at each of ≤ n merge steps. |
| Priority-queue (HF `tokenizers`, `tiktoken`) | O(n log n) per piece | Heap keyed by rank + linked-list pointers between symbols. |

For our parity-test purposes the O(n²) version is fine — pieces are tiny
(typically < 20 symbols after pre-tokenization). Test suite runs in 0.3s.

### Vocab lookup

After merges finish, look each merged symbol up in `token_to_id`
(`pyds4/tokenizer.py:377-386`). The vocab is guaranteed to contain every
byte-encoded starting symbol plus every merged symbol learned during
training, so lookups always succeed. There's a per-codepoint fallback path
for safety, but it never fires in practice.

---

## TL;DR

- **Tokenizer**: byte-level BPE with a custom 9-branch pre-tokenizer
  ("JoyAI", as antirez labels it in ds4.c). Vocab size and special-token
  shapes match DeepSeek V3 + V4 specials; parity is verified against ds4,
  not HuggingFace.
- **Pre-tokenizer**: splits text into pieces; BPE never crosses boundaries.
- **Byte-encode**: 256-byte ↔ 256-printable-codepoint bijection so BPE
  operates on text-shaped strings while staying lossless on arbitrary bytes.
- **BPE**: greedy merges by **lowest training rank** until no adjacent pair
  is mergeable. Train by frequency, apply by rank.
- **Special tokens**: matched as literal substrings before BPE runs; emitted
  by their fixed IDs.

See `pyds4/tokenizer.py` for the implementation, `tests/test_tokenizer.py`
for parity tests against `./ds4 --dump-tokens`.
