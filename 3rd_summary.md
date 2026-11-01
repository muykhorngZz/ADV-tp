# 3rd Lesson Summary: Fine-Tuning LLMs with Unsloth

---

## 1. Introduction to Unsloth

**What is Unsloth?**

- A Python library that helps you fine-tune large language models (LLMs) efficiently.
- Supports popular models such as Llama-3, Mistral, Phi-4, and Gemma.
- Reduces memory usage by about 70% while keeping accuracy the same.

**Why use Unsloth?**

| Reason | Meaning |
|---|---|
| Faster training | Optimized code makes training finish sooner |
| Less memory usage | Works well even on computers with a small GPU |
| High accuracy | Gets strong fine-tuning results |

**Easy example:**
Normally, fine-tuning a model might need a GPU with 16 GB of memory. With Unsloth's 70% saving, the same job may fit in around 5 GB, so a free Google Colab GPU can be enough.

---

## 2. Installation Requirements

Before installing Unsloth, check these four things:

1. **Operating system** - Works on Linux, Windows, MacOS, Google Colab, and Kaggle.
2. **Python version** - Needs Python 3.9 or later.
3. **GPU support (optional but recommended)**
   - Supports NVIDIA CUDA GPUs for fast performance.
   - Can run on CPU, but it will be much slower.
4. **pip installed and updated** - pip is the Python package manager.

```bash
pip install --upgrade pip
```

**Easy example:**
Check your Python version first:

```bash
python --version
```

If it shows `Python 3.8.10`, it is too old and you need to upgrade. If it shows `Python 3.11.4`, you are ready.

---

## 3. Customizable Chat Templates

A **chat template** is the format used to arrange a conversation before giving it to the model. Each model expects its own format, so templates can be changed to fit different models.

**Example: Llama-3 template**

```python
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
```

**Key points:**

- **Placeholders** - `{SYSTEM}`, `{INPUT}`, and `{OUTPUT}` are blanks that get filled with real text.
- **Role headers** - Separate sections for `system`, `user`, and `assistant`.
- **Special tokens** - Markers that tell the model where things start and stop:

| Token | Meaning |
|---|---|
| `<\|begin_of_text\|>` | Start of the whole conversation |
| `<\|start_header_id\|>` | Start of a role name |
| `<\|end_header_id\|>` | End of a role name |
| `<\|eot_id\|>` | End of one message (end of turn) |

**Easy example:**
If SYSTEM = "You are a math teacher.", INPUT = "What is 2 + 3?", and OUTPUT = "2 + 3 = 5.", the filled template becomes:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a math teacher.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2 + 3?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

2 + 3 = 5.<|eot_id|>
```

Think of it like a form with labeled boxes: the model always knows which box is the instruction, which is the question, and which is the answer.

---

## 4. Benefits of Fine-Tuning with Unsloth

- **Task-specific adaptation** - The model becomes more accurate in a specialized area.
- **Resource efficiency** - Uses less GPU and memory during training.
- **Improved performance** - Answers become more relevant and precise.

**Code example: setting up a vision model for fine-tuning (LoRA)**

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,  # False if not fine-tuning the vision part
    finetune_language_layers   = True,  # False if not fine-tuning the language part
    finetune_attention_modules = True,  # False if not fine-tuning attention layers
    finetune_mlp_modules       = True,  # False if not fine-tuning MLP layers

    r = 16,              # Larger = more accurate, but may overfit
    lora_alpha = 16,     # Recommended: alpha equal to r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # Rank-stabilized LoRA support
    loftq_config = None, # LoftQ support
)
```

**What the code does:**

- Chooses which parts of the model to train: vision layers, language layers, attention modules, and MLP modules. All are set to `True` by default.
- Sets the LoRA parameters:
  1. `r = 16` - the rank, meaning how much "learning space" is added.
  2. `lora_alpha = 16` - how strongly the new learning affects the model (recommended setting).
  3. `use_rslora = False` - rank-stabilized LoRA is turned off.

**Easy example of LoRA:**
Instead of rewriting a whole textbook (the full model), LoRA lets you add small sticky notes to the pages. The textbook stays the same; the sticky notes add new knowledge. A bigger `r` means bigger sticky notes: more room to learn, but also more risk of memorizing the training data (overfitting).

---

## 5. Fine-Tuning

**Definition:** Fine-tuning takes a model that is already trained (pretrained) and continues training it on a smaller, task-specific dataset. It keeps the general knowledge from large-scale training and adjusts the model's parameters to do better on the new task.

**Process:**

```
Large corpus of data        Task-specific dataset
        |                            |
        v                            v
      [LLM]  ------->  [Fine-tuning the model]  ------->  [Final Model]
                                                    (ready for real-world use)
```

1. An LLM first learns from a large amount of general data.
2. It is then trained again on a small dataset for one specific task.
3. The result is a final model ready to use in real applications.

**Easy example:**
A medical student first studies general medicine for years (pretraining). Then they spend extra time training only in heart care (fine-tuning). Now they are a heart specialist (final model), ready to treat real patients.

In AI terms: a general model like Llama-3 can be fine-tuned on 1,000 customer-support conversations from a phone shop, so it learns to answer that shop's customers accurately.

---

## Quick Review

| Topic | One-line summary |
|---|---|
| Unsloth | Python library for faster, lower-memory LLM fine-tuning |
| Requirements | Python 3.9+, updated pip, NVIDIA GPU recommended |
| Chat template | Fixed format with placeholders and special tokens for each role |
| LoRA settings | `r` and `lora_alpha` control how much and how strongly the model learns |
| Fine-tuning | Pretrained model + small task dataset = specialized model |
