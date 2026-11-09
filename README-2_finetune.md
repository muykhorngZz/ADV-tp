# Fine-Tuning Gemma for Khmer Question Answering (LoRA + Unsloth)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13HhuzTUBf2z0j1YvtKlAnOdQCDDcM_6l)



**Notebook:** `Hang Muykhorng.ipynb`
**Open in Google Colab:** https://colab.research.google.com/drive/13HhuzTUBf2z0j1YvtKlAnOdQCDDcM_6l
**Google Drive link:** https://drive.google.com/file/d/13HhuzTUBf2z0j1YvtKlAnOdQCDDcM_6l/view?usp=sharing

> The notebook is shared as *"Anyone with the link can view"*. All cells have already been run, so every output (training logs, loss tables, loss plots and model answers) can be seen without re-running anything.

This homework fine-tunes two instruction-tuned Gemma models on Khmer question–answer data with **LoRA** (via **Unsloth**, **PEFT** and **TRL**), monitors training and validation loss, and compares the answers of the base model with the fine-tuned model.



| | Experiment 1 | Experiment 2 (different dataset) |
|---|---|---|
| Model | `unsloth/gemma-3-4b-it` | `unsloth/gemma-4-12b-it` |
| Dataset | ACLEDA Bank FAQ (Khmer, custom CSV) | `rinabuoy/text-pairs-khmer-part-2` (Hugging Face) |
| Examples used | 514 (462 train / 52 validation) | 5,000 sampled from 179,084 (4,500 / 500) |
| Goal | Teach the model bank-specific facts | Teach a short, "teacher-explains-it-simply" answering style |

---

## Table of Contents

1. [Course Summary](#1-course-summary)
2. [Notebook Configuration](#2-notebook-configuration)
3. [Loss Analysis](#3-loss-analysis)
4. [Practical Experiment: Different Dataset](#4-practical-experiment-different-dataset)
5. [Observations and Notes](#5-observations-and-notes)
6. [How to Run the Notebook](#6-how-to-run-the-notebook)

---

## 1. Course Summary

### 1.1 The LLM fine-tuning workflow

The notebook follows the standard supervised fine-tuning (SFT) pipeline:

```
Install libraries ─► Load base model (4-bit) ─► Attach LoRA adapters ─► Prepare dataset
      ─► Apply chat template ─► Train on responses only ─► Monitor train / validation loss
      ─► Compare base vs fine-tuned answers ─► Save LoRA adapters ─► Reload & inference
```

| Step | What it does in the notebook |
|---|---|
| Installation | `pip install unsloth` (brings in `transformers`, `trl`, `peft`, `bitsandbytes`) |
| Load model | `FastModel.from_pretrained(..., load_in_4bit=True, full_finetuning=False)` |
| Add adapters | `FastModel.get_peft_model(...)` with rank `r` and `lora_alpha` |
| Data prep | Load CSV / Hugging Face dataset, rename columns, drop empty rows |
| Chat template | Convert each Q&A pair into Gemma's `user` / `model` turn format |
| Training | `SFTTrainer` + `train_on_responses_only` so loss is computed only on the answer |
| Evaluation | Hold-out validation split, `eval_loss` every 10 steps, best checkpoint kept |
| Inference | Same question asked with adapters **off** (base) and **on** (fine-tuned) |
| Saving | `model.save_pretrained(...)` stores only the small LoRA adapter weights |

### 1.2 Gemma 3

Gemma 3 is Google's family of open-weight models. The notebook uses **`gemma-3-4b-it`**, the 4-billion-parameter *instruction-tuned* version, so it already knows how to follow a chat format. It is multimodal (text + vision); since only text is needed, `finetune_vision_layers=False` keeps the vision tower frozen. Gemma 3 uses this chat format:

```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
```

Experiment 2 uses the newer **Gemma 4 (12B)**, whose chat template is different (`<|turn>user ... <turn|>`) and adds an empty "thought" channel to non-thinking answers. That's why the template and the `instruction_part` / `response_part` markers had to be changed.

### 1.3 PEFT and LoRA

**PEFT (Parameter-Efficient Fine-Tuning)** means updating only a small number of parameters instead of all model weights. This saves GPU memory, time and storage.

**LoRA (Low-Rank Adaptation)** freezes the original weight matrix `W` and learns a small update made of two thin matrices:

```
W' = W + (lora_alpha / r) · B·A        where A is r×d and B is d×r, with r ≪ d
```

- **`r` (rank)**: size of the adapter. Larger `r` means more capacity, but also more parameters and more risk of overfitting a small dataset.
- **`lora_alpha`**: scaling factor for the update. Setting `lora_alpha = r` gives a scale of 1.
- **Target modules**: adapters are attached to the attention (`q, k, v, o`) and MLP (`gate, up, down`) projections of the language layers.

Effect in this notebook:

| | Trainable params | Total params | % trained |
|---|---|---|---|
| Gemma 3 4B, `r=8` | 14,901,248 | 4,314,980,720 | **0.35 %** |
| Gemma 4 12B, `r=64` | 262,275,072 | 12,222,005,248 | **2.15 %** |

The base model is also loaded in **4-bit (QLoRA-style)**, so Gemma 3 4B used only ~5.2 GB of the L4 GPU's 22 GB during training.

### 1.4 Dataset preparation

1. **Load:** `load_dataset("csv", ...)` for the bank FAQ; `load_dataset("rinabuoy/text-pairs-khmer-part-2")` for experiment 2.
2. **Clean column names:** the CSV header came in as `input,output` and `__index_level_0__`, so they were renamed to `answer` and `question`.
3. **Remove missing values:** 7 rows had an empty `answer`. They were filtered out, leaving 521 → **514** rows.
4. **Subsample (exp. 2):** 179k rows is too many for a homework run, so a shuffled sample of **5,000** was used.
5. **Format with the chat template:** a `formatting_prompts_func` builds `[{"role":"user"}, {"role":"assistant"}]` messages and calls `tokenizer.apply_chat_template(...)`, removing the leading `<bos>` (the tokenizer adds it again).
6. **Train on responses only:** `train_on_responses_only` sets the labels of the question tokens to `-100`, so the model learns to produce answers and is not graded on repeating the question. The notebook checks this by decoding row 100: after masking, only the answer remains.

---

## 2. Notebook Configuration

### 2.1 Hyperparameters used

| Hyperparameter | Exp. 1 (Gemma 3 4B) | Exp. 2 (Gemma 4 12B) | Meaning / why |
|---|---|---|---|
| `max_seq_length` | 1024 | 2048 | Max tokens per example. Exp. 2 answers are longer explanations. |
| `load_in_4bit` | True | True | 4-bit quantised base model to save VRAM |
| `r` | 8 | 64 | Small rank for a small dataset; larger rank for more, more varied data |
| `lora_alpha` | 8 | 64 | Kept equal to `r` (scale = 1) |
| `lora_dropout` | 0 | 0 | No dropout on adapters (Unsloth-optimised default) |
| `finetune_vision_layers` | False | (default) | Text-only task |
| Validation split | 10 % | 10 % | `train_test_split(test_size=0.1)` |
| `per_device_train_batch_size` | 8 | 32 | Bigger batch on the larger GPU and larger dataset |
| `gradient_accumulation_steps` | 1 | 1 | Effective batch = batch size |
| `num_train_epochs` | 2 | 1 | Small dataset needs more passes; 4,500 rows is enough in 1 epoch |
| `learning_rate` | 2e-4 | 2e-4 | Standard LoRA learning rate |
| `warmup_steps` | 5 | 5 | LR ramps up for the first 5 steps |
| `lr_scheduler_type` | linear | linear | LR decays linearly to 0 |
| `optim` | `adamw_torch` | `adamw_8bit` | 8-bit AdamW saves optimizer memory for the 12B model |
| `weight_decay` | 0.01 | 0.01 | Light regularisation |
| `bf16` | True | True | Both GPUs support bfloat16 |
| `eval_steps` / `save_steps` | 10 / 10 | 10 / 10 | Validation loss every 10 steps |
| `load_best_model_at_end` | True | True | Keep checkpoint with lowest `eval_loss` |
| GPU | NVIDIA L4 (22 GB) | NVIDIA RTX PRO 6000 Blackwell (95 GB) | |

### 2.2 Total training steps

```
steps = ceil(train_examples / (batch_size × grad_accum)) × epochs

Exp. 1:  ceil(462 / 8)  × 2 = 58 × 2 = 116 steps
Exp. 2:  ceil(4500 / 32) × 1 = 141 steps
```

### 2.3 Training cost

| | Exp. 1 | Exp. 2 |
|---|---|---|
| Training time | 512.8 s (**8.55 min**) | 484.2 s (**8.07 min**) |
| Peak reserved GPU memory | 5.17 GB (23.5 % of 22 GB) | 16.6 GB (17.5 % of 95 GB) |

### 2.4 What I changed and why

- **Model size:** 4B → 12B for the second experiment, because open-domain Khmer explanations need more general knowledge than a bank FAQ.
- **LoRA rank:** `r=8` → `r=64`. The FAQ has only ~500 rows, so a small adapter avoids memorising noise; the second dataset is 10× larger and more diverse, so it gets more capacity.
- **Batch size / epochs:** 8 × 2 epochs → 32 × 1 epoch, so both runs took about the same number of steps (116 vs 141) and the same time (~8 min).
- **Optimizer:** switched to `adamw_8bit` to fit the 12B model comfortably.
- **Sequence length:** 1024 → 2048 for longer answers.
- **Generation settings** (inference): `temperature=0.5, top_p=0.95, top_k=64`, plus `repetition_penalty=1.15` in experiment 1 to stop the model from repeating itself.

---

## 3. Loss Analysis

**Training loss** measures how well the model predicts answers it is trained on. **Validation loss** measures the same thing on held-out questions it never saw. Comparing them shows:

| Pattern | Diagnosis |
|---|---|
| Both high and flat | **Underfitting**: model or training too weak |
| Both decreasing, small gap | **Good fit**: still learning and generalising |
| Training ↓ while validation ↑ | **Overfitting**: memorising the training set |

### 3.1 Experiment 1: Gemma 3 4B on ACLEDA FAQ

![Exp 1 loss](images/loss_exp1_gemma3_aceleda.png)

| Step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 116 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Training loss | 1.983 | 1.667 | 1.545 | 1.604 | 1.484 | 0.995 | 1.207 | 1.218 | 1.118 | 1.172 | 1.395 | 1.140 |
| Validation loss | 1.944 | 1.633 | 1.514 | 1.434 | 1.375 | 1.341 | 1.310 | 1.284 | 1.262 | 1.247 | 1.239 | **1.237** |

**Interpretation**

- Training loss falls sharply from about **3.4** in the first steps to about **1.1–1.2**. The steep drop in the first ~10 steps is the model learning the answer format and the bank's vocabulary.
- Validation loss decreases **at every evaluation**, from 1.944 to **1.237** (−36 %). The best checkpoint is the last one.
- The gap between training (~1.14) and validation (~1.24) at the end is small, and validation loss never goes up, so there is **no overfitting**.
- Validation loss is flattening (only −0.002 between steps 110 and 116), so most of the learning has happened. One more epoch could lower it a little further, but it would need watching for the point where validation loss starts to rise.
- The training curve is noisy (e.g., 0.995 at step 60, 1.395 at step 110) because each value is from a single batch of 8 examples. The validation loss, averaged over 52 examples, is the more reliable signal.

### 3.2 Experiment 2: Gemma 4 12B on Khmer Q&A pairs

![Exp 2 loss](images/loss_exp2_gemma4_khmer_qa.png)

| Step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 141 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Training loss | 1.972 | 1.903 | 1.949 | 1.808 | 1.840 | 1.785 | 1.927 | 1.712 | 1.734 | 1.872 | 1.828 | 1.706 | 1.724 | 1.768 |
| Validation loss | 1.927 | 1.878 | 1.855 | 1.838 | 1.822 | 1.808 | 1.799 | 1.792 | 1.781 | 1.774 | 1.769 | 1.762 | 1.758 | **1.757** |

**Interpretation**

- Validation loss decreases steadily from 1.927 to **1.757** (−9 %) and never goes up, so there is **no overfitting**.
- The training loss looks "unstable", but the y-axis of this plot only covers 1.70–2.05. The swings are about ±0.1, mostly because every batch contains very different questions (physics, politics, cameras, …).
- Training and validation loss stay close together (~1.77 vs ~1.76), and both are still slowly decreasing at the end. This points to **mild underfitting**: the model has not yet fully learned this data after 1 epoch on 5,000 of the 179,084 available rows. More data, more epochs or more steps would probably lower the loss further.

### 3.3 Why the final losses are different

Experiment 2 ends at a higher loss (1.76 vs 1.24), even though it uses a bigger model. This **does not mean it is worse**. The two datasets are very different:

- The **ACLEDA FAQ** is narrow and repetitive: the same bank name, card names and phrases appear again and again, so the answers are very predictable.
- The **Khmer Q&A pairs** cover open-ended topics, answers are free-form explanations, and many are machine-translated. There are many valid ways to answer, so the per-token loss stays higher.

Loss values can only be compared fairly **within the same dataset**.

---

## 4. Practical Experiment: Different Dataset

### 4.1 Dataset used

**`rinabuoy/text-pairs-khmer-part-2`** (Dr. Rina Buoy, Hugging Face): 179,084 Khmer `question` → `summary` pairs of general factual questions. The answers read like a teacher explaining something to a student: short, simple and to the point. The `text` column was renamed to `question`, 5,000 random rows were selected (`seed=42`), and the pairs were formatted with the **Gemma 4** chat template.

### 4.2 Results comparison

**Experiment 1: bank FAQ (Gemma 3 4B)**

Question: *តើប័ណ្ណឥណពន្ធចេស៊ីប៊ី អេស៊ីលីដា ធម្មតា មានសុពលភាពប្រើប្រាស់បានរហូតដល់រយៈពេលប៉ុន្មានឆ្នាំ* ("How many years is the ACLEDA JCB Classic debit card valid?")

| | Answer (summary in English) | Verdict |
|---|---|---|
| Base model (adapters off) | Misspells the card and bank names, treats the card as a **medicine**, and switches into Italian and Thai mid-sentence. Claims "3 years from production" and recommends asking a doctor. | ❌ Hallucination, mixed languages |
| Fine-tuned model | "ប័ណ្ណឥណពន្ធចេស៊ីប៊ី អេស៊ីលីដា ធម្មតា មានសុពលភាពប្រើប្រាស់បានរហូតដល់រយៈពេល ៥ឆ្នាំ។" (valid for **5 years**) | ✅ Correct, matches the dataset |

After reloading the saved adapters, the model also answered a different FAQ question (*does ACLEDA Mobile need an initial deposit / minimum balance?*) in fluent, bank-style Khmer.

**Experiment 2: general Khmer Q&A (Gemma 4 12B)**

| Question | Output |
|---|---|
| *ហេតុអ្វីបានជាយន្តហោះអាចហោះបាន* ("Why can airplanes fly?"), adapters **on** | Short, simple explanation that mentions *lift* (the dataset's style), but then **repeats one sentence in a loop** until the token limit. |
| Same question, adapters **off** (`model.disable_adapter()`) | Long, structured markdown answer explaining the "four forces of flight" (lift, weight, thrust, …), Bernoulli and Newton's 3rd law; a few misspelled Khmer words. |
| *តើទឹកពុះនៅសីតុណ្ហភាពប៉ុន្មាន?* ("At what temperature does water boil?"), fine-tuned | "ទឹកពុះនៅសីតុណ្ហភាព 212°F (100°C) ។ នេះគឺនៅកម្ពស់សមុទ្រ។ …" Correct, short, and adds that the boiling point is lower at high altitude, exactly the teacher-style tone of the dataset. |

### 4.3 Comparison of the two experiments

| | Exp. 1: ACLEDA FAQ | Exp. 2: Khmer Q&A |
|---|---|---|
| What the model learned | **New facts** (bank products, limits, fees) | **A style** (short, simple explanations) |
| Validation loss | 1.944 → 1.237 (−36 %) | 1.927 → 1.757 (−9 %) |
| Over/underfitting | Good fit, close to converged | Mild underfitting, still improving |
| Effect on answers | Big improvement: hallucination → correct fact | Answers became shorter and more direct; can loop without a repetition penalty |
| Main lesson | Fine-tuning is very effective for teaching domain knowledge the base model doesn't have | The base 12B model already knew general facts; fine-tuning mostly changed *how* it answers |

---

## 5. Observations and Notes

- **Labels in the experiment 2 inference cells are swapped.** The cell titled *"Base Model"* runs `model.generate(...)` with the adapters **active**, and the cell titled *"Fine Tuned model"* wraps generation in `with model.disable_adapter():`, which is actually the **base** model. Section 4.2 describes the outputs by whether the adapters were on or off. The short, looping answer comes from the fine-tuned model, which matches the dataset's short style; the long four-forces answer comes from the base Gemma 4.
- **Repetition loop:** the fine-tuned Gemma 4 output looped because `repetition_penalty` was not set in that cell (it was set to 1.15 in experiment 1). Adding `repetition_penalty=1.15` or lowering `max_new_tokens` would fix this.
- The markdown above the experiment 1 training cell says "1 %" validation, but the code uses `test_size=0.1` (**10 %**), which gives 462 / 52 examples.
- For experiment 2, the memory-stats cell shows "0.0 GB for training". This is because the peak memory was already reached before training started (16.6 GB reserved after loading), not because training used no memory.
- The xformers build did not match the installed PyTorch version, so Unsloth fell back to PyTorch attention. This affects speed only, not results.

---

## 6. How to Run the Notebook

1. Open the notebook with the **Open in Colab** button at the top.
2. **Runtime → Change runtime type → GPU.** Experiment 1 was run on an **L4**; experiment 2 (12B model) was run on an **RTX PRO 6000 (95 GB)**, although ~17 GB of VRAM was enough.
3. **Experiment 1** reads the ACLEDA FAQ CSV from Google Drive:
   `/content/drive/MyDrive/Colab Notebooks/Meikuan Project/dataasilida_cleaned.csv`
   Mount Drive (`from google.colab import drive; drive.mount('/content/drive')`) and upload the CSV to that path, or change the path.
4. **Experiment 2** downloads its dataset from Hugging Face automatically.
5. Run all cells from top to bottom. Each experiment trains in about **8 minutes**.
6. LoRA adapters are saved to `gemma-3/` and `gemma-4/`. To reload them, change `if False:` to `if True:` in the "load saved model" cells.





### Libraries

`unsloth 2026.11.3` · `transformers 5.17.0` · `trl 1.15.0` · `bitsandbytes 0.50.2` · `torch 2.14.1` · `datasets` · `pandas` · `matplotlib`




### References

- Unsloth: https://github.com/unslothai/unsloth
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021): https://arxiv.org/abs/2106.09685
- Hugging Face PEFT: https://github.com/huggingface/peft
- Dataset (exp. 2): https://huggingface.co/datasets/rinabuoy/text-pairs-khmer-part-2
