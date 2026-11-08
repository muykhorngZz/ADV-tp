# Fine-Tuning Gemma for Khmer Q&A (Unsloth + LoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13HhuzTUBf2z0j1YvtKlAnOdQCDDcM_6l)

This notebook fine-tunes Google's **Gemma** models to answer questions in **Khmer (ខ្មែរ)**. It uses [Unsloth](https://github.com/unslothai/unsloth) with 4-bit quantization and LoRA adapters, so a full run takes only a few minutes on one GPU.

It contains two experiments:

| | Run 1: Banking FAQ | Run 2: General Khmer Q&A |
|---|---|---|
| **Base model** | `unsloth/gemma-3-4b-it` | `unsloth/gemma-4-12b-it` |
| **Dataset** | 514 Khmer question/answer pairs about ACLEDA Bank products (custom CSV) | 5,000 random samples from [`rinabuoy/text-pairs-khmer-part-2`](https://huggingface.co/datasets/rinabuoy/text-pairs-khmer-part-2) (179,084 rows) |
| **Chat template** | `gemma-3` | `gemma-4` |
| **LoRA rank / alpha** | 8 / 8 | 64 / 64 |
| **Trainable params** | 14.9M of 4.3B (0.35%) | 262M of 12.2B (~2.1%) |
| **Epochs / steps** | 3 epochs / 87 steps | 1 epoch / 36 steps |
| **Effective batch size** | 16 (8 × 2) | 128 (64 × 2) |
| **Training time** | ~2 min | ~5.6 min |
| **Peak GPU memory** | 5.3 GB | 16.6 GB |
| **Validation loss** | 1.69 → **1.11** | 2.61 → **1.89** |

---

## What the notebook does

1. **Install & load the model.** Installs Unsloth and loads Gemma in 4-bit with `FastModel.from_pretrained`.
2. **Add LoRA adapters.** Only the language layers (attention + MLP) are trained; the vision tower stays frozen.
3. **Prepare the data.**
   - Run 1: loads the CSV from Google Drive, fixes the mis-parsed column names (`input,output` → `answer`, `__index_level_0__` → `question`), and drops 7 rows with no answer (521 → 514).
   - Run 2: loads the Hugging Face dataset and samples 5,000 rows.
   - Each pair is formatted with the Gemma chat template (`user` → `model`).
4. **Train.** Uses TRL's `SFTTrainer` with a 90/10 train/validation split, evaluates every 10 steps, and keeps the checkpoint with the lowest validation loss. `train_on_responses_only` makes sure the loss is computed only on the model's answers, not on the questions.
5. **Plot losses.** Draws training vs. validation loss and saves the plot and log history to Google Drive.
6. **Compare with the base model.** Runs the same prompt with LoRA turned off (`model.disable_adapter()`) and with it on.
7. **Save.** Saves the LoRA adapters and tokenizer locally (`gemma-3/`, `gemma-4/`).

---

## Results

### Run 1: Gemma 3 4B on ACLEDA Bank FAQ
- Training loss fell **3.53 → 1.11** and validation loss **1.69 → 1.11**. The gap at the end is only 0.006, so there is **no sign of overfitting**.
- **Before fine-tuning:** asked *"តើប័ណ្ណឥណពន្ធចេស៊ីប៊ី អេស៊ីលីដា ធម្មតា ជាអ្វី?"* ("What is the ACLEDA JCB Classic debit card?"), the base model made up an answer about China Construction Bank / ICBC.
- **After fine-tuning:** the model answers about the ACLEDA card in the same style as the training data.
- ⚠️ The fine-tuned model can still mix up facts and sometimes repeats itself on longer answers (see the "benefits" example). Check its answers against the source data.

### Run 2: Gemma 4 12B on general Khmer text pairs
- Validation loss fell **2.61 → 1.89** and was still going down at the last step, so the model is probably **under-trained**. A second epoch would likely help.
- On a general question (*"Why can airplanes fly?"*), the **base model gave a better answer** than the fine-tuned one. The dataset mostly teaches summarising, not answering, so it is not a good fit for this kind of question.
- ⚠️ The final test (*"At what temperature does water boil?"*) starts with a correct answer (100 °C) and then falls into a repetition loop. Adding `repetition_penalty` or training longer may fix this.

---

## Requirements

- A GPU runtime. The runs above used an **NVIDIA RTX PRO 6000 (96 GB)**, but Run 1 needs only ~6 GB and Run 2 ~17 GB, so a smaller GPU (e.g. an A100 or L4) should work. Run 1 may fit on a free T4.
- Python packages (installed by the first cell):
  ```bash
  pip install unsloth
  ```
  This also installs `torch`, `transformers`, `trl`, `peft`, `datasets` and `bitsandbytes`. The notebook also uses `pandas` and `matplotlib`.
- **Google Drive** access: the notebook mounts Drive to read the Run 1 dataset and to save the loss plots.

---

## How to run

1. Click the **Open in Colab** badge above.
2. Go to **Runtime → Change runtime type** and pick a GPU.
3. **Run 1 dataset:** this is a private CSV and is not included in this repo. Put your own CSV with `question` and `answer` columns in Google Drive, and change this path in the notebook:
   ```python
   dataset = load_dataset(path = "/content/drive/MyDrive/Colab Notebooks/Meikuan Project")
   ```
4. Run the cells from top to bottom. Run 2 downloads its dataset from Hugging Face automatically.
5. To load the saved adapters later, change `if False:` to `if True:` in the loading cell:
   ```python
   from unsloth import FastModel
   model, tokenizer = FastModel.from_pretrained(
       model_name = "gemma-3",   # or "gemma-4"
       max_seq_length = 2048,
       load_in_4bit = True,
   )
   ```

---

## Ideas for improvement

- Train Run 2 for 2+ epochs, or on more of the 179k examples.
- Use a Khmer **question-answering** dataset (not summarisation) for general Q&A.
- Add `repetition_penalty` (e.g. 1.1–1.2) when generating, to stop repetition loops.
- Fix the Run 1 CSV so the columns parse correctly without renaming.
- Evaluate on a held-out set with a metric (e.g. exact match or human review), not only loss.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth): fast LoRA fine-tuning. This notebook is based on their Gemma 3 notebook.
- [Google Gemma](https://ai.google.dev/gemma)
- [Hugging Face TRL](https://huggingface.co/docs/trl/sft_trainer)
- Dataset: [`rinabuoy/text-pairs-khmer-part-2`](https://huggingface.co/datasets/rinabuoy/text-pairs-khmer-part-2)

## Author

**Hang Muykhorng**
