# carescribe

Your repository for Project 3: fine-tune and evaluate a clinical summarizer. It arrives with the corpus, pinned
library versions and a machine check. Everything else (the spec, the scripts, the dataset, the report) is yours to
build, following the project lessons.

## What is already here

| Path | What it is |
|---|---|
| `data/raw/` | The corpus: 700 note-and-summary pairs and a medication lexicon. Read `data/raw/README.md` |
| `data/fixtures/phi_seeded.jsonl` | Eight invented notes with planted identifiers, for testing your scrubber |
| `requirements.txt` | Library versions the lessons were verified on. Use them as they are |
| `requirements-cuda.txt` | The same, plus 4-bit loading for NVIDIA GPUs |
| `constraints.txt` | Exact versions of every sub-dependency, so an install next month matches an install today |
| `src/check_env.py` | Reads your machine and tells you which hardware tier you are in |
| `src/build_corpus_pmc.py` | How the corpus was made. For reading, not for running |
| `notebooks/colab_train.ipynb` | Runs your training scripts on a free hosted GPU, if your laptop has none |
| `configs/`, `eval/`, `assets/` | Empty, ready for your work |

## Set up your machine

You need Python 3.10 to 3.13 and Git. Pick your machine below, then run the check at the end.

### Mac with Apple Silicon (M1, M2, M3, M4)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

Your GPU is used through PyTorch's `mps` device. Two things differ from the lessons, which assume NVIDIA:
4-bit loading (QLoRA) is not available, so you train plain LoRA with the base model in full precision, on the 0.5B
class or, with 16 GB, a 1.5B model as a stretch; and there is no `nvidia-smi`, so `check_env.py` prints the machine
line for your spec. Close memory-hungry apps before training and keep the lid open: macOS pauses a run when the
laptop sleeps (`caffeinate -i python src/train.py ...` prevents idle sleep).

If Python reports `CERTIFICATE_VERIFY_FAILED` when downloading, run the `Install Certificates.command` that came
with your python.org installer, once.

### Windows or Linux with an NVIDIA GPU

Install PyTorch from the CUDA index first. On Windows this matters: the default `pip install torch` gives you the
CPU build, and your GPU will sit unused.

```bash
python -m venv .venv
# Windows:            .venv\Scripts\activate
# Linux:              source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-cuda.txt -c constraints.txt
```

Run `nvidia-smi` and note the memory line and the driver version. If `torch.cuda.is_available()` is false after
this, update the NVIDIA driver before anything else.

### No usable GPU (any laptop, including Intel Macs)

Install `requirements.txt` as in the Mac section. You have two routes, and your spec names which one you took:
the 0.5B model under plain LoRA on the CPU, at hours per run; or `notebooks/colab_train.ipynb`, which runs the
same scripts on a free hosted GPU. Do Task 1 and the data work of Task 2 on your laptop either way.

Intel Macs cannot install current PyTorch at all. Skip `torch` there, do the data work locally, and use the Colab
notebook for every step that loads a model.

### Check it

```bash
python src/check_env.py
```

It prints your tier, confirms the pinned libraries import, and gives you the lines to paste into section 6 of
`finetune_spec.md`. Fix anything marked `[FAIL]` before you start Task 1.

## Three things that will save you an evening

1. **Do not upgrade the pinned libraries.** Newer major versions change behaviour the lessons rely on. For
   example, on `transformers` 5 `len(tokenizer.apply_chat_template(messages, tokenize=True))` returns 2, not the
   token count, and nothing warns you.
2. **Name your MLflow store.** Put `mlflow.set_tracking_uri("sqlite:///mlflow.db")` at the top of every script
   that logs, and use the same line before `mlflow.search_runs()`. It works on every MLflow version.
3. **Expect about 2.5 GB of downloads** the first time: the libraries, your base model (about 1 GB for 0.5B, 3 GB
   for 1.5B) and the BERTScore encoder `roberta-large` (1.4 GB). Start them before you sit down to work.

## Git and DVC

`data/raw/` is in Git. The dataset you build goes in `data/dataset/`, which `.gitignore` already excludes: you run
`dvc add data/dataset` and commit the small `data/dataset.dvc` pointer it writes. Commit your final adapter
(`runs/<name>/adapter/`, tens of megabytes); checkpoints are ignored.
