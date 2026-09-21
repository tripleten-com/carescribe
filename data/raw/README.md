# data/raw: the corpus you start from

This folder is tracked in Git and is your input. The dataset you build from it goes in `data/dataset/`, which is
tracked by DVC and never committed to Git.

| File | What it is |
|---|---|
| `pairs.jsonl` | 700 rows, one JSON object per line: `id`, `note`, `summary` |
| `med_lexicon.json` | 82 medication names found in the notes, for your factuality check. A starting point: it misses brand names and anything that did not appear in a structured treatment list |

**Notes** are real. They are patient narratives from case reports published in PubMed Central, taken from the
`AGBonnet/augmented-clinical-notes` dataset on Hugging Face (MIT licence), which is built on PMC-Patients. The
authors and the journals de-identified them before publication. They run 310 to 450 words.

**Summaries** were written for this course by an open 7B instruct model (`Qwen2.5-7B-Instruct`), from each note,
under the five-section template and a facts-from-the-note-only instruction. `src/build_corpus_pmc.py` is the script
that did it, kept here so you can read exactly how your labels were made. You do not need to run it: it needs a
7B model running locally and takes about three hours.

Nothing has been cleaned for you. The corpus is as it came out of that script, which means it contains what real
generated data contains: repeated notes, summaries that state something the note does not, and the occasional
identifier a journal missed. Finding them is Task 2.
