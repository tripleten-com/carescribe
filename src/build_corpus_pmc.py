"""Build data/raw/ once: real published case reports + model-written five-section summaries.

Notes: `full_note` from AGBonnet/augmented-clinical-notes (MIT), i.e. PMC-Patients narratives,
sampled by length. Summaries: written by a local 7B instruct model through Ollama under a
facts-from-the-note-only instruction. Resumable: rerun and it skips ids already written.

    python src/build_corpus_pmc.py --n 700
"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

RAW = Path("data/raw")
PAIRS = RAW / "pairs.jsonl"
LEXICON = RAW / "med_lexicon.json"
WRITER = "qwen2.5:7b-instruct"
SEED = 7
MIN_WORDS, MAX_WORDS = 150, 450

SECTIONS = [
    "Why you came in",
    "What we found",
    "Your medications",
    "What to do next",
    "When to seek help",
]

WRITER_SYSTEM = (
    "You write after-visit summaries for patients at a clinic. You are given a clinician's note. "
    "Write the summary in plain language a patient can follow, addressed to the patient as 'you', "
    "in exactly these five sections, in this order, each heading in bold markdown exactly as shown:\n"
    + "\n".join(f"**{s}:**" for s in SECTIONS)
    + "\nRules: use ONLY facts stated in the note. Never add a dose, a duration, a medication, a test "
    "result or an instruction the note does not contain. If the note names no medications, say so in "
    "one sentence. If the note gives no warning signs, write one generic line telling the patient to "
    "call the clinic if anything gets worse. Explain medical terms briefly in brackets. Keep the whole "
    "summary under 220 words. Output the five sections and nothing else."
)


def ollama_chat(system, user, temperature=0.2, num_predict=450):
    body = json.dumps({
        "model": WRITER,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": temperature, "seed": SEED, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"].strip()


GENERIC = {"therapy", "treatment", "medication", "medications", "surgery", "surgical", "follow-up", "antibiotic",
           "antibiotics", "intravenous", "infusion", "injection", "tablet", "tablets", "daily", "solution", "cream",
           "ointment", "drops", "fluids", "fluid", "saline", "blood", "transfusion", "oxygen", "chemotherapy",
           "radiotherapy", "radiation", "steroids", "steroid", "corticosteroids", "analgesics", "analgesia",
           "anesthesia", "anaesthesia", "general", "local", "topical", "systemic", "supportive", "course", "regimen",
           "dose", "doses", "high-dose", "low-dose", "cycles", "cycle", "adjuvant", "empirical", "empiric", "broad",
           "spectrum", "prophylaxis", "prophylactic", "replacement", "supplementation", "sodium", "potassium",
           "calcium", "vitamin", "acid", "human", "normal", "packed", "cells", "fresh", "frozen", "plasma", "platelet",
           "concentrate", "first", "second", "line", "combination", "based", "agents", "agent", "drugs", "inhibitor",
           "inhibitors", "blockers", "blocker", "proton", "pump", "channel", "molecular", "weight", "release"}


def build_lexicon(rows):
    """Drug-like single words from the source dataset's structured treatment lists. Only treatments
    that carry a dosage are read, because procedures do not have one; generic words are dropped.
    READ THE OUTPUT before trusting it: a lexicon with 'surgery' in it rejects every pair."""
    names = {}
    for row in rows:
        try:
            summ = json.loads(row["summary"], strict=False)
        except (json.JSONDecodeError, TypeError):
            continue
        treatments = summ.get("treatments")
        if not isinstance(treatments, list):
            continue
        for t in treatments:
            if not isinstance(t, dict) or str(t.get("dosage", "None")).strip().lower() in ("none", "", "n/a"):
                continue
            for w in re.findall(r"[a-z][a-z\-]{4,}", str(t.get("name", "")).lower()):
                if w not in GENERIC:
                    names[w] = names.get(w, 0) + 1
    return names


SOURCE_URL = "https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes/resolve/main/augmented_notes_30K.jsonl"
SAMPLE_CACHE = RAW / "source_sample.jsonl"


def sample_notes(n, scan_rows):
    """Stream the source file (355 MB) and stop after `scan_rows` rows rather than downloading it all.
    Keeps only the fields this build uses, and caches the sample so reruns are offline."""
    if SAMPLE_CACHE.exists():
        return [json.loads(l) for l in SAMPLE_CACHE.open()]
    band, seen = [], 0
    import ssl

    import certifi  # python.org builds on macOS ship no CA bundle for urllib
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(SOURCE_URL, timeout=120, context=ctx) as r:
        for line in r:
            seen += 1
            row = json.loads(line)
            if MIN_WORDS <= len(row["full_note"].split()) <= MAX_WORDS:
                band.append({"idx": row["idx"], "full_note": row["full_note"], "summary": row["summary"]})
            if seen >= scan_rows:
                break
    random.Random(SEED).shuffle(band)
    sample = band[:n]
    print(f"scanned {seen} rows, {len(band)} in the {MIN_WORDS}-{MAX_WORDS} word band, sampling {len(sample)}")
    with SAMPLE_CACHE.open("w") as f:
        for row in sample:
            f.write(json.dumps(row) + "\n")
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--scan-rows", type=int, default=6000)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    sample = sample_notes(args.n, args.scan_rows)

    if not LEXICON.exists():
        LEXICON.write_text(json.dumps(build_lexicon(sample), indent=0, sort_keys=True))

    done = set()
    if PAIRS.exists():
        done = {json.loads(l)["id"] for l in PAIRS.open()}
    t0 = time.time()
    with PAIRS.open("a") as out:
        for k, row in enumerate(sample):
            rid = f"pmc-{row['idx']}"
            if rid in done:
                continue
            note = row["full_note"].strip()
            summary = ollama_chat(WRITER_SYSTEM, f"Clinician's note:\n{note}")
            out.write(json.dumps({"id": rid, "note": note, "summary": summary}) + "\n")
            out.flush()
            if (k + 1) % 10 == 0:
                rate = (time.time() - t0) / max(1, k + 1 - len(done))
                print(f"{k + 1}/{len(sample)}  {rate:.1f}s per summary", flush=True)
    print("done", sum(1 for _ in PAIRS.open()), "pairs")


if __name__ == "__main__":
    main()
