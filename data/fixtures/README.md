# Fixtures

`phi_seeded.jsonl` holds eight short invented notes. Six carry identifiers that were planted on purpose, listed in
each row's `planted` field as `[label, exact text]`. Two carry none. Every name is invented, every phone number is
in a range reserved for fiction, and the email domain is `example.org`.

Why it exists: the project corpus is published, de-identified text, so a scrubber run only on the corpus finds
almost nothing, and you cannot tell a scrubber that works from one that does nothing. Run yours on this file and
count two things:

- **Recall**: how many of the planted spans did it replace? It should be all of them.
- **Precision**: what else did it replace? Read every one. A general-purpose NER model tags "Foley", "Doppler",
  "Fig", "McBurney", "Hodgkin", "Glasgow" and drug names such as "Bortezomib" as people. Replacing those with
  invented names destroys the clinical content your factuality check depends on, so treat NER hits as a list for
  you to read, and replace only what is really a person.

The dates, lab values, blood pressure and the PMID in these notes must NOT be replaced.

Your own known-answer fixture for the review script (duplicates, an empty summary, an over-length example) is
yours to write in Task 2; keep it in this folder.
