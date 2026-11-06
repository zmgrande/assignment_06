"""
explore.py — run the pipeline one step at a time on the five test reviews. (Provided.)

Tests do not stop at breakpoints; this script does. Open it, set a breakpoint on
any line below (or inside one of your functions), and run it with
Run and Debug -> "Python Debugger: Current File". Each step prints what came
back, so you can see a table take shape the moment your function builds it.

The first run calls the API (about 15 calls for the five reviews — the subject
step is slow the first time, seconds per mention). After that everything comes
from cache/, so run it as often as you like. Delete a file in cache/ to redo
that one step.

A step you have not written yet returns None; the script says so and stops.
"""

import json

import pandas as pd

from reviews import (
    TEST_REVIEWS_FILE,
    build_mentions,
    cache_path,
    entities_step,
    entities_table,
    entity_mentions,
    load_aliases,
    opinions_table,
    sentences_table,
    sentiment_step,
    subjects_step,
    summarize_items,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 50)


def show(label: str, result) -> None:
    """Print one step's result, or stop with a hint if the step is not written yet."""
    print(f"\n=== {label}")
    if result is None:
        print("    returned None — this function is still a stub. Write it, run again.")
        raise SystemExit(0)
    if isinstance(result, pd.DataFrame):
        print(f"    {result.shape[0]} rows x {result.shape[1]} columns")
        print(result.head(15))
    else:
        print(json.dumps(result[0], indent=1)[:1500], "\n    ...")


reviews = pd.read_csv(TEST_REVIEWS_FILE)
print(reviews)

documents = sentiment_step(reviews, cache_path(TEST_REVIEWS_FILE, "sentiment"))
show("Step 2: sentiment_step (first document)", documents)
entity_documents = entities_step(reviews, cache_path(TEST_REVIEWS_FILE, "entities"))
show("Step 2: entities_step (first document)", entity_documents)

sentences = sentences_table(documents)
show("Step 2: sentences_table", sentences)
opinions = opinions_table(documents)
show("Step 2: opinions_table", opinions)
entities = entities_table(entity_documents)
show("Step 2: entities_table", entities)

entity_rows = entity_mentions(entities, sentences, opinions)
show("Step 3: entity_mentions", entity_rows)
if opinions is not None and entity_rows is not None:
    combined = pd.concat([opinions, entity_rows], ignore_index=True)
    subjects = subjects_step(reviews, combined,
                             cache_path(TEST_REVIEWS_FILE, "subjects", "csv"))
    show("Step 2: subjects_step", subjects)
    mentions = build_mentions(reviews, sentences, opinions, entity_rows, subjects,
                              load_aliases())
    show("Step 3: build_mentions", mentions)
    show("Step 3: summarize_items", summarize_items(mentions))
