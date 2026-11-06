"""
aspects.py — Step 3: from API output to "what are people saying about what".

The tables from Step 2 each know one thing. This step puts them together into
`mentions` — one row per thing a customer had an opinion about — and then asks
the questions the owner cares about.

Two sources of mentions, because each API catches what the other misses:

- **Opinion targets** (Azure opinion mining): "the mocha latte was overpriced" —
  a target, its own sentiment, and the words behind it. The best evidence.
- **Named entities** (Azure NER): "Otto always says hi" has no opinion words for
  opinion mining to latch onto, but the entity recognizer knows Otto is a
  Person. These mentions borrow their sentence's sentiment.

No walkthrough. The docstrings state what each function returns; `tests/` says
how it is checked.
"""

import pandas as pd

from .apicalls import get_entities
from .pipeline import (
    entities_step,
    entities_table,
    opinions_table,
    sentences_table,
    sentiment_step,
    subjects_step,
)
from .settings import (
    ALIASES_FILE,
    LOCATION_WORDS,
    MENTION_CATEGORIES,
    SUBJECT_WORDS,
    SYSTEM_PROMPT,
    cache_path,
)

MENTION_COLUMNS = ["review_id", "sentence_id", "mention", "mention_sentiment",
                   "assessments", "mention_offset", "source"]


def entity_mentions(entities: pd.DataFrame, sentences: pd.DataFrame,
                    opinions: pd.DataFrame) -> pd.DataFrame:
    """Entities opinion mining missed, as mentions with their sentence's sentiment.

    *(Provided.)* Read it — `build_mentions` stacks its output under
    `opinions_table`'s, and explore.py shows you what it returns.

    Same seven columns as `opinions_table`. Each entity is placed in the sentence
    whose span contains it — `sentence_offset <= entity_offset <
    sentence_offset + sentence_length` — and takes that sentence's
    `sentence_id` and `sentence_sentiment` (as `mention_sentiment`).
    `assessments` is `""` and `source` is `"entity"`.

    Leave out an entity when an opinion target in the same review and sentence
    already covers it: their lowercased texts are equal, or one contains the
    other ("oat milk" inside "oat milk cortado"). Sorted by `review_id` then
    `mention_offset`, plain index.
    """
    if entities.empty:
        return pd.DataFrame(columns=MENTION_COLUMNS)
    placed = pd.merge(entities, sentences, on="review_id")
    inside = (placed["entity_offset"] >= placed["sentence_offset"]) & (
        placed["entity_offset"] < placed["sentence_offset"] + placed["sentence_length"]
    )
    placed = placed[inside]

    covered = {}
    for row in opinions.to_dict("records"):
        key = (row["review_id"], row["sentence_id"])
        covered.setdefault(key, []).append(row["mention"].lower())

    def is_covered(row) -> bool:
        """True when an opinion target in the same sentence names this entity."""
        text = row["entity_text"].lower()
        targets = covered.get((row["review_id"], row["sentence_id"]), [])
        return any(text in target or target in text for target in targets)

    keep = placed[~placed.apply(is_covered, axis=1)] if len(placed) else placed
    rows = pd.DataFrame({
        "review_id": keep["review_id"],
        "sentence_id": keep["sentence_id"],
        "mention": keep["entity_text"],
        "mention_sentiment": keep["sentence_sentiment"],
        "assessments": "",
        "mention_offset": keep["entity_offset"],
        "source": "entity",
    }, columns=MENTION_COLUMNS)
    return rows.sort_values(["review_id", "mention_offset"]).reset_index(drop=True)


def load_aliases() -> dict:
    """data/aliases.csv as a dict: alternate spelling -> the item's one name."""
    aliases = pd.read_csv(ALIASES_FILE)
    return dict(zip(aliases["alias"].str.lower(), aliases["item"]))


def item_name(mention: str, aliases: dict) -> str:
    """The canonical name for a mention: lowercased, then looked up in the aliases.

        item_name("Parking downtown", aliases)  ->  "parking"
        item_name("Mocha Latte", aliases)        ->  "mocha latte"   # not an alias
    """
    return aliases.get(mention.strip().lower(), mention.strip().lower())


def build_mentions(reviews: pd.DataFrame, sentences: pd.DataFrame,
                   opinions: pd.DataFrame, entity_rows: pd.DataFrame,
                   subjects: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """Everything together: one row per mention, with its subject, sentence and review.

    Columns, in this order: `review_id`, `date`, `location`, `stars`, `item`,
    `mention`, `subject`, `mention_sentiment`, `assessments`, `sentence_text`,
    `source`, `subject_confidence`.

    `opinions` and `entity_rows` stacked (`pd.concat`), then merged with
    `subjects` on (`review_id`, `mention`), `sentences` on (`review_id`,
    `sentence_id`) and `reviews` on `review_id`. `item` is `item_name` of the
    mention. Rows whose subject is `"other"` are dropped, and so is a second
    mention of the same item in the same sentence ("oat milk" and "oat milk
    cortado" both becoming "oat milk cortado"). Sorted by `review_id` then
    `mention_offset`, plain index.
    """
    
    master = pd.concat([opinions, entity_rows], ignore_index=True)
    
    master = pd.merge(master, subjects, on=["review_id", "mention"], how="left")
    master = master[master["subject"] != "other"]
    
    master = pd.merge(master, sentences, on=["review_id", "sentence_id"], how="left")
    master = pd.merge(master, reviews, on="review_id", how="left")
    
    master["item"] = master["mention"].apply(lambda x: item_name(x, aliases))
    
    master = (
        master.sort_values(["review_id", "mention_offset"])
              .drop_duplicates(subset=["review_id", "sentence_id", "item"], keep="first")
              .reset_index(drop=True)
    )
    
    return master[["review_id", "date", "location", "stars", "item", "mention",
            "subject", "mention_sentiment", "assessments", "sentence_text",
            "source", "subject_confidence"]]


def analyze_reviews(reviews_file: str) -> pd.DataFrame:
    """The whole pipeline for one reviews file: raw CSV in, `mentions` out.

    Reads the CSV, runs `sentiment_step` and `entities_step` (cached with
    `cache_path(reviews_file, "sentiment")` and `"entities"`), builds the three
    tables, `entity_mentions`, then `subjects_step` on the combined mentions
    (cached with `cache_path(reviews_file, "subjects", "csv")`), and finally
    `build_mentions` with `load_aliases()`. Call the functions; do not redo them.
    """
    sentiment_cache = cache_path(reviews_file, "sentiment")
    entities_cache = cache_path(reviews_file, "entities")
    subjects_cache = cache_path(reviews_file, "subjects", "csv")
    
    reviews = pd.read_csv(reviews_file)
    
    sentiments = sentiment_step(reviews, cache_file=sentiment_cache)
    
    entities = entities_table(entities_step(reviews, cache_file=entities_cache))
    sentences = sentences_table(sentiments)
    opinions = opinions_table(sentiments)
    
    entity_table = entity_mentions(entities, sentences, opinions)
    
    all_mentions = pd.concat([opinions, entity_table], ignore_index=True)
    subjects = subjects_step(reviews, all_mentions, cache_file=subjects_cache)
    
    return build_mentions(reviews, sentences, opinions, entity_table, subjects, load_aliases())
    


def summarize_items(mentions: pd.DataFrame) -> pd.DataFrame:
    """Most-discussed items: one row per (subject, item).

    Columns: `subject`, `item`, `mentions` (how many rows), `positive`,
    `negative` (how many of those rows have that `mention_sentiment`), and `net`
    (positive minus negative). Sorted by `mentions` descending, then `item`;
    plain index.

    How to build it — a pivot table, as in Assignment 05, then one engineered
    column:

    - `pd.pivot_table` with `index=["subject", "item"]`,
      `columns="mention_sentiment"`, `aggfunc="size"` and `fill_value=0`: one
      row per item, one column of counts per sentiment.
    - `mentions` is each row's total across those columns (`.sum(axis=1)`).
    - `positive` and `negative` are two of the columns — but a sentiment that
      never appears gets no column (filter the page to one location and there
      may be no complaints at all). `table.get("negative", 0)` gives the column,
      or 0 when there isn't one.
    - **Engineer** `net` = `positive` - `negative`: one number that says whether
      an item is loved or a problem.
    """
    results = mentions.pivot_table(
        index=["subject", "item"],
        columns="mention_sentiment",
        aggfunc="size",
        fill_value=0
    )
    
    results["mentions"] = results.sum(axis=1)
    results["positive"] = results.get("positive", 0)
    results["negative"] = results.get("negative", 0)
    results["net"] = results["positive"] - results["negative"]
    
    results = results.sort_values(["mentions", "item"], ascending=[False, True]).reset_index()
    
    return results[["subject", "item", "mentions", "positive", "negative", "net"]]


# --- The Q&A box --------------------------------------------------------------------


def question_scope(question: str, entity_document: dict, mentions: pd.DataFrame,
                   aliases: dict) -> dict:
    """What a question is about: its items, locations and subjects. *(Provided.)*

    Read this one rather than write it — it is how the Q&A box decides what to
    look up, and README Reference #8 walks through why it works this way.

    Returns a dict with three lists, any of which may be empty:

        {"items": [...], "locations": [...], "subjects": [...]}

    `entity_document` is `get_entities(question)["results"]["documents"][0]`.

    - **items**: its Person and Product entities, put through `item_name` —
      kept even when no review mentions them, so "chai latte" is in the scope
      of a question about it — then every item in `mentions["item"]` whose
      lowercased name appears in the lowercased question (entity recognition
      does not know that "wifi" or "parking" are things). No repeats,
      entities first.
    - **locations**: each location in `LOCATION_WORDS` with any of its words in
      the lowercased question, in `LOCATION_WORDS` order.
    - **subjects**: the same, with `SUBJECT_WORDS`.

        question_scope("Do people like the mocha latte?", ...)
            -> {"items": ["mocha latte"], "locations": [], "subjects": []}
        question_scope("How is the staff at Armory Square?", ...)
            -> {"items": [], "locations": ["Armory Square"],
                "subjects": ["customer service"]}
    """
    lowered = question.lower()
    candidates = [
        item_name(entity["text"], aliases)
        for entity in entity_document["entities"]
        if entity["category"] in MENTION_CATEGORIES
    ]
    known = sorted(mentions["item"].unique())
    candidates += [item for item in known if item.lower() in lowered]
    items = []
    for item in candidates:
        if item not in items:
            items.append(item)

    def named(word_lists: dict) -> list:
        """The keys with at least one of their words in the question."""
        return [name for name, words in word_lists.items()
                if any(word in lowered for word in words)]

    return {"items": items, "locations": named(LOCATION_WORDS),
            "subjects": named(SUBJECT_WORDS)}


SCOPE_COLUMNS = {"items": "item", "subjects": "subject", "locations": "location"}


def evidence_for(mentions: pd.DataFrame, scope: dict) -> pd.DataFrame:
    """The mentions inside a question's scope — the rows the answer is built from.

    Items and subjects say *what* the question is about, so a row matches when
    its item is one of the items **or** its subject is one of the subjects.
    Locations say *where*, and narrow that further. An empty list does not
    filter, so a question that names nothing gets every mention.
    """
    selected = mentions
    if scope["items"] or scope["subjects"]:
        what = (mentions["item"].isin(scope["items"])
                | mentions["subject"].isin(scope["subjects"]))
        selected = selected[what]
    if scope["locations"]:
        selected = selected[selected["location"].isin(scope["locations"])]
    return selected.reset_index(drop=True)


def describe_scope(scope: dict) -> str:
    """A scope in words, for the page: "mocha latte · Westcott", or "all reviews"."""
    parts = [", ".join(scope[key]) for key in SCOPE_COLUMNS if scope[key]]
    return " · ".join(parts) if parts else "all reviews"


def build_messages(question: str, evidence: pd.DataFrame, limit: int = 20) -> list:
    """The conversation for `ask_llm`: a system prompt, then question and evidence.

    Returns a list of two dicts:

        {"role": "system", "content": SYSTEM_PROMPT}
        {"role": "user",   "content": <the text below>}

    The user content is, line by line:

        Question: <question>
        Facts: <n> mentions — <p> positive, <q> negative, <r> neutral or mixed.
        Items, most mentioned first:
        - <item> (<subject>): <m> mentions, <pos> positive, <neg> negative, net <net>
        - ...
        Review sentences:
        - (<location>, <date>, <item>, <mention_sentiment>) <sentence_text>
        - ...

    One item line per row of `summarize_items(evidence)`, in its order: `<m>`,
    `<pos>`, `<neg>` and `<net>` are its `mentions`, `positive`, `negative` and
    `net` columns, with net written with its sign (`f"{net:+d}"` -> "+3", "-2",
    "+0"). One sentence line per evidence row, at most `limit` of them. The counts cover
    every row, so "what is the most popular product?" is answered from the
    whole table, not from whichever sentences fit. pandas does the counting; the
    language model only writes the answer. Language models are bad at counting
    and good at prose — give each the job it is good at.
    """
    items = summarize_items(evidence)

    n = len(evidence)
    pos = (evidence["mention_sentiment"] == "positive").sum()
    neg = (evidence["mention_sentiment"] == "negative").sum()
    other = n - pos - neg

    lines = [
        f"Question: {question}",
        f"Facts: {n} mentions — {pos} positive, {neg} negative, {other} neutral or mixed.",
        "Items, most mentioned first:",
    ]
    for row in items.itertuples(index=False):
        lines.append(f"- {row.item} ({row.subject}): {row.mentions} mentions, "
                     f"{row.positive} positive, {row.negative} negative, net {row.net:+d}")

    lines.append("Review sentences:")
    for row in evidence.head(limit).itertuples(index=False):
        lines.append(f"- ({row.location}, {row.date}, {row.item}, "
                     f"{row.mention_sentiment}) {row.sentence_text}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]

def answer_question(question: str, mentions: pd.DataFrame, aliases: dict):
    """What a question is about and the evidence for it: (scope, evidence)."""
    document = get_entities(question)["results"]["documents"][0]
    scope = question_scope(question, document, mentions, aliases)
    return scope, evidence_for(mentions, scope)
