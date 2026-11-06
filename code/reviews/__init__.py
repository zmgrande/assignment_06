"""
reviews — what Salt City Coffee's customers are saying.  *(Provided for you.)*

    from reviews import analyze_reviews, summarize_items, REVIEWS_FILE

Three steps in three modules, after the provided settings:

    settings.py   the portal, your key, file locations, the model's questions  (given)
    apicalls.py   Step 1 — one function per API endpoint                       (you)
    pipeline.py   Step 2 — cached API calls, JSON into tables                  (you)
    aspects.py    Step 3 — mentions, summaries, and the Q&A box's pieces       (you)

Same as Assignments 04 and 05: every name below must exist in its module, even
before you have written its body, or `import reviews` fails and every test goes
red at once. That is why the stubs keep every function signature.
"""

from .apicalls import ask_llm, classify_mentions, get_entities, get_sentiment
from .aspects import (
    analyze_reviews,
    answer_question,
    build_mentions,
    build_messages,
    describe_scope,
    entity_mentions,
    evidence_for,
    item_name,
    load_aliases,
    question_scope,
    summarize_items,
)
from .pipeline import (
    entities_step,
    entities_table,
    opinions_table,
    sentences_table,
    sentiment_step,
    subjects_step,
)
from .settings import REVIEWS_FILE, TEST_REVIEWS_FILE, cache_path, get_api_key

__all__ = [
    "get_sentiment", "get_entities", "classify_mentions", "ask_llm",
    "sentiment_step", "entities_step", "sentences_table", "opinions_table",
    "entities_table", "subjects_step",
    "entity_mentions", "load_aliases", "item_name", "build_mentions", "analyze_reviews",
    "summarize_items", "question_scope", "evidence_for", "describe_scope",
    "build_messages", "answer_question",
    "REVIEWS_FILE", "TEST_REVIEWS_FILE", "cache_path", "get_api_key",
]
