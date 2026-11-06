"""Pipeline & app tests — JSON into tables, the cached API steps, the whole pipeline, the page.

Three kinds of check:

- **Transforms on recorded responses** (no network). `tests/fixtures/` holds the
  portal's real answers for the five test reviews, so `sentences_table`,
  `opinions_table`, `entities_table`, `build_mentions`,
  `summarize_items` and `build_messages` are checked against
  exact, known output — the same output you can check by hand in the README.
- **The API steps, live**, on the five test reviews into a temporary cache
  folder — including that a second run reads the cache instead of calling.
- **The page**, driven with Streamlit's AppTest on the five test reviews. The Q&A
  test swaps `ask_llm` for a stand-in, so it checks the conversation your page
  *sends*, not what the model happens to say back.
"""

import os

import pandas as pd
import pytest

import reviews
from helpers import fixture, load_app, metrics, no_exception, page_text, snapshot, assert_unmodified, widget
from reviews import (
    TEST_REVIEWS_FILE,
    analyze_reviews,
    build_mentions,
    build_messages,
    entities_step,
    entities_table,
    entity_mentions,
    evidence_for,
    load_aliases,
    opinions_table,
    sentences_table,
    sentiment_step,
    subjects_step,
    summarize_items,
)
from reviews.settings import SYSTEM_PROMPT

APP = "code/review_explorer.py"
SENTIMENT = "reviews_test_sentiment.json"
ENTITIES = "reviews_test_entities.json"
SUBJECTS = "reviews_test_subjects.csv"

OPINION_COLUMNS = ["review_id", "sentence_id", "mention", "mention_sentiment",
                   "assessments", "mention_offset", "source"]
MENTION_COLUMNS = ["review_id", "date", "location", "stars", "item", "mention", "subject",
                   "mention_sentiment", "assessments", "sentence_text", "source",
                   "subject_confidence"]


def review_rows():
    return pd.read_csv(TEST_REVIEWS_FILE)


def tables():
    sentences = sentences_table(fixture(SENTIMENT))
    opinions = opinions_table(fixture(SENTIMENT))
    entities = entities_table(fixture(ENTITIES))
    return sentences, opinions, entities


def expected_mentions():
    sentences, opinions, entities = tables()
    entity_rows = entity_mentions(entities, sentences, opinions)
    return build_mentions(review_rows(), sentences, opinions, entity_rows,
                          fixture(SUBJECTS), load_aliases())


# --- JSON into tables -----------------------------------------------------------------


def test_sentences_table():
    out = sentences_table(fixture(SENTIMENT))
    assert list(out.columns) == ["review_id", "sentence_id", "sentence_text", "sentence_sentiment",
                                 "sentence_offset", "sentence_length"], f"columns are {list(out.columns)}"
    assert len(out) == 11, f"the five test reviews have 11 sentences, got {len(out)}"
    assert list(out["sentence_id"]) == [0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1], (
        "sentence_id counts 0, 1, 2, ... within each review (groupby + cumcount)"
    )
    assert out.loc[2, "sentence_text"] == "The mocha latte was overpriced and lukewarm.", (
        "sentence_text comes from the sentence's 'text', with the trailing space stripped"
    )
    assert list(out["sentence_sentiment"].iloc[4:6]) == ["negative", "positive"]


def test_opinions_table():
    out = opinions_table(fixture(SENTIMENT))
    assert list(out.columns) == OPINION_COLUMNS, f"columns are {list(out.columns)}"
    assert list(out["mention"]) == ["almond croissant", "mocha latte", "wifi", "seating", "music",
                                    "Patty", "brew", "blueberry scone", "mocha latte", "murals"]
    by_mention = out.set_index(["review_id", "mention"])
    assert by_mention.loc[("T2", "mocha latte"), "assessments"] == "overpriced, lukewarm", (
        "assessments are the texts the target's relations point to, joined with ', '"
    )
    assert by_mention.loc[("T3", "seating"), "mention_sentiment"] == "positive", (
        "a target has its own sentiment — the seating is positive inside a negative sentence"
    )
    assert by_mention.loc[("T3", "seating"), "assessments"] == "love, comfortable"
    assert by_mention.loc[("T3", "music"), "assessments"] == "too loud", (
        "two targets share T3's first sentence — each gets only the assessments its "
        "relations point to, not every assessment in the sentence"
    )
    assert by_mention.loc[("T3", "music"), "sentence_id"] == 0
    assert (out["source"] == "opinion").all()


def test_opinions_table_with_no_targets():
    document = {"review_id": "X1", "sentences": [{"text": "Hello.", "sentiment": "neutral",
                                                  "targets": [], "assessments": []}]}
    out = opinions_table([document])
    assert list(out.columns) == OPINION_COLUMNS and out.empty, (
        "no targets still gives an empty table with the seven columns"
    )


def test_entities_table():
    out = entities_table(fixture(ENTITIES))
    assert list(out.columns) == ["review_id", "entity_text", "entity_category", "entity_offset"]
    assert set(out["entity_category"]) <= {"Person", "Product"}, "keep only Person and Product"
    people = out[out["entity_category"] == "Person"]["entity_text"].tolist()
    assert people == ["Pete", "Patty", "Tally", "Otto"]
    assert list(out.index) == list(range(len(out))), "plain index after filtering"


# --- Step 3 transforms ----------------------------------------------------------------


def test_build_mentions():
    out = expected_mentions()
    assert list(out.columns) == MENTION_COLUMNS, f"columns are {list(out.columns)}"
    assert len(out) == 14, f"the test reviews have 14 mentions, got {len(out)}"
    assert list(out["item"]) == ["Pete", "oat milk cortado", "almond croissant", "mocha latte",
                                 "wifi", "seating", "music", "Patty", "cold brew",
                                 "blueberry scone", "Tally", "mocha latte", "murals", "Otto"], (
        "items in review then offset order, through the aliases ('brew' -> 'cold brew')"
    )
    tally = out[out["item"] == "Tally"].iloc[0]
    assert (tally["subject"], tally["mention_sentiment"], tally["location"]) == (
        "customer service", "negative", "Armory Square")
    assert tally["sentence_text"] == "Tally at the register forgot my order."


def test_summarize_items():
    mentions = expected_mentions()
    before = snapshot(mentions)
    out = summarize_items(mentions)
    assert list(out.columns) == ["subject", "item", "mentions", "positive", "negative", "net"]
    assert out.iloc[0].tolist() == ["product", "mocha latte", 2, 1, 1, 0], (
        "most-mentioned first: the mocha latte, 2 mentions, 1 positive, 1 negative, net 0"
    )
    assert len(out) == 13 and int(out["mentions"].sum()) == 14
    assert int(out["net"].sum()) == 9 - 5
    assert_unmodified(mentions, before, "summarize_items")


def test_summarize_items_without_complaints():
    """Filter to one sentiment and that pivot table has no "negative" column at all."""
    mentions = expected_mentions()
    praise = mentions[mentions["mention_sentiment"] == "positive"]
    out = summarize_items(praise)
    assert list(out.columns) == ["subject", "item", "mentions", "positive", "negative", "net"], (
        "still all six columns when no mention is negative — table.get('negative', 0)"
    )
    assert out["negative"].tolist() == [0] * len(out)
    assert out["net"].tolist() == out["positive"].tolist() == out["mentions"].tolist()


def scope(items=(), locations=(), subjects=()):
    return {"items": list(items), "locations": list(locations), "subjects": list(subjects)}


MOCHA_MESSAGE = "\n".join([
    "Question: Do people like the mocha latte?",
    "Facts: 2 mentions — 1 positive, 1 negative, 0 neutral or mixed.",
    "Items, most mentioned first:",
    "- mocha latte (product): 2 mentions, 1 positive, 1 negative, net +0",
    "Review sentences:",
    "- (Westcott, 2026-09-19, mocha latte, negative) The mocha latte was overpriced and lukewarm.",
    "- (Westcott, 2026-10-10, mocha latte, positive) Best mocha latte in Syracuse!",
])


def test_build_messages():
    evidence = evidence_for(expected_mentions(), scope(items=["mocha latte"]))
    messages = build_messages("Do people like the mocha latte?", evidence)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == MOCHA_MESSAGE


def test_build_messages_counts_every_item():
    evidence = evidence_for(expected_mentions(), scope(subjects=["product"]))
    content = build_messages("What is the most popular product?", evidence)[1]["content"]
    lines = content.split("\n")
    assert lines[1] == "Facts: 6 mentions — 4 positive, 2 negative, 0 neutral or mixed."
    start = lines.index("Items, most mentioned first:")
    assert lines[start + 1:start + 6] == [
        "- mocha latte (product): 2 mentions, 1 positive, 1 negative, net +0",
        "- almond croissant (product): 1 mentions, 1 positive, 0 negative, net +1",
        "- blueberry scone (product): 1 mentions, 0 positive, 1 negative, net -1",
        "- cold brew (product): 1 mentions, 1 positive, 0 negative, net +1",
        "- oat milk cortado (product): 1 mentions, 1 positive, 0 negative, net +1",
    ], "one line per row of summarize_items(evidence), in its order"
    assert lines[start + 6] == "Review sentences:"


def test_build_messages_limits_the_evidence():
    evidence = pd.concat([evidence_for(expected_mentions(), scope(items=["mocha latte"]))] * 15,
                         ignore_index=True)
    content = build_messages("Q?", evidence, limit=20)[1]["content"]
    sentences = content.split("Review sentences:\n")[1]
    assert sentences.count("- (") == 20, "at most `limit` review sentences"
    assert "Facts: 30 mentions — 15 positive, 15 negative" in content, "the facts count every row"
    assert "- mocha latte (product): 30 mentions, 15 positive, 15 negative, net +0" in content, (
        "so do the item counts"
    )


# --- The API steps, live ------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path


def test_sentiment_step_calls_then_caches(tmp_cache, monkeypatch):
    cache_file = os.path.join(tmp_cache, "sentiment.json")
    documents = sentiment_step(review_rows(), cache_file)
    assert [d["review_id"] for d in documents] == ["T1", "T2", "T3", "T4", "T5"], (
        "one document per review, each with its review_id added"
    )
    assert os.path.exists(cache_file), "sentiment_step should write its cache file"
    assert opinions_table(documents)["mention"].tolist() == opinions_table(fixture(SENTIMENT))["mention"].tolist()

    def refuse(_text):
        raise AssertionError("sentiment_step called the API although its cache file exists")

    monkeypatch.setattr(reviews.pipeline, "get_sentiment", refuse)
    assert sentiment_step(review_rows(), cache_file) == documents, "second run reads the cache"


def test_entities_step_calls_then_caches(tmp_cache, monkeypatch):
    cache_file = os.path.join(tmp_cache, "entities.json")
    documents = entities_step(review_rows(), cache_file)
    assert [d["review_id"] for d in documents] == ["T1", "T2", "T3", "T4", "T5"]
    assert os.path.exists(cache_file), "entities_step should write its cache file"

    def refuse(_text):
        raise AssertionError("entities_step called the API although its cache file exists")

    monkeypatch.setattr(reviews.pipeline, "get_entities", refuse)
    assert entities_step(review_rows(), cache_file) == documents, "second run reads the cache"


def test_subjects_step_live(tmp_cache):
    sentences, opinions, entities = tables()
    combined = pd.concat([opinions, entity_mentions(entities, sentences, opinions)], ignore_index=True)
    cache_file = os.path.join(tmp_cache, "subjects.csv")
    out = subjects_step(review_rows(), combined, cache_file)
    assert list(out.columns) == ["review_id", "mention", "subject", "subject_confidence"]
    assert os.path.exists(cache_file), "subjects_step should write its cache file"
    got = dict(zip(zip(out["review_id"], out["mention"]), out["subject"]))
    expected = fixture(SUBJECTS)
    assert got == dict(zip(zip(expected["review_id"], expected["mention"]), expected["subject"]))


def test_analyze_reviews_end_to_end(tmp_cache, monkeypatch):
    monkeypatch.setattr(reviews.settings, "CACHE_DIR", str(tmp_cache))
    out = analyze_reviews(TEST_REVIEWS_FILE)
    want = expected_mentions()
    assert list(out.columns) == MENTION_COLUMNS
    assert out[["review_id", "item", "subject", "mention_sentiment"]].values.tolist() == \
        want[["review_id", "item", "subject", "mention_sentiment"]].values.tolist()
    assert len(os.listdir(tmp_cache)) == 3, "three cache files: sentiment, entities, subjects"


# --- The page --------------------------------------------------------------------------------


@pytest.fixture
def test_set(monkeypatch):
    monkeypatch.setenv("REVIEWS_FILE", TEST_REVIEWS_FILE)


def test_app_shows_the_reviews(test_set):
    app = load_app(APP)
    no_exception(app, APP)
    assert "Salt City Coffee — What Customers Are Saying" in page_text(app)
    shown = metrics(app)
    assert shown == {"Reviews": "5", "Mentions": "14", "Positive": "9", "Negative": "5"}, shown
    labels = [tab.label for tab in app.tabs]
    assert labels == ["Most praised", "Most complained about", "What people said"], (
        f"three st.tabs, in this order: {labels}"
    )
    assert len(app.dataframe) >= 3, "a table in each tab"


def test_app_filters(test_set):
    app = load_app(APP)
    widget(app.selectbox, "location", "selectbox").set_value("Westcott")
    app.run()
    assert metrics(app) == {"Reviews": "2", "Mentions": "5", "Positive": "3", "Negative": "2"}
    widget(app.selectbox, "subject", "selectbox").set_value("product")
    app.run()
    no_exception(app, APP)
    assert metrics(app)["Mentions"] == "2", "Westcott products: two mocha lattes"


def ask_with_fake_model(monkeypatch, question):
    """Ask a question on the page with `ask_llm` swapped out; return (app, messages sent)."""
    sent = []

    def fake_ask_llm(messages):
        sent.append(messages)
        return "FAKE ANSWER: opinions are split."

    monkeypatch.setattr(reviews, "ask_llm", fake_ask_llm)
    app = load_app(APP)
    widget(app.text_input, "question", "text_input").set_value(question)
    widget(app.button, "ask", "button").click()
    app.run()
    no_exception(app, APP)
    return app, sent


def test_app_asks_the_model(test_set, monkeypatch):
    app, sent = ask_with_fake_model(monkeypatch, "Do people like the mocha latte?")
    assert len(sent) == 1, "clicking Ask should call ask_llm once"
    evidence = evidence_for(expected_mentions(), scope(items=["mocha latte"]))
    assert sent[0] == build_messages("Do people like the mocha latte?", evidence), (
        "the page should send build_messages(question, evidence) for the mocha latte"
    )
    assert "FAKE ANSWER" in page_text(app), "show the model's reply on the page"


def test_app_answers_from_every_mention(test_set, monkeypatch):
    question = "What do customers complain about most?"
    _app, sent = ask_with_fake_model(monkeypatch, question)
    assert len(sent) == 1, "a question that names nothing is still answered"
    assert sent[0] == build_messages(question, expected_mentions()), (
        "with every mention as the evidence"
    )


def test_app_does_not_ask_about_nothing(test_set, monkeypatch):
    app, sent = ask_with_fake_model(monkeypatch, "Do people like the chai latte?")
    assert not sent, "no review mentions the chai latte: do not spend a call on the model"
    assert app.warning, "say so with an st.warning"
