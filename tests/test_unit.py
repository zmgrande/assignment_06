"""Unit tests — the four API functions, and the one-line name cleaner.

Two kinds of test for each API function:

- **The request** (no network): `requests.post` is swapped for a stand-in that
  records what your function sent and hands back a canned response. These check
  the URL, the X-API-KEY header, form `data=` versus `json=`, the query string,
  and that `raise_for_status()` is called — the parts that are your code.
- **The real thing** (needs your key in `.env`): one live call each, with
  inputs the portal has answered before, so they come back from its cache and
  do not count against your 150 a day.

If every live test fails with "No API key found", that is the message to read:
create the `.env` file (README Reference #1).
"""

import pytest
import requests

from reviews import ask_llm, classify_mentions, get_entities, get_sentiment, item_name
from reviews.settings import BASE_URL, LLM_MODEL, LLM_REASONING_EFFORT, subject_questions

T2 = "The mocha latte was overpriced and lukewarm. The wifi kept dropping, so I couldn't get any work done."


class FakeResponse:
    """Just enough of a requests.Response for the functions under test."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.raised = False

    def raise_for_status(self):
        self.raised = True
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self.payload


class FakePost:
    """Stands in for requests.post: records each call, answers with `payload`/`status`."""

    def __init__(self):
        self.calls = []
        self.payload = {"ok": True}
        self.status = 200

    def __call__(self, url, **kwargs):
        response = FakeResponse(self.payload, self.status)
        self.calls.append({"url": url, "response": response, **kwargs})
        return response


@pytest.fixture
def fake_post(monkeypatch):
    """Replace requests.post for one test."""
    monkeypatch.setenv("CENT_API_KEY", "test-key-123")
    fake = FakePost()
    monkeypatch.setattr(requests, "post", fake)
    return fake


def only_call(fake, name):
    calls = fake.calls
    assert len(calls) == 1, f"{name} should make exactly one requests.post call, made {len(calls)}"
    call = calls[0]
    assert call["response"].raised, f"{name} should call response.raise_for_status()"
    assert (call.get("headers") or {}).get("X-API-KEY") == "test-key-123", (
        f"{name} should send the key from get_api_key() in an X-API-KEY header"
    )
    return call


# --- The requests ----------------------------------------------------------------------


def test_get_sentiment_request(fake_post):
    result = get_sentiment("I love it.")
    call = only_call(fake_post, "get_sentiment")
    assert call["url"] == f"{BASE_URL}/api/azure/sentiment", f"wrong URL: {call['url']}"
    assert call.get("data") == {"text": "I love it."}, "send the text as form data: data={'text': text}"
    assert result == {"ok": True}, "return response.json()"


def test_get_entities_request(fake_post):
    result = get_entities("Patty was great.")
    call = only_call(fake_post, "get_entities")
    assert call["url"] == f"{BASE_URL}/api/azure/entityrecognition", f"wrong URL: {call['url']}"
    assert call.get("data") == {"text": "Patty was great."}, "send the text as form data"
    assert result == {"ok": True}, "return response.json()"


def test_classify_mentions_request(fake_post):
    result = classify_mentions("The latte was cold.", ["latte"])
    call = only_call(fake_post, "classify_mentions")
    assert call["url"] == f"{BASE_URL}/api/ollama/systemone", f"wrong URL: {call['url']}"
    assert call.get("json") == {"state": {"review": "The latte was cold."},
                                "questions": subject_questions(["latte"])}, (
        "send json={'state': {'review': review_text}, 'questions': subject_questions(mentions)}"
    )
    assert result == {"ok": True}, "return response.json()"


def test_ask_llm_request(fake_post):
    fake_post.payload = {"choices": [{"message": {"role": "assistant", "content": "Mostly yes."}}]}
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}]
    result = ask_llm(messages)
    call = only_call(fake_post, "ask_llm")
    assert call["url"] == f"{BASE_URL}/api/azure/foundry/chat/completions", f"wrong URL: {call['url']}"
    assert call.get("json") == messages, "the body is the conversation itself: json=messages"
    params = call.get("params") or {}
    assert params == {"model": LLM_MODEL, "reasoning_effort": LLM_REASONING_EFFORT}, (
        "pass params={'model': LLM_MODEL, 'reasoning_effort': LLM_REASONING_EFFORT}"
    )
    assert result == "Mostly yes.", "return only the reply text: ['choices'][0]['message']['content']"


@pytest.mark.parametrize(
    "function, args",
    [
        (get_sentiment, ("text",)),
        (get_entities, ("text",)),
        (classify_mentions, ("text", ["thing"])),
        (ask_llm, ([{"role": "user", "content": "hi"}],)),
    ],
    ids=["get_sentiment", "get_entities", "classify_mentions", "ask_llm"],
)
def test_errors_are_raised(fake_post, function, args):
    fake_post.status = 401
    with pytest.raises(requests.exceptions.HTTPError):
        function(*args)


# --- The real thing ----------------------------------------------------------------------


def test_get_sentiment_live():
    document = get_sentiment(T2)["results"]["documents"][0]
    assert document["sentiment"] == "negative"
    first = document["sentences"][0]
    assert [t["text"] for t in first["targets"]] == ["mocha latte"], (
        "opinion mining should find the mocha latte as the target of the first sentence"
    )
    assert [a["text"] for a in first["assessments"]] == ["overpriced", "lukewarm"]


def test_get_entities_live():
    entities = get_entities("How is Patty doing?")["results"]["documents"][0]["entities"]
    assert ("Patty", "Person") in [(e["text"], e["category"]) for e in entities]


def test_classify_mentions_live():
    answers = classify_mentions(T2, ["mocha latte", "wifi"])["answers"]
    assert answers["mocha latte"]["choice"] == "product"
    assert answers["wifi"]["choice"] == "atmosphere"
    assert 0 <= answers["wifi"]["confidence"] <= 1


# --- item_name --------------------------------------------------------------------------

ALIASES = {"brew": "cold brew", "parking downtown": "parking", "otto": "Otto"}


@pytest.mark.parametrize(
    "mention, expected",
    [
        ("mocha latte", "mocha latte"),
        ("Mocha Latte", "mocha latte"),          # lowercased
        (" wifi ", "wifi"),                       # stripped
        ("brew", "cold brew"),                    # an alias
        ("Parking downtown", "parking"),          # an alias, after lowercasing
        ("Otto", "Otto"),                         # a name keeps its capital through the alias
    ],
)
def test_item_name(mention, expected):
    assert item_name(mention, ALIASES) == expected, f"item_name({mention!r}) should be {expected!r}"
