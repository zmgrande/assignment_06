"""
apicalls.py — Step 1: one function per API endpoint.

Every function here does the same four things, the algorithm from lesson 4-1:

    1. build the URL from BASE_URL and the endpoint's path
    2. send the request with your key in the X-API-KEY header
    3. response.raise_for_status() — turn a 4xx/5xx into an exception, right here
    4. return response.json()

`raise_for_status()` is the line people leave out, and it is the one that saves
you an hour: without it, a bad key or a typo in the path comes back as a
perfectly normal-looking dictionary of error text, and your code fails three
steps later with a KeyError that points nowhere near the real problem.

These functions do not loop, do not build DataFrames, and do not cache. They make
one call and hand back what the API said. Everything else is Step 2's job.

This step is walked through line by line. The next two steps give you less.
"""

import requests

from .settings import (
    BASE_URL,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    get_api_key,
    subject_questions,
)


def get_sentiment(text: str) -> dict:
    """Azure sentiment with opinion mining for one piece of text.

    POST {BASE_URL}/api/azure/sentiment with the text as form data. The useful part
    of the response is `["results"]["documents"][0]`, which has the overall
    `sentiment`, and `sentences` — each with its own `sentiment`, `text`, `offset`
    and `length`, plus `targets` (the things the writer has an opinion about) and
    `assessments` (the opinion words).

    How to build it:

    - `url = f"{BASE_URL}/api/azure/sentiment"`
    - `headers = {"X-API-KEY": get_api_key()}` — the key comes from your `.env`
      file, never from a string in this file.
    - `response = requests.post(url, headers=headers, data={"text": text})` —
      `data=` sends a form, which is what this endpoint expects (lesson 4-2).
    - `response.raise_for_status()`, then `return response.json()`.
    """
    url = f"{BASE_URL}/api/azure/sentiment"
    headers = {"X-API-KEY": get_api_key()}
    response = requests.post(url, headers=headers, data={"text": text})
    response.raise_for_status()
    return response.json()


def get_entities(text: str) -> dict:
    """Azure named-entity recognition for one piece of text.

    POST {BASE_URL}/api/azure/entityrecognition with the text as form data. The
    entities are a list at `["results"]["documents"][0]["entities"]`, each with
    `text`, `category` ("Person", "Product", "Location", ...), `offset` and
    `confidenceScore`.

    How to build it:

    - Exactly `get_sentiment` with a different path. Write it again rather than
      copy-paste-and-forget: the path is the only line that changes.
    """
    url = f"{BASE_URL}/api/azure/entityrecognition"
    headers = {"X-API-KEY": get_api_key()}
    response = requests.post(url, headers=headers, data={"text": text})
    response.raise_for_status()
    return response.json()


def classify_mentions(review_text: str, mentions: list) -> dict:
    """Ask the decision model what each mention in a review is about.

    POST {BASE_URL}/api/ollama/systemone with a JSON body of two parts:

        {"state": {"review": review_text},
         "questions": subject_questions(mentions)}

    `subject_questions` (in settings.py) writes one `choice` question per mention,
    so the answers come back keyed by the mention itself:

        response["answers"]["mocha latte"]["choice"]       ->  "product"
        response["answers"]["mocha latte"]["confidence"]   ->  0.99

    How to build it:

    - Same URL-and-headers start as the others, with path `/api/ollama/systemone`.
    - Build the body as a Python dict, exactly as above.
    - This endpoint takes **JSON**, not a form: `requests.post(url,
      headers=headers, json=body)`. `json=` turns the dict into JSON and sets
      the Content-Type for you. (`data=` would send a form and get a 4xx back —
      which `raise_for_status()` would tell you about immediately.)
    - The decision model is slow the first time a question is asked (seconds per
      mention) and instant after that, because the portal caches every answer.
    - `raise_for_status()`, `return response.json()`.
    """
    url = f"{BASE_URL}/api/ollama/systemone"
    headers = {"X-API-KEY": get_api_key()}
    body = {"state": {"review": review_text}, "questions": subject_questions(mentions)}
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()


def ask_llm(messages: list) -> str:
    """Send a conversation to the language model; return its reply as a string.

    POST {BASE_URL}/api/azure/foundry/chat/completions. The body is the
    conversation itself — a JSON **list** of `{"role": ..., "content": ...}`
    dicts. The query string says which model to use and how hard it should
    think: `model=LLM_MODEL` and `reasoning_effort=LLM_REASONING_EFFORT` (both
    in settings.py). There is no `temperature`: reasoning models do not have
    one, and the portal ignores it. Instead the portal caches each answer for a
    day, so the same conversation gets the same answer — and a repeat does not
    count against your daily limit.

    The reply text is at `response.json()["choices"][0]["message"]["content"]`
    (lesson 4-3). Return just that string.

    How to build it:

    - URL and headers as before, path `/api/azure/foundry/chat/completions`.
    - `params = {"model": LLM_MODEL, "reasoning_effort": LLM_REASONING_EFFORT}`
    - `requests.post(url, headers=headers, json=messages, params=params)`
    - `raise_for_status()`, then dig the content out of the JSON and return it.
    """
    url = f"{BASE_URL}/api/azure/foundry/chat/completions"
    headers = {"X-API-KEY": get_api_key()}
    params = {"model": LLM_MODEL, "reasoning_effort": LLM_REASONING_EFFORT}
    response = requests.post(url, headers=headers, json=messages, params=params)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
