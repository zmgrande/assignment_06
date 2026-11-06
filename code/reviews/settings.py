"""
settings.py — where things live, and how to reach the API.  *(Provided for you.)*

Everything the rest of the package needs to *find*: the portal's address, your
API key, the data files, the cache folder, and the exact wording of the
questions sent to the decision model. You do not change this file, but you do
need to read it.

Your API key
------------
The CENT iSchool IoT Portal (https://portal.app.cent-su.org) needs your key in an
`X-API-KEY` header on every request. Your key is on the portal's Home page. It is
a password: it never goes in your code, and never in a commit. Put it in a file
named `.env` at the root of this repository, one line:

    CENT_API_KEY=your-key-here

`.env` is listed in `.gitignore`, so git will not offer to commit it. In GitHub
Codespaces you can instead add a Codespaces secret named `CENT_API_KEY`. Either
way `get_api_key()` finds it: it uses python-dotenv's `load_dotenv()` to read
`.env` into the environment, the standard way Python projects keep secrets out
of their code.

Your limit is 150 requests a day, but a request the portal has seen before is
answered from its cache and does not count. Re-running the pipeline on the same
reviews is free.
"""

import os

from dotenv import load_dotenv

# The portal. An environment variable can point the code somewhere else; you will
# not need to.
BASE_URL = os.environ.get("CENT_API_BASE", "https://portal.app.cent-su.org")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

REVIEWS_FILE = os.path.join(DATA_DIR, "reviews.csv")         # the 30 reviews
TEST_REVIEWS_FILE = os.path.join(DATA_DIR, "reviews_test.csv")  # 5 to check by hand
ALIASES_FILE = os.path.join(DATA_DIR, "aliases.csv")


def get_api_key() -> str:
    """Your portal API key, from the CENT_API_KEY environment variable or `.env`.

    `load_dotenv` (from the python-dotenv package) reads `.env` and puts each
    NAME=value line into the environment variables, where `os.environ` can see
    it. It never overwrites a variable that is already set — so a Codespaces
    secret, or the grader's own key, wins over the file.
    """
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    key = os.environ.get("CENT_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "No API key found. Create a file named .env at the root of the repository "
            "containing one line, CENT_API_KEY=your-key-here (your key is on the Home "
            "page of https://portal.app.cent-su.org). See README Reference #1."
        )
    return key


def cache_path(reviews_file: str, step: str, ext: str = "json") -> str:
    """Where a pipeline step caches its output for one reviews file.

        cache_path(".../reviews_test.csv", "sentiment")
        -> ".../cache/reviews_test_sentiment.json"
    """
    stem = os.path.splitext(os.path.basename(reviews_file))[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{stem}_{step}.{ext}")


# --- The decision model's questions -------------------------------------------------
# Every student sends exactly these words, so the portal answers the second person
# to ask from its cache. Change a word and you pay for a fresh (slow) answer.

SUBJECTS = {
    "customer service": (
        "A person who works there (barista, cashier, cook), "
        "or the speed and friendliness of the service"
    ),
    "atmosphere": (
        "The space itself: seating, music, noise, wifi, decor, cleanliness, parking"
    ),
    "product": "Food or drink the shop sells: its taste, temperature, freshness, price",
    "other": "None of the above",
}


def subject_questions(mentions: list) -> dict:
    """One `choice` question per mention, in the form /api/ollama/systemone expects."""
    return {
        mention: {
            "type": "choice",
            "instructions": (
                f"In this review, which category does '{mention}' belong to?"
            ),
            "criteria": SUBJECTS,
        }
        for mention in mentions
    }


# Named-entity categories worth treating as a mention: people and products.
MENTION_CATEGORIES = ("Person", "Product")

# --- The Q&A box -------------------------------------------------------------------
# Words that tell `question_scope` a question is about a location or a subject.
# Entity recognition cannot do this job: it tags "Westcott" as a Location but
# "Armory Square" as an Organization, and "product" as nothing at all. A short
# list of the words people actually use is more reliable. Each word is matched
# anywhere in the lowercased question, so "barista" also catches "baristas".

LOCATION_WORDS = {
    "Armory Square": ["armory", "downtown"],
    "Westcott": ["westcott", "wescott"],          # a common misspelling
    "University Hill": ["university hill", "the hill", "campus"],
}

SUBJECT_WORDS = {
    "customer service": ["service", "staff", "barista", "employee", "cashier"],
    "atmosphere": ["atmosphere", "vibe", "ambiance", "space", "decor"],
    "product": ["product", "food", "drink", "menu", "pastry", "pastries"],
}

# The language model behind the Q&A box: GPT-6 Luna on Azure AI Foundry, through the
# portal. It is a *reasoning* model — it thinks before it answers — and "low" effort is
# plenty for two or three sentences from facts you hand it.
LLM_MODEL = "gpt-6-luna"
LLM_REASONING_EFFORT = "low"

# The Q&A box's instructions to the language model.
SYSTEM_PROMPT = (
    "You answer questions about Salt City Coffee from its customer reviews. "
    "Use only the facts provided: the counts cover every matching mention, and "
    "the review sentences are examples. If the facts do not answer the question, "
    "say so. Answer in two or three sentences. When asked what is most popular or "
    "best, say which item is mentioned most and which is liked most if they "
    "differ, and mention locations when opinions differ between them."
)
