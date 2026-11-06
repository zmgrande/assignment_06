"""Code structure tests — read the source and check how the pipeline is built.

The other tests check what your functions *return*. These check *how*: that only
`apicalls.py` touches the network, that every call raises on an HTTP error, that
your key never appears in your code, that the extract steps cache, that the
tables come out of `json_normalize`, and that the page assembles the pipeline
instead of redoing it.

Everything here reads the **parsed** source (`ast`), so a technique mentioned in
a comment does not count. When a test fails, the message names the file and the
thing it was looking for.
"""

import ast
import re

from helpers import repo_path

APICALLS = "code/reviews/apicalls.py"
PIPELINE = "code/reviews/pipeline.py"
ASPECTS = "code/reviews/aspects.py"
APP = "code/review_explorer.py"
STUDENT_FILES = (APICALLS, PIPELINE, ASPECTS, APP)

API_FUNCTIONS = {
    "get_sentiment": ("/api/azure/sentiment", "data"),
    "get_entities": ("/api/azure/entityrecognition", "data"),
    "classify_mentions": ("/api/ollama/systemone", "json"),
    "ask_llm": ("/api/azure/foundry/chat/completions", "json"),
}


def source(script: str) -> str:
    with open(repo_path(script), encoding="utf-8") as fh:
        return fh.read()


def parse(script: str) -> ast.Module:
    return ast.parse(source(script), filename=script)


def calls(tree: ast.AST) -> list:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def called_names(tree: ast.AST) -> set:
    return {call_name(node) for node in calls(tree)}


def calls_named(tree: ast.AST, name: str) -> list:
    return [node for node in calls(tree) if call_name(node) == name]


def keyword_names(node: ast.Call) -> set:
    return {keyword.arg for keyword in node.keywords}


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name}() not found")


def imported_modules(tree: ast.Module) -> set:
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def strings(tree: ast.AST) -> list:
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def names_used(tree: ast.AST) -> set:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


# --- Step 1 -------------------------------------------------------------------------------


def test_every_api_function_follows_the_algorithm():
    """URL from BASE_URL, key from get_api_key(), requests.post, raise_for_status, .json()."""
    tree = parse(APICALLS)
    for name, (path, body) in API_FUNCTIONS.items():
        fn = function(tree, name)
        called = called_names(fn)
        posts = calls_named(fn, "post")
        assert posts, f"{APICALLS}: {name} should send its request with requests.post(...)"
        assert any(body in keyword_names(node) for node in posts), (
            f"{APICALLS}: {name} should send its body with {body}= — "
            + ("the Azure endpoints take a form" if body == "data" else "this endpoint takes JSON")
        )
        assert any("headers" in keyword_names(node) for node in posts), (
            f"{APICALLS}: {name} should pass headers= with the X-API-KEY"
        )
        assert "raise_for_status" in called, f"{APICALLS}: {name} should call response.raise_for_status()"
        assert "json" in called, f"{APICALLS}: {name} should read the response with .json()"
        assert "get_api_key" in called, f"{APICALLS}: {name} should get the key from get_api_key()"
        assert "BASE_URL" in names_used(fn), f"{APICALLS}: {name} should build its URL from BASE_URL"
        assert any(path in text for text in strings(fn)), f"{APICALLS}: {name} should call {path}"


def test_no_api_key_in_the_code():
    """Your key lives in .env — never in a file that gets committed."""
    assert "get_api_key" in called_names(parse(APICALLS)), (
        f"{APICALLS} should read the key with get_api_key() — the API functions are not written yet"
    )
    for script in STUDENT_FILES:
        for text in strings(parse(script)):
            assert not re.fullmatch(r"[A-Za-z0-9_\-]{20,}", text), (
                f"{script} contains a string that looks like an API key ({text[:4]}...). "
                "Keys go in .env, read with get_api_key() — never in code you commit."
            )


def test_only_apicalls_touches_the_network():
    """The pipeline and the page go through apicalls.py; nothing else imports requests."""
    assert {"get_sentiment", "get_entities"} <= called_names(parse(PIPELINE)), (
        f"{PIPELINE} should make its API calls through get_sentiment() and get_entities()"
    )
    for script in (PIPELINE, ASPECTS, APP):
        assert "requests" not in imported_modules(parse(script)), (
            f"{script} imports requests — call the functions in apicalls.py instead, so there is "
            "exactly one place that knows about URLs, keys and HTTP errors"
        )


# --- Step 2 -------------------------------------------------------------------------------


def test_extract_steps_cache():
    """Each API step checks for its cache file first, and writes it after calling."""
    tree = parse(PIPELINE)
    for name, writer in (("sentiment_step", "dump"), ("entities_step", "dump"), ("subjects_step", "to_csv")):
        called = called_names(function(tree, name))
        assert "exists" in called, f"{PIPELINE}: {name} should check os.path.exists(cache_file) first"
        assert writer in called, f"{PIPELINE}: {name} should save what it got with {writer}(...)"
    for name, api in (("sentiment_step", "get_sentiment"), ("entities_step", "get_entities"),
                      ("subjects_step", "classify_mentions")):
        assert api in called_names(function(tree, name)), f"{PIPELINE}: {name} should call {api}()"


def test_tables_use_json_normalize():
    """sentences_table and entities_table flatten the JSON with json_normalize."""
    tree = parse(PIPELINE)
    for name in ("sentences_table", "entities_table"):
        normalizes = calls_named(function(tree, name), "json_normalize")
        assert normalizes, f"{PIPELINE}: {name} should flatten the documents with pd.json_normalize(...)"
        assert any({"record_path", "meta"} <= keyword_names(node) for node in normalizes), (
            f"{PIPELINE}: {name} should use record_path= (the list to flatten) and meta= "
            "(the parent fields to copy down) — lesson 4-1"
        )
    assert "cumcount" in called_names(function(tree, "sentences_table")), (
        f"{PIPELINE}: sentence_id is a groupby('review_id').cumcount()"
    )


# --- Step 3 -------------------------------------------------------------------------------


def test_aspects_combine_and_summarize_with_pandas():
    """build_mentions merges and concats; summarize_items is a pivot table plus an engineered net."""
    tree = parse(ASPECTS)
    built = called_names(function(tree, "build_mentions"))
    assert "merge" in built and "concat" in built, (
        f"{ASPECTS}: build_mentions should stack with pd.concat and join with pd.merge"
    )
    summary = function(tree, "summarize_items")
    assert "pivot_table" in called_names(summary), (
        f"{ASPECTS}: summarize_items should count with pd.pivot_table(...)"
    )
    assert "net" in {node.value for node in ast.walk(summary)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)}, (
        f"{ASPECTS}: summarize_items should engineer the net column: summary['net'] = ..."
    )
    analyze = called_names(function(tree, "analyze_reviews"))
    for step in ("sentiment_step", "entities_step", "sentences_table", "opinions_table",
                 "entities_table", "entity_mentions", "subjects_step", "build_mentions"):
        assert step in analyze, f"{ASPECTS}: analyze_reviews should call {step}()"


# --- The page --------------------------------------------------------------------------------


def test_app_assembles_the_pipeline():
    """The page builds on analyze_reviews and the Q&A pieces, with the required widgets."""
    tree = parse(APP)
    called = called_names(tree)
    assert "reviews" in imported_modules(tree), f"{APP} should import from the reviews package"
    for name in ("analyze_reviews", "summarize_items", "answer_question", "build_messages", "ask_llm"):
        assert name in called, f"{APP} should call {name}()"
    for widget in ("selectbox", "metric", "tabs", "dataframe", "text_input", "button",
                   "warning"):
        assert widget in called, f"{APP} should use st.{widget}(...)"
    assert not called & {"get_sentiment", "classify_mentions", "groupby", "pivot_table"}, (
        f"{APP}: the page assembles — the API calls and the grouping belong in the package"
    )
    assert "print" not in called and "input" not in called, f"{APP}: use widgets, not the console"
