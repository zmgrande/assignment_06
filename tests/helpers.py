"""Shared helpers for the tests: recorded API responses, and driving the Streamlit page."""

import json
import os

import pandas as pd
from streamlit.testing.v1 import AppTest

# tests/ lives directly under the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts: str) -> str:
    """Build an absolute path from the repository root, whatever the working directory."""
    return os.path.join(REPO_ROOT, *parts)


def fixture(name: str):
    """A recorded API response from tests/fixtures/ (JSON parsed, CSV as a DataFrame)."""
    path = repo_path("tests", "fixtures", name)
    if name.endswith(".csv"):
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """A deep copy to compare against after a function has run."""
    return frame.copy(deep=True)


def assert_unmodified(handed: pd.DataFrame, before: pd.DataFrame, step: str) -> None:
    """The frame the function was handed still looks exactly like it did before the call."""
    assert list(handed.columns) == list(before.columns) and handed.astype(str).equals(before.astype(str)), (
        f"{step} modified the DataFrame it was given — work on a copy, or build a new frame."
    )


# --- Streamlit ----------------------------------------------------------------------

TEXT_ELEMENTS = (
    "title", "header", "subheader", "markdown", "text", "caption",
    "info", "success", "warning", "error",
)


def load_app(script: str) -> AppTest:
    """Load a Streamlit script from the repository and run it once, as opening it would."""
    app = AppTest.from_file(repo_path(script), default_timeout=60)
    app.run()
    return app


def page_text(app: AppTest) -> str:
    """Everything readable on the page, joined into one string."""
    parts = []
    for element_type in TEXT_ELEMENTS:
        parts.extend(str(element.value) for element in app.get(element_type))
    for metric in app.metric:
        parts.append(f"{metric.label}: {metric.value}")
    return "\n".join(parts)


def metrics(app: AppTest) -> dict[str, str]:
    """The page's st.metric cards as {label: value}; values are the displayed strings."""
    return {metric.label: str(metric.value) for metric in app.metric}


def widget(elements, key: str, kind: str):
    """The widget with `key=`, or the only widget of that kind, or a clear error."""
    for element in elements:
        if getattr(element, "key", None) == key:
            return element
    if len(elements) == 1:
        return elements[0]
    raise AssertionError(f"expected one st.{kind} (or one with key={key!r}), found {len(elements)}")


def no_exception(app: AppTest, script: str) -> None:
    assert not app.exception, f"{script} raised: {app.exception[0].value}"
