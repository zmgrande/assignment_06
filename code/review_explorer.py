"""
review_explorer.py — what Salt City Coffee's customers are saying, at a glance.

The owner reads every review on the shop's website, but thirty reviews across
three locations is more than anyone can hold in their head. This page shows
what customers praise and complain about — by location and by subject — and
answers plain-English questions like "Do people like the mocha latte?" or "What
is the most popular product?" from the reviews themselves, with the evidence
underneath.

The page is assembly again: `analyze_reviews` does all the API work (from the
cache after the first run), and the Q&A box is three calls you have already
written — find the question's scope and evidence, build the messages, ask the
model.

Run it:  Run and Debug -> "Streamlit Run: Current File"
Test it: pytest tests/test_pipeline.py -k app
"""

import os

import streamlit as st

from reviews import (
    REVIEWS_FILE,
    analyze_reviews,
    answer_question,
    ask_llm,
    build_messages,
    describe_scope,
    load_aliases,
    summarize_items,
)

# --- The page: the algorithm, step by step ------------------------------------------
#
# Every function this page needs is already in the reviews package (imported above),
# and every widget is one you have used before. Write the code for each step below
# its comment. The tests (pytest tests/test_pipeline.py -k app) check the title, the
# widget keys, the metric labels and values, the tab labels, and the conversation
# the page sends to the model. README Step 8 has the same details as a table.
#
# Run the page as you go (Run and Debug -> "Streamlit Run: Current File") and save
# the file to see each step appear.
#
# 1. WHICH REVIEWS
#    reviews_file <- the REVIEWS_FILE environment variable if it is set,
#                    otherwise REVIEWS_FILE from the package.
#    (os.environ.get(name, default) does exactly this. The tests set the variable
#    to the 5 test reviews; when you run the page you see all 30.)

reviews_file = os.environ.get("REVIEWS_FILE", REVIEWS_FILE)

# 2. HEADING
#    st.title: Salt City Coffee — What Customers Are Saying
#    st.markdown: one sentence telling the owner what the page does, e.g. "Every
#    customer review, broken into the things people talk about. Filter by location
#    or subject, or ask a question at the bottom of the page."

st.title("Salt City Coffee — What Customers Are Saying")
st.markdown("Every customer review, broken into the things people talk about. Filter by location or subject, or ask a question at the bottom of the page.")

# 3. THE DATA
#    mentions <- analyze_reviews(reviews_file)
#    (One row per thing a customer had an opinion about. The first run fills
#    cache/; after that this takes a second.)

mentions = analyze_reviews(reviews_file)

# 4. TWO FILTERS, SIDE BY SIDE
#    a. Make two columns with st.columns(2).
#    b. In the first column, an st.selectbox labeled "Location" with key "location".
#       Its options: "All" followed by the sorted unique values of
#       mentions["location"].
#    c. In the second column, an st.selectbox labeled "Subject" with key "subject".
#       Its options: "All" followed by the sorted unique values of
#       mentions["subject"].
#    d. selected <- mentions
#       If the location is not "All", keep only the rows of selected whose
#       location equals it. Then, if the subject is not "All", keep only the rows
#       whose subject equals it. (Two separate ifs, so both filters can apply.)

location, subject = st.columns(2)

locations = ["All"] + sorted(mentions["location"].unique()).tolist()
subjects = ["All"] + sorted(mentions["subject"].unique()).tolist()

with location:
    location_filter = st.selectbox(label="Location", options=locations, key="location")
with subject:
    subject_filter = st.selectbox(label="Subject", options=subjects, key="subject")

selected = mentions.copy()
if location_filter != "All":
    selected = selected[selected["location"] == location_filter]
if subject_filter != "All":
    selected = selected[selected["subject"] == subject_filter]

# 5. FOUR METRICS, SIDE BY SIDE — all computed from selected, not mentions
#    a. Make four columns with st.columns(4).
#    b. st.metric "Reviews":  the number of unique review_id values
#    c. st.metric "Mentions": the number of rows
#    d. st.metric "Positive": how many rows have mention_sentiment == "positive"
#    e. st.metric "Negative": how many rows have mention_sentiment == "negative"
#    (A boolean Series .sum() counts the Trues. Wrap it in int() so the metric
#    shows 9, not np.int64(9).)

reviews_col, mentions_col, positive_col, negative_col = st.columns(4)
with reviews_col:
    st.metric("Reviews", value=selected["review_id"].nunique())
with mentions_col:
    st.metric("Mentions", value=selected.shape[0])
with positive_col:
    st.metric("Positive", value=int((selected["mention_sentiment"] == "positive").sum()))
with negative_col:
    st.metric("Negative", value=int((selected["mention_sentiment"] == "negative").sum()))

# 6. THREE TABLES IN THREE TABS
#    a. summary <- summarize_items(selected)
#    b. Make three tabs with st.tabs, labeled exactly, in this order:
#       "Most praised", "Most complained about", "What people said"
#       (st.tabs returns one object per tab; call .dataframe on each one.)
#    c. Most praised: the rows of summary with net > 0, sorted by net from
#       highest to lowest, first 5 rows. Show with hide_index=True.
#    d. Most complained about: the rows of summary with net < 0, sorted by net
#       from lowest to highest (most negative first), first 5 rows.
#       hide_index=True.
#    e. What people said: selected, with only these columns, in this order:
#       date, location, item, subject, mention_sentiment, assessments,
#       sentence_text. hide_index=True.

summary = summarize_items(selected)
praise, complaints, vibe = st.tabs(["Most praised", "Most complained about", "What people said"])
with praise:
    st.dataframe(summary[summary["net"] > 0].sort_values(by="net", ascending=False).head(5), hide_index=True)
with complaints:
    st.dataframe(summary[summary["net"] < 0].sort_values(by="net", ascending=True).head(5), hide_index=True)
with vibe:
    st.dataframe(selected[["date", "location", "item", "subject", "mention_sentiment", "assessments", "sentence_text"]], hide_index=True)

# 7. THE QUESTION BOX
#    a. st.subheader: Ask the reviews
#    b. question <- st.text_input labeled "Your question" with key "question".
#       Give it a placeholder that shows the owner what to ask, e.g.
#       "What is the most popular product?"
#    c. An st.button labeled "Ask" with key "ask".

st.subheader("Ask the reviews")
user_question = st.text_input(label="Your question", key="question", placeholder="What is the most popular product?")
ask_button = st.button(label="Ask", key="ask")

# 8. WHEN ASK IS CLICKED (and the question is not blank — question.strip())
#    a. scope, evidence <- answer_question(question, mentions, load_aliases())
#       (Provided. scope says what the question is about; evidence is the rows
#       of mentions that answer it. Use mentions here, not selected: the
#       question, not the filters, decides what to look up.)
#    b. If evidence is empty:
#          st.warning saying no reviews mention describe_scope(scope), with a
#          hint to try a menu item, a staff member, or part of the shop.
#          Do NOT call the model — there is nothing for it to answer from.
#    c. Otherwise:
#          i.   messages <- build_messages(question, evidence)
#          ii.  answer <- ask_llm(messages), inside `with st.spinner(...):` so the
#               owner sees "Asking the model..." for the few seconds it takes.
#          iii. st.markdown(answer)
#          iv.  st.caption: "Based on <number of evidence rows> mentions —
#               <describe_scope(scope)>:"
#          v.   st.dataframe of evidence, with only these columns: date,
#               location, item, mention_sentiment, sentence_text.
#               hide_index=True.

if user_question.strip() and ask_button:
    scope, evidence = answer_question(user_question, mentions, load_aliases())
    if evidence.empty:
        st.warning(f"No reviews mention {describe_scope(scope)}. Try a menu item, a staff member, or part of the shop.")
    else:
        messages = build_messages(user_question, evidence)
        with st.spinner("Asking the model..."):
            answer = ask_llm(messages)
        st.markdown(answer)
        st.caption(f"Based on {len(evidence)} mentions — {describe_scope(scope)}:")
        st.dataframe(evidence[["date", "location", "item", "mention_sentiment", "sentence_text"]], hide_index=True)

# What the page does NOT do: call the API directly, group or pivot, or build the
# prompt itself. That logic lives in the package; the page only assembles it.
