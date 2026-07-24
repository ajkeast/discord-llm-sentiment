# Understanding Discord Chat with AI

This project scores a private Discord archive with a language model — tagging each message for mood, emotion, sarcasm, and toxicity. Below is the simple version of *why* that matters and *what changed* to make it practical.

## What is sentiment analysis?

**Sentiment analysis** asks: *how does this text feel?*

A message might be positive ("this update rules"), negative ("this is broken"), neutral ("meeting at 5"), or mixed ("love the idea, hate the bugs"). Richer versions also look for emotions (joy, anger, amusement), sarcasm, and whether the tone is aimed at a person, a group, or a topic.

It is a way to turn a mountain of chat into something you can measure and compare over time — not just skim.

## What is NLP?

**NLP** stands for **Natural Language Processing**: teaching computers to work with human language — the messy, slangy, context-heavy kind people actually write.

Sentiment analysis is one NLP task. Others include translation, summarization, spam detection, and search. The shared hard problem is that words alone are not enough; meaning depends on context, tone, and community norms.

## How was NLP done before LLMs?

Before large language models, most systems were narrower and more brittle:

1. **Rules and dictionaries** — lists of "good" and "bad" words. Fast, but easy to fool (`sick` can mean cool or ill).
2. **Classical machine learning** — train a model on hand-labeled examples using word counts or similar features. Better than word lists, but still shallow about context.
3. **Specialized deep learning** — neural nets trained for one job (e.g. "positive vs negative"). Stronger, but you still needed lots of labeled data, and each new nuance (sarcasm, Discord slang, in-jokes) often meant more labeling and retraining.

None of these were great at short chat where "lol" can mean genuine laughter, awkwardness, or a soft dismissal — depending on the messages before it.

## Why is it different after LLMs?

**LLMs** (large language models) are trained on huge amounts of text to predict language. Along the way they pick up patterns of tone, irony, and everyday conversation.

That changes the workflow:

- You can describe what you want in plain English ("score polarity, emotions, sarcasm, toxicity") instead of building a custom model from scratch.
- The model can use **nearby messages as context**, which matters a lot in Discord threads and reply chains.
- One general model can handle many labels at once — mood, emotion, toxicity — without a separate pipeline for each.

It is not magic, and it can still be wrong. But for chat-shaped text, it is a much closer fit than older keyword or single-purpose classifiers.

## Why run this on a private Discord dataset?

Public sentiment tools and demos are usually trained on reviews, tweets, or news — not your server's inside jokes, channel culture, or shorthand.

Running analysis on **your own archive** lets you:

- See how the mood of a community shifts over months or years
- Spot channels or stretches of time that turn unusually toxic or negative
- Study patterns that only make sense with local context (sarcasm, banter, recurring topics)
- Keep the data private — messages stay in your database; you are not shipping the corpus to a third-party "vibes dashboard"

In short: the interesting signal is in *this* community's language, not a generic model of the internet.

## What this repo does

It takes Discord messages (exported from MySQL), gives each one a little preceding channel context, and asks Gemini to return a structured score per message — polarity, emotions, sarcasm, toxicity, and a short rationale. Results can be stored back in the database and summarized in a report.

For bot / nightly integration details, see [`docs/DISCORD_BOT_NIGHTLY.md`](docs/DISCORD_BOT_NIGHTLY.md).
