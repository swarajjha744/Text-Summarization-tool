"""
Text Summarization Tool
Internship Task 1

Two approaches:
  1. Extractive NLP  — picks the best sentences from your text (no internet needed)
  2. Abstractive AI  — uses Claude to write a fresh summary in natural language

Author : Swaraj Jha
"""

import re
import string
import heapq
from collections import defaultdict


# ---------------------------------------------------------------------------
# Words that carry no real meaning and should be ignored during analysis.
# These are called "stopwords" in NLP.
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "his", "her",
    "their", "with", "from", "not", "also", "can", "about", "which",
    "who", "what", "when", "where", "how", "than", "then", "just",
    "more", "some", "into", "over", "after", "before", "if", "very",
    "quite", "such", "any", "all", "both", "each", "few", "most",
    "other", "no", "only", "same", "too", "now", "said", "one", "two",
    "however", "therefore", "thus", "hence", "although", "though",
    "since", "while", "still", "already", "yet", "even", "often",
}


# ---------------------------------------------------------------------------
# Step 1: Clean the text
#
# We strip punctuation, lowercase everything, and remove stopwords.
# This leaves only the words that actually carry meaning.
# ---------------------------------------------------------------------------

def clean_words(text: str) -> list:
    words = text.lower().split()
    result = []
    for word in words:
        word = word.strip(string.punctuation)
        if word and len(word) > 2 and word not in STOPWORDS:
            result.append(word)
    return result


def get_sentences(text: str) -> list:
    """Split a block of text into individual sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Keep only sentences with enough words to be meaningful
    return [s.strip() for s in raw if len(s.split()) > 4]


# ---------------------------------------------------------------------------
# Step 2: Score each word by how often it appears (TF-IDF style)
#
# Words that appear more often are considered more important to the document.
# We normalize scores so the most frequent word always gets a score of 1.0.
# ---------------------------------------------------------------------------

def word_importance(text: str) -> dict:
    words = clean_words(text)
    counts = defaultdict(int)

    for word in words:
        counts[word] += 1

    highest = max(counts.values()) if counts else 1
    return {word: count / highest for word, count in counts.items()}


# ---------------------------------------------------------------------------
# Step 3: Score each sentence
#
# A sentence's score = average importance of all its meaningful words.
# Sentences packed with important words bubble to the top.
# ---------------------------------------------------------------------------

def score_sentences(sentences: list, importance: dict) -> dict:
    scores = {}
    for sentence in sentences:
        words = clean_words(sentence)
        if not words:
            continue
        total = sum(importance.get(w, 0) for w in words)
        scores[sentence] = round(total / len(words), 5)
    return scores


# ---------------------------------------------------------------------------
# EXTRACTIVE SUMMARIZER
#
# This is the main NLP approach. It:
#   1. Splits the text into sentences
#   2. Scores each sentence based on word importance
#   3. Picks the top N sentences
#   4. Returns them in the original reading order
#
# Nothing is rewritten — we are "extracting" the best parts.
# ---------------------------------------------------------------------------

def summarize_extractive(text: str, num_sentences: int = 3) -> str:
    sentences = get_sentences(text)

    if len(sentences) <= num_sentences:
        return text.strip()

    importance = word_importance(text)
    scores = score_sentences(sentences, importance)

    # Find the top N sentences by score
    top = heapq.nlargest(num_sentences, scores, key=scores.get)

    # Put them back in the order they appeared in the original text
    ordered = [s for s in sentences if s in top]
    return " ".join(ordered)


def summarize_bullets(text: str, num_points: int = 5) -> str:
    """Same as extractive, but formatted as a bullet list."""
    sentences = get_sentences(text)
    importance = word_importance(text)
    scores = score_sentences(sentences, importance)

    top = heapq.nlargest(num_points, scores, key=scores.get)
    return "\n".join(f"  • {s.strip()}" for s in top)


# ---------------------------------------------------------------------------
# ABSTRACTIVE SUMMARIZER  (requires Claude API)
#
# Instead of picking existing sentences, this asks an AI to read the text
# and write a brand-new summary — the way a human writer would.
#
# Setup:
#   pip install anthropic
#   export ANTHROPIC_API_KEY="sk-ant-..."
# ---------------------------------------------------------------------------

def summarize_abstractive(text: str, style: str = "brief") -> str:
    try:
        import anthropic
    except ImportError:
        return (
            "Abstractive mode needs the Anthropic library.\n"
            "Run:  pip install anthropic\n"
            "Then: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    instructions = {
        "concise":  "Summarize the following text in exactly one sentence.",
        "brief":    "Write a clear, natural-sounding paragraph summarizing the text. Aim for 3–4 sentences.",
        "detailed": "Summarize with 3 key points. Label them Point 1, Point 2, Point 3.",
        "bullets":  "Summarize as a bullet list of 5 key takeaways. Start each with •.",
        "eli5":     "Summarize in simple everyday language anyone can understand. No jargon.",
    }

    prompt = instructions.get(style, instructions["brief"])
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nText:\n\n{text}"
        }]
    )

    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Utility: print a clean result to the terminal
# ---------------------------------------------------------------------------

def show_result(label: str, original: str, summary: str):
    orig_words = len(original.split())
    summ_words = len(summary.split())
    reduction  = round((1 - summ_words / orig_words) * 100, 1) if orig_words else 0

    print()
    print("=" * 62)
    print(f"  {label}")
    print("=" * 62)
    print()
    print(f"Original text  ({orig_words} words):")
    print("-" * 62)
    preview = original[:380] + ("…" if len(original) > 380 else "")
    print(preview)
    print()
    print(f"Summary  ({summ_words} words  —  {reduction}% shorter):")
    print("-" * 62)
    print(summary)
    print()


# ---------------------------------------------------------------------------
# Sample article used for the demo
# ---------------------------------------------------------------------------

SAMPLE = """
Artificial intelligence is rapidly transforming industries across the globe.
From healthcare to finance, education to agriculture, AI systems are being
deployed to improve efficiency, reduce costs, and discover insights that were
previously impossible to find manually. Machine learning algorithms can now
diagnose diseases from medical images with accuracy that rivals trained physicians.
In the financial sector, AI is used for fraud detection, algorithmic trading,
and personalized financial advice. Natural language processing, a branch of AI,
enables computers to understand, interpret, and generate human language, powering
applications like virtual assistants, chatbots, and automated translation services.

However, the rise of AI also brings significant challenges. Concerns about job
displacement are widespread, as automation threatens to make many traditional roles
obsolete. Issues of bias in AI systems have emerged, where algorithms trained on
historical data can perpetuate and amplify existing social inequalities. Privacy
concerns are paramount, as AI systems often require vast amounts of personal data
to function effectively. Governments and regulatory bodies worldwide are working
to establish frameworks that encourage innovation while protecting citizens.

Despite these challenges, experts remain optimistic about AI's potential to address
humanity's greatest problems — from climate modeling and drug discovery to sustainable
energy optimization. The key lies in responsible development: creating AI systems
that are transparent, fair, and aligned with human values. Education and reskilling
programs will be critical to help workers adapt to the changing job landscape.
"""


# ---------------------------------------------------------------------------
# Run the demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    text = SAMPLE.strip()

    print()
    print("=" * 62)
    print("  Text Summarization Tool  |  Internship Task 1")
    print("  NLP Approach: Extractive (TF-IDF) + Abstractive (AI)")
    print("=" * 62)

    # Demo 1: brief summary (3 best sentences)
    show_result(
        "Extractive — brief summary (3 sentences)",
        text,
        summarize_extractive(text, num_sentences=3)
    )

    # Demo 2: one-sentence version
    show_result(
        "Extractive — one sentence",
        text,
        summarize_extractive(text, num_sentences=1)
    )

    # Demo 3: bullet points
    show_result(
        "Extractive — bullet points",
        text,
        summarize_bullets(text, num_points=4)
    )

    # Demo 4: abstractive (AI) — uncomment once API key is ready
    print("-" * 62)
    print("  AI Abstractive mode is ready but commented out below.")
    print("  Install: pip install anthropic")
    print("  Key:     export ANTHROPIC_API_KEY='sk-ant-...'")
    print("-" * 62)
    print()

    # Once your API key is set, uncomment these:
    # show_result(
    #     "Abstractive — AI generated (brief)",
    #     text,
    #     summarize_abstractive(text, style="brief")
    # )
    # show_result(
    #     "Abstractive — AI generated (bullet points)",
    #     text,
    #     summarize_abstractive(text, style="bullets")
    # )

    print("Done.")
    print()
