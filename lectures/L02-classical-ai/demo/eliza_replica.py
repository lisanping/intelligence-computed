"""ELIZA-style pattern-matching chatbot replica.

Demonstrates the GOFAI era's "stimulus-response" approach to conversation:
no learning, no embedding — just hand-crafted pattern → response rules.
Created for L02 (Classical AI). Compatible with Joseph Weizenbaum 1966 design.

Run:
  python eliza_replica.py
  > Hello
  > I am sad
  > my mother always said no
  quit
"""
from __future__ import annotations
import re
import random
from dataclasses import dataclass
from typing import List, Tuple


# Pronoun reflection: "I" -> "you", "my" -> "your", etc.
REFLECT = {
    "i": "you", "me": "you", "my": "your", "mine": "yours",
    "am": "are", "i'm": "you're", "i've": "you've", "i'll": "you'll",
    "you": "I", "your": "my", "yours": "mine", "yourself": "myself",
    "myself": "yourself", "are": "am",
}


def reflect(fragment: str) -> str:
    return " ".join(REFLECT.get(w.lower(), w) for w in fragment.split())


# (priority, regex, [response templates])
RULES: List[Tuple[int, re.Pattern, List[str]]] = [
    (10, re.compile(r"\b(hello|hi|hey)\b", re.I),
     ["Hello. How are you feeling today?",
      "Hi there. What's on your mind?"]),
    (10, re.compile(r"\bI need (.+)", re.I),
     ["Why do you need {0}?",
      "Would it really help you to get {0}?",
      "Are you sure you need {0}?"]),
    (8, re.compile(r"\bI am (.+)", re.I),
     ["Why do you think you are {0}?",
      "How long have you been {0}?",
      "Do you enjoy being {0}?"]),
    (8, re.compile(r"\bI'?m (.+)", re.I),
     ["How does being {0} make you feel?",
      "Why do you say you are {0}?"]),
    (7, re.compile(r"\bI feel (.+)", re.I),
     ["Tell me more about feeling {0}.",
      "Do you often feel {0}?"]),
    (6, re.compile(r"\bbecause\b", re.I),
     ["Is that the real reason?",
      "What other reasons might there be?"]),
    (6, re.compile(r"\bsorry\b", re.I),
     ["No need to apologize.",
      "Apologies are not necessary here."]),
    (5, re.compile(r"\b(mother|father|sister|brother|family)\b", re.I),
     ["Tell me more about your {0}.",
      "How do you feel about your {0}?"]),
    (5, re.compile(r"\byes\b", re.I),
     ["I see.",
      "You seem certain."]),
    (5, re.compile(r"\bno\b", re.I),
     ["Why not?",
      "Are you sure?"]),
    (4, re.compile(r"\b(can|could|would) you (.+)", re.I),
     ["Why do you ask if I {1}?",
      "Would it matter to you if I {1}?"]),
    (3, re.compile(r"\?$"),
     ["Why do you ask that?",
      "What do you think?"]),
    (1, re.compile(r".*"),
     ["Please tell me more.",
      "Go on.",
      "I see. And what does that suggest to you?"]),
]


def respond(text: str) -> str:
    text = text.strip()
    if not text:
        return "..."
    # priority-sorted rules
    for _, pattern, templates in sorted(RULES, key=lambda r: -r[0]):
        m = pattern.search(text)
        if m:
            template = random.choice(templates)
            args = [reflect(g) for g in m.groups()]
            try:
                return template.format(*args)
            except IndexError:
                return template
    return "Tell me more."


def main() -> None:
    print("=" * 60)
    print("ELIZA replica  (Weizenbaum 1966 — symbolic AI era)")
    print("=" * 60)
    print("This is a 90-line pattern-matching bot.")
    print("It has no understanding, no memory, no learning.")
    print("Type 'quit' to exit.\n")
    print("ELIZA: Hello. How are you feeling today?")
    while True:
        try:
            user = input("YOU>  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nELIZA: Goodbye.")
            break
        if user.lower() in {"quit", "exit", "bye"}:
            print("ELIZA: Goodbye.")
            break
        print(f"ELIZA: {respond(user)}")


if __name__ == "__main__":
    main()
