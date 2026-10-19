"""Access to a local Ollama install: read-only probes plus one-shot generation.

Everything here talks to http://localhost:11434 and nothing else. The probes
only *look* at the local API; generate_ideas() and generate_build_guide() ask the
already-installed MODEL for project ideas and for one idea's build guide. No
pull, no subprocess, no cloud call, no retry loop, and every failure mode
degrades to a value instead of an exception.
"""

import json
import re
import time
from dataclasses import dataclass, field

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:1b"
TIMEOUT = 1.5  # seconds: localhost probe, so keep it short
# ponytail: 120s ceiling for one CPU-bound 1B generation, enforced as a real
# wall-clock deadline while reading Ollama's NDJSON - see _read_stream. If a
# real model needs more, raise this number; never add a retry loop.
GENERATE_TIMEOUT = 120
# ponytail: 1024 tokens. A compliant 3-idea reply is ~350 tokens and a
# 3-7 step guide ~300; a 1B that overruns the word caps lands near 700, and
# overshooting is a hard rejection.
NUM_PREDICT = 1024

# Prompt fragments. A 1B model follows short rules far better than long ones.
# The template shows THREE items on purpose: a 1B copies the SHAPE it can see,
# and a one-item template reliably produced exactly one idea (measured 6/6 runs
# either failed or returned 1 idea). Showing the shape is what fixes it.
def _idea_item(minutes: int = 20) -> str:
    """One idea object as shown in a prompt.

    `minutes` is a real, always-legal value rather than 0 because this 1B copies
    the placeholder it is shown: measured guides claimed "0 minutes" in 4 of 4
    runs while the template said 0. Callers pass min(time_minutes, 20) so the
    shown number can never break the limit it is shown next to.
    """
    return ('{"title": "", "description": "", "materials_used": [], '
            f'"additional_items": [], "estimated_minutes": {minutes}, '
            '"difficulty": "", "why_it_fits": ""}')


def _ideas_schema_text(minutes: int) -> str:
    """Three idea objects in a row: a 1B copies the count it can see, and a
    one-item template reliably produced exactly one idea (6 of 6 measured runs
    either failed or returned a single idea)."""
    return '{"ideas": [' + ", ".join([_idea_item(minutes)] * 3) + "]}"
_TIME_RULES = {
    15: "very simple and compact, one shape, nothing fiddly",
    30: "a small project with only a few steps",
    60: "more involved, but still finishable in one sitting",
}
_BUDGET_RULES = {
    # "the list of things to buy" rather than a field name: this table is shared
    # by both prompts, and each prompt names its own field (additional_items /
    # tools_needed) in its own rules.
    "zero": "rupees. Spend NOTHING - the list of things to buy must be empty",
    "100": "rupees. Prefer their materials; any purchase must be genuinely "
    "necessary and cost under 100",
    "500": "rupees. Prefer their materials; any purchase must be genuinely "
    "necessary and cost under 500",
    "flexible": "- extra materials are allowed, but never buy something "
    "pointless",
}
_SKILL_RULES = {
    "Beginner": "simple steps and common techniques, nothing that needs precision",
    "Intermediate": "some assembly and light problem-solving is welcome",
    "Advanced": "more complex is fine, but only with tools they actually have",
}
_IDEA_FIELDS = (
    "title",
    "description",
    "materials_used",
    "additional_items",
    "estimated_minutes",
    "difficulty",
    "why_it_fits",
)
_LIST_FIELDS = ("materials_used", "additional_items")
_TEXT_FIELDS = ("title", "description", "difficulty", "why_it_fits")
_IDEA_COUNT = 3

# Ollama's structured outputs: `format` as a JSON Schema constrains the reply's
# GRAMMAR, so the model physically cannot skip a required field or close an
# object early. Plain `format: "json"` only forced valid JSON, and a measured
# run under it dropped the last two fields of all three ideas; the schema form
# returned 3/3 complete replies on the same machine and the same prompt. Still
# one POST to localhost and no new dependency - the installed Ollama does it.
def _short(max_length: int) -> dict:
    """A short string in the grammar.

    The prompts already ask for brief fields; a cap here also stops a rambling
    1B from spending the whole token budget inside one field. Measured: one run
    reached num_predict after 44s and was rejected as truncated.
    """
    return {"type": "string", "maxLength": max_length}


_IDEA_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _short(60),
        "description": _short(170),
        "materials_used": {"type": "array", "items": _short(60), "minItems": 1},
        "additional_items": {"type": "array", "items": _short(60)},
        "estimated_minutes": {"type": "integer"},
        "difficulty": _short(10),
        "why_it_fits": _short(130),
    },
    "required": list(_IDEA_FIELDS),
}
_IDEAS_SCHEMA_BODY = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "minItems": _IDEA_COUNT,
            "maxItems": _IDEA_COUNT,
            "items": _IDEA_OBJECT_SCHEMA,
        },
    },
    "required": ["ideas"],
}


def _empty_only(schema: dict, *fields: str) -> dict:
    """Copy `schema` with `fields` allowed to hold nothing at all.

    Where a list must stay empty, the grammar refuses the alternative rather
    than the prompt asking for it: `maxItems: 0` makes a non-empty list
    ungrammatical, so the model cannot slip a purchase past the validator.
    Measured on this machine at ₹0: 0 purchases in 4 runs with this, against a
    purchase slipping into 2 of 4 runs without it.
    """
    if not fields:
        return schema
    empty = {"type": "array", "items": _short(60), "maxItems": 0}
    return {
        **schema,
        "properties": {**schema["properties"],
                       **{name: empty for name in fields}},
    }


def _ideas_schema(budget: str) -> dict:
    """Three complete ideas; at ₹0 a purchase is ungrammatical."""
    if budget != "zero":
        return _IDEAS_SCHEMA_BODY
    return {**_IDEAS_SCHEMA_BODY,
            "properties": {
                "ideas": {
                    **_IDEAS_SCHEMA_BODY["properties"]["ideas"],
                    "items": _empty_only(_IDEA_OBJECT_SCHEMA,
                                         "additional_items"),
                },
            }}

# Build Mode (Sprint 3): one project, not three ideas.
def _guide_schema_text(minutes: int) -> str:
    """The guide shape. estimated_minutes shows a real, always-legal number for
    the same reason _idea_item does - a 0 placeholder produced "0 minutes" in 4
    of 4 measured runs."""
    return (
        '{"title": "", "purpose": "", '
        f'"estimated_minutes": {minutes}, '
        '"materials_used": [], "tools_needed": [], '
        '"steps": [{"title": "", "instruction": "", "tip": ""}], '
        '"safety_notes": []}'
    )
_STEP_MIN, _STEP_MAX = 3, 7
_GUIDE_TEXT_FIELDS = ("title", "purpose")
_GUIDE_LIST_FIELDS = ("materials_used", "tools_needed", "safety_notes")
_GUIDE_FIELDS = _GUIDE_TEXT_FIELDS + _GUIDE_LIST_FIELDS + (
    "estimated_minutes",
    "steps",
)
_GUIDE_SCHEMA_BODY = {
    "type": "object",
    "properties": {
        "title": _short(70),
        "purpose": _short(110),
        "estimated_minutes": {"type": "integer"},
        "materials_used": {"type": "array", "items": _short(60), "minItems": 1},
        "tools_needed": {"type": "array", "items": _short(60)},
        "steps": {
            "type": "array",
            "minItems": _STEP_MIN,
            "maxItems": _STEP_MAX,
            "items": {
                "type": "object",
                "properties": {
                    "title": _short(60),
                    "instruction": _short(170),
                    "tip": _short(110),
                },
                "required": ["title", "instruction", "tip"],
            },
        },
        "safety_notes": {"type": "array", "items": _short(90)},
    },
    "required": list(_GUIDE_FIELDS),
}

# Constraint Remix (Sprint 4): ONE adapted idea, same seven fields.
_REMIX_SCHEMA_BODY = {
    "type": "object",
    "properties": {"idea": _IDEA_OBJECT_SCHEMA},
    "required": ["idea"],
}


def _guide_schema(budget: str) -> dict:
    """One build guide; at ₹0 a tool to buy is ungrammatical."""
    return (_empty_only(_GUIDE_SCHEMA_BODY, "tools_needed")
            if budget == "zero" else _GUIDE_SCHEMA_BODY)


def _remix_schema(budget: str, mode: str) -> dict:
    """One adapted idea; zero-buy (or ₹0) makes a purchase ungrammatical."""
    if budget != "zero" and mode != "zero_buy":
        return _REMIX_SCHEMA_BODY
    return {**_REMIX_SCHEMA_BODY,
            "properties": {"idea": _empty_only(_IDEA_OBJECT_SCHEMA,
                                               "additional_items")}}


# Material check: a claimed material passes only when EVERY significant word of
# it is present in what the user actually typed. Words this dull carry no
# meaning, so they can't be the reason a material is accepted or rejected.
_MATERIAL_STOPWORDS = frozenset({
    "a", "an", "the", "some", "piece", "pieces", "of", "and", "or", "optional",
    "plus", "with", "bit", "little", "few", "their", "his", "her", "your", "my",
})
_WORD_RE = re.compile(r"[a-z0-9]+")


_ES_PLURAL = ("ses", "xes", "zes", "ches", "shes")


def _singular(word: str) -> str:
    """The crudest possible plural fold: 'tubes' -> 'tube', 'glasses' -> 'glass'.

    Applied to BOTH sides of every comparison, so it can only ever make two
    spellings of the same word agree. No stemming, no real linguistics.
    """
    if len(word) > 4 and word.endswith(_ES_PLURAL):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _significant(text: str) -> set[str]:
    """The words in `text` that actually carry meaning, folded to lowercase
    singulars. Everything else - filler, punctuation, plurals - is dropped."""
    return {_singular(word) for word in _WORD_RE.findall(text.lower())
            if len(word) > 2 and word not in _MATERIAL_STOPWORDS}


def _covered_by_inventory(item: str, inventory: set[str]) -> bool:
    """True only when `item` is really made of things the user listed.

    Deliberately strict, because the whole app rests on this one comparison.
    Every significant word of the claimed material must appear in the user's
    own words, and only as a whole word: one shared word is NOT evidence
    ("cardboard" says nothing about "glue gun"), and neither is a substring
    across a word boundary ("ink" is not in "pink"). An ambiguous claim is
    therefore treated as unproven and rejected, never waved through.

    Direction matters: the model's claim must be covered by the inventory, so
    "tape" passes against "masking tape" while "masking tape" does not pass
    against "tape".
    """
    tokens = _significant(item)
    if not tokens:
        return True  # nothing meaningful claimed, so nothing to contradict
    return tokens <= inventory


# Restriction phrase -> words that break it. Covers the common hard restrictions
# only: "must fit on a desk", "indoor only", "no mess" and similar are not
# word-checkable, so those stay advisory and are reported to the user as
# unchecked - never presented as verified (see unchecked_restrictions).
# One shared negation so wording variants cost nothing: "no sharp tools",
# "without sharp tools", "avoid sharp blades", "must not use scissors",
# "nothing sharp" and "don't use sharp tools" all reach the same rule.
_NEG = r"(?:no|not|never|nothing|without|avoid|don'?t)\s+(?:\w+\s+){0,2}"
_RESTRICTION_RULES = (
    (r"\b" + _NEG + r"(?:sharp|blade\w*|knife\w*|scissors|razor\w*|saw|"
     r"scalpel|chisel)\b",
     r"\b(cut|cuts|cutting|knife|knives|razor|saw|sawdust|scissors|blade|"
     r"blades|scalpel|chisel|pincers)\b",
     "no sharp tools"),
    (r"\b" + _NEG + r"(?:glue\w*|gum\w*|adhesive\w*|cement\w*)\b",
     r"\b(glue\w*|gum\w*|adhesive\w*|cement\w*)\b",
     "no glue"),
    (r"\b" + _NEG + r"(?:electric\w*|power|batter\w*)\b",
     # "plug" is narrowed: "plug the gap between the tubes" is not electricity.
     r"\b(electric\w*|batter\w*|mains|plug\s+it\s+in|"
     r"plug\s+in|socket|led|bulb|solder\w*)\b",
     "no power"),
    (r"\b" + _NEG + r"(?:fire|flame\w*|heat\w*|hot|candle|stove|boil\w*)\b",
     # "hot" is narrowed so a safety note like "let it cool, it stays hot" passes.
     r"\b(fire|flame\w*|burn\w*|candle|stove|heat|hot\s+(?:glue|water|"
     r"liquid|plate|iron|pan|surface)|boiling)\b",
     "no fire or heat"),
)
# One actionable line per checkable rule, quoted verbatim into the prompt so the
# model is told the same thing the validator enforces. Keyed by the rule's own
# label, and a test asserts every rule has one, so the two cannot drift apart.
_RESTRICTION_GUIDE = {
    "no sharp tools": (
        'The words cut, knife, scissors, blade and saw are banned everywhere - '
        'not even to trim or shape. Build with folding, scoring a line with a '
        'blunt edge, tearing or punching instead. Good: "Fold the flap over '
        'and score it with a teaspoon." Bad: "Cut the cardboard to size."'
    ),
    "no glue": (
        'Glue, gum, adhesive and cement are banned everywhere. Good: "Fold the '
        'flap over and wrap string around it." Bad: "Glue the two halves '
        'together."'
    ),
    "no power": (
        'Batteries, plugs, sockets, bulbs and soldering are banned everywhere. '
        'Good: "Wind the string tightly around the tube." Bad: "Stick a '
        'battery pack to the base."'
    ),
    "no fire or heat": (
        'Flames, candles, hot glue, irons and boiling water are banned '
        'everywhere. Good: "Press the fold flat with your fingers." Bad: '
        '"Warm it over a candle until soft."'
    ),
}
_CLAUSE_SPLIT_RE = re.compile(r"[,;\n]|\band\b|\bbut\b", re.I)
# Genuinely hazardous only. Bare "cut"/"sharp"/"scissors" are absent on purpose -
# ordinary craft uses them; only a "no sharp tools" restriction may reject those.
# Adjectives are narrowed too ("a thinner layer" is not a thinner).
_SAFETY_TERMS = (
    r"mains",
    r"live\s+wire",
    r"outlet\s+wiring",
    r"solder\w*",
    r"blowtorch",
    r"heat\s+gun",
    r"open\s+flame",
    r"button\s+batter\w*",
    r"bleach",
    r"ammonia",
    r"solvent",
    r"paint\s*thinner|turpentine",
    r"acetone",
    r"methylene",
    r"firework\w*",
    r"gunpowder",
    r"projectile\w*",
    r"throw\s+(?:it|the\s+\w+)\s+(?:hard|fast|forceful\w*)",
    r"load[\s-]?bearing",
    r"weight[\s-]?bearing",
    r"structural",
)
_SAFETY_RE = re.compile(r"\b(?:" + "|".join(_SAFETY_TERMS) + r")\b", re.I)


@dataclass(frozen=True)
class GenerateResult:
    """One generation attempt: either validated ideas, a guide, or a reason."""

    ok: bool = False
    ideas: list = field(default_factory=list)
    guide: dict = field(default_factory=dict)
    error: str = ""


def is_running() -> bool:
    """True if a local Ollama server answers on its API endpoint."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/version", timeout=TIMEOUT)
        return response.ok
    except requests.RequestException:
        return False


def has_model() -> bool:
    """True if MODEL is already installed locally."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT)
        response.raise_for_status()
        names = [entry.get("name", "") for entry in response.json()["models"]]
    except (requests.RequestException, KeyError, AttributeError, TypeError, ValueError):
        return False
    return MODEL in names or f"{MODEL}:latest" in names


def status() -> tuple[bool, str]:
    """Return (model_ready, human_readable_status). Never raises."""
    try:
        if not is_running():
            return False, "Ollama isn't running on this Mac yet."
        if not has_model():
            return False, f"Ollama is running, but {MODEL} isn't pulled yet."
        return True, f"Ollama is running with {MODEL} ready."
    except Exception:  # noqa: BLE001 - a status line must never break the UI
        return False, "Couldn't reach Ollama on this Mac."


def _restriction_block(restrictions: str) -> str:
    """The hard restrictions as their own numbered rules, or "" when there are
    none this app can check.

    Built from _RESTRICTION_RULES - the very table the validator uses - so the
    prompt can never promise less than the gate enforces. One short paragraph
    per active rule with a compliant example, because measured on this 1B a
    restriction mentioned only inside a longer sentence was reliably broken:
    it reaches for "cut" the way any craft description does, and it only
    stopped when the banned verb and a workable substitute were named
    side by side.
    """
    lines = [
        f"{number}. {label} - REQUIRED: {_RESTRICTION_GUIDE[label]}"
        for number, (phrase, _forbidden, label) in enumerate(
            _RESTRICTION_RULES, start=1)
        if re.search(phrase, restrictions or "", re.I)
    ]
    if not lines:
        return ""
    return ("\nNon-negotiable restrictions. Obey each one in EVERY field -\n"
            "title, purpose, description, materials_used, tools_needed and "
            "every step\ninstruction, not just the words of the rule:\n"
            + "\n".join(lines) + "\n")


def build_prompt(
    materials, time_minutes, budget, skill_level, restrictions, preference
) -> str:
    """Render the user's constraints into one short, rule-shaped prompt."""
    hard_rules = restrictions.strip() or "none given"
    return f"""You suggest DIY projects built only from things people already have.

Materials and tools they have: {materials.strip()}
Treat that as the only supplies. Never assume a glue gun, hot glue, drill, saw,
soldering iron, sewing machine or any other specialist tool unless it is listed.

Time available: {time_minutes} minutes - keep it {_TIME_RULES.get(time_minutes, "finishable in one sitting")}.
Budget: {budget} {_BUDGET_RULES.get(budget, "rupees.")}
Skill level: {skill_level} - {_SKILL_RULES.get(skill_level, "adapt the steps to their level")}.
Hard restrictions, never break them: {hard_rules}
Preferred kind of project: {preference}
{_restriction_block(restrictions)}
Rules:
1. Use only the materials above plus plain everyday items like scissors, tape, string or water.
2. estimated_minutes must be at most {time_minutes}.
3. If the budget is 0, additional_items must be [].
4. Drop any idea of yours that breaks a restriction.
5. Give exactly 3 ideas that are genuinely different from each other: different
   purpose and different approach, not three variations of one project.
6. Buildability beats novelty. Never suggest something they cannot finish.
7. title: at most 5 words. description: at most 30 words. why_it_fits: at most 20
   words. difficulty: easy, medium or hard.
8. materials_used may only contain items from the materials list above. Never
   put an item the user already has in additional_items - that list is for
   things they must go and buy.

Answer with JSON only, no prose, no markdown, exactly this shape with 3 items:
{_ideas_schema_text(min(time_minutes, 20))}"""


def _slow_message() -> str:
    return (f"{MODEL} didn't answer within {GENERATE_TIMEOUT}s. The "
            f"first run loads the model, so give it a moment and try again.")


def _read_stream(response, deadline: float) -> tuple[str, str, bool]:
    """Drain Ollama's NDJSON reply until it says done - or until `deadline`.

    Streaming is what makes the ceiling real. The previous implementation
    spawned a daemon thread and joined it, which bounded the CALLER's wait but
    left the worker running and the request open on the server; the tests then
    needed a time.sleep() to let that stray thread die quietly. Reading the
    stream here means the only thing running is this function, so giving up
    closes the connection and ends the request instead of leaking it.

    Returns (text, done_reason, reply_was_not_an_object).
    """
    parts: list[str] = []
    seen, bad_shape, done_reason = False, False, ""
    for line in response.iter_lines():
        if time.monotonic() > deadline:
            raise requests.Timeout()  # our own wall clock, not the socket's
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except ValueError:
            continue  # a keep-alive or a partial line is not the answer
        if not seen:
            seen, bad_shape = True, not isinstance(chunk, dict)
        if not isinstance(chunk, dict):
            continue
        piece = chunk.get("response")
        if isinstance(piece, str):
            parts.append(piece)
        if chunk.get("done"):
            done_reason = chunk.get("done_reason") or ""
    return "".join(parts), done_reason, bad_shape


def _post(prompt: str, schema: dict) -> tuple[dict, str]:
    """The one and only request to Ollama. Returns (payload, "") or ({}, why).

    Every generator shares it so there is a single place that checks for the
    model, posts with the JSON grammar constraint, survives every transport
    failure, and turns the envelope into a JSON object. No retries, no second
    "fix it up" call, and no background thread.
    """
    if not has_model():
        return {}, (f"Ollama doesn't have {MODEL} installed. "
                   f"Fetch it once with: ollama pull {MODEL}")

    body = {
        "model": MODEL,
        "prompt": prompt,
        # Streamed so _read_stream can hold a wall-clock deadline over the whole
        # generation. See the note there.
        "stream": True,
        # Ollama's structured-outputs grammar constraint - the reason a 1B is
        # parseable AND complete. See the note on _IDEA_OBJECT_SCHEMA.
        "format": schema,
        "keep_alive": "10m",  # keep the model resident so repeat clicks are fast
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": NUM_PREDICT,
        },
    }
    deadline = time.monotonic() + GENERATE_TIMEOUT
    try:
        with requests.post(f"{OLLAMA_URL}/api/generate", json=body, stream=True,
                           timeout=(TIMEOUT, GENERATE_TIMEOUT)) as response:
            if not response.ok:
                return {}, (f"Ollama answered with HTTP {response.status_code}: "
                            f"{response.text[:120]}")
            # Leaving the with-block closes the connection, which is what stops
            # the request - on timeout as much as on success.
            text, done_reason, bad_shape = _read_stream(response, deadline)
    except requests.Timeout:
        return {}, _slow_message()
    except requests.RequestException:
        return {}, ("Couldn't reach Ollama on this Mac. "
                    "Is `ollama serve` running?")
    except Exception as exc:  # noqa: BLE001 - a bad reply must never raise out
        return {}, f"Ollama couldn't take that request ({type(exc).__name__})."

    if bad_shape:
        return {}, "Ollama's reply had an unexpected shape."
    if not text.strip():
        return {}, f"{MODEL} returned an empty answer."
    if done_reason == "length":
        return {}, (f"{MODEL} ran out of tokens before finishing its JSON. "
                    f"Try again.")
    payload = _extract_json_object(text)
    if payload is None:
        return {}, f"{MODEL} didn't return parseable JSON."
    return payload, ""


def checked_restrictions(restrictions: str) -> list[str]:
    """The hard restrictions this app really does enforce, word by word."""
    return [label for phrase, _forbidden, label in _RESTRICTION_RULES
            if re.search(phrase, restrictions or "", re.I)]


def unchecked_restrictions(restrictions: str) -> list[str]:
    """The user's own words that NO check can confirm, so never claim them.

    "must fit on my desk" is a real constraint and a word search cannot touch
    it. Silently passing it - or showing the project as satisfying it - would
    be claiming compliance from the absence of evidence, so the app names it
    as unchecked instead.
    """
    phrases = [(phrase, _f) for phrase, _f, _label in _RESTRICTION_RULES]
    pending: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(restrictions or ""):
        clause = clause.strip(" .;-!?")
        if len(clause) < 3:
            continue
        if any(re.search(phrase, clause, re.I) for phrase, _f in phrases):
            continue
        if clause.lower() not in {seen.lower() for seen in pending}:
            pending.append(clause)
    return pending


def generate_ideas(
    materials, time_minutes, budget, skill_level, restrictions, preference
) -> GenerateResult:
    """Ask the local model for exactly three ideas. One request, no retries."""
    payload, error = _post(
        build_prompt(materials, time_minutes, budget, skill_level,
                     restrictions, preference),
        _ideas_schema(budget),
    )
    if error:
        return GenerateResult(error=error)

    ideas, error = _validate(payload, time_minutes, budget,
                             inventory=_significant(materials))
    if error:
        return GenerateResult(error=error)
    return GenerateResult(ok=True, ideas=ideas)


def build_guide_prompt(
    idea, materials, time_minutes, budget, skill_level, restrictions, preference
) -> str:
    """Render ONE selected idea plus the original constraints into one prompt.

    Only the chosen idea and the user's own inputs go in - never the other ideas,
    never any earlier run.
    """
    hard_rules = restrictions.strip() or "none given"
    extras = ", ".join(idea["additional_items"]) or "nothing extra"
    return f"""You give step-by-step build instructions for ONE DIY project.

Project to build: {idea["title"].strip()}
What it is: {idea["description"].strip()}

Materials they have: {materials.strip()}
Treat that as the only supplies. materials_used lists only items from that list,
worded as they worded them. Never invent another material - if a step needs
something they do not have, put it in tools_needed instead.
Already planned extras for this project: {extras}

Time available: {time_minutes} minutes - keep it {_TIME_RULES.get(time_minutes, "finishable in one sitting")}.
estimated_minutes must be an integer between 1 and {time_minutes}.
Budget: {budget} {_BUDGET_RULES.get(budget, "rupees.")}
Skill level: {skill_level} - {_SKILL_RULES.get(skill_level, "adapt the steps to their level")}.
Hard restrictions, never break them: {hard_rules}
Preferred kind of project: {preference}
{_restriction_block(restrictions)}
Rules:
1. This ONE project only. Never swap it for a different or second project.
2. {_STEP_MIN}-{_STEP_MAX} steps in order, each a single small action. The
   simplest projects get the fewest steps.
3. tools_needed lists extra tools to buy, and must be [] when the budget is 0.
   Never list something they already have - materials_used is for that.
4. safety_notes lists real hazards only, and [] when the project is harmless.
5. No explanations, no reasoning, no commentary about the project.
6. title: at most 6 words. purpose: at most 15 words. instruction: at most 30
   words. tip: at most 15 words, and "" when there is nothing useful to add.
7. Obey every non-negotiable restriction above. If obeying one seems hard,
   choose different verbs and a different project, never the restriction.

Answer with JSON only, no prose, no markdown, exactly this shape, with the steps
array repeated {_STEP_MIN}-{_STEP_MAX} times:
{_guide_schema_text(min(time_minutes, 20))}"""


def generate_build_guide(
    idea, materials, time_minutes, budget, skill_level, restrictions, preference
) -> GenerateResult:
    """Ask the local model to build out ONE idea. One request, no retries."""
    payload, error = _post(
        build_guide_prompt(idea, materials, time_minutes, budget, skill_level,
                           restrictions, preference),
        _guide_schema(budget),
    )
    if error:
        return GenerateResult(error=error)

    guide, error = _validate_guide(
        payload, materials, time_minutes, budget, restrictions
    )
    if error:
        return GenerateResult(error=error)
    return GenerateResult(ok=True, guide=guide)


# Constraint Remix (Sprint 4). Four modes, all adapting the SELECTED idea for
# one changed condition. Each entry is (what must change, the mode's own gate).
_REMIX_MODES = ("zero_buy", "use_it_up", "remove_material", "simplify")
_REMIX_CHANGE = {
    "zero_buy": (
        "Make it a zero-buy version: it must work with nothing bought at all.",
        "additional_items must be [] - the empty list, never a note.",
    ),
    "use_it_up": (
        "Rework it so these materials get used up first: {focus}",
        "materials_used must be items from that list. Buying stays subject to "
        "the budget rule above.",
    ),
    "remove_material": (
        "Rework it without this material: {exclude}",
        "'{exclude}' must not appear in materials_used, and no part of the "
        "project may need it.",
    ),
    "simplify": (
        "Make it a beginner version: fewer parts, fewer steps, nothing that "
        "needs precision or a steady hand.",
        'difficulty must be "easy".',
    ),
}


def build_remix_prompt(
    idea, mode, materials, time_minutes, budget, skill_level, restrictions,
    preference, focus=None, exclude=None
) -> str:
    """Render ONE selected idea, its original constraints and ONE mode.

    Built like build_guide_prompt: the clicked idea and the user's own inputs,
    never another idea, never an earlier run, plus a brief that spells out what
    must change and what must stay the same.
    """
    change, must = _REMIX_CHANGE[mode]
    gone = (exclude or "").strip()
    change = change.format(focus=(focus or materials).strip() or "what they have",
                           exclude=gone)
    # The ban is repeated as a numbered rule: measured, one line inside the
    # brief was not enough - this 1B kept naming the removed material in
    # materials_used, which the gate then (correctly) rejected.
    extra_rule = (f"6. '{gone}' is not available for this version. Never name it "
                  f"in materials_used or in what the project does, and build "
                  f"from what is left.\n" if mode == "remove_material" else "")
    hard_rules = restrictions.strip() or "none given"
    used = ", ".join(idea["materials_used"]) or "none listed"
    extras = ", ".join(idea["additional_items"]) or "nothing extra"
    return f"""You adapt ONE DIY project for a single changed condition. You
never replace it with a different project.

Project to adapt: {idea["title"].strip()}
What it is: {idea["description"].strip()}
Materials it uses now: {used}
Extra items it needs now: {extras}
Why it fits their situation: {idea["why_it_fits"].strip()}

Materials they have: {materials.strip()}
Treat that as the only supplies. materials_used lists only items from that list,
worded as they worded them.
Time available: {time_minutes} minutes - keep it {_TIME_RULES.get(time_minutes, "finishable in one sitting")}.
Budget: {budget} {_BUDGET_RULES.get(budget, "rupees.")}
Skill level: {skill_level} - {_SKILL_RULES.get(skill_level, "adapt the steps to their level")}.
Hard restrictions, never break them: {hard_rules}
Preferred kind of project: {preference}
{_restriction_block(restrictions)}
What must change: {change}
{must}
What must stay the same: the purpose of the project, the {time_minutes} minute
time limit, the budget rule and every hard restriction.

Rules:
1. The SAME project, adapted - not a different one, not a second project.
2. estimated_minutes must be an integer between 1 and {time_minutes}.
3. title: at most 5 words. description: at most 30 words. why_it_fits: at most
   20 words. difficulty: easy, medium or hard.
4. No explanations, no reasoning, no commentary about the change.
5. additional_items may only hold things they must go and buy. Never list a
   material they already have.
{extra_rule}
Answer with JSON only, no prose, no markdown, exactly this shape:
{{"idea": {_idea_item(min(time_minutes, 20))}}}"""


def _title_key(idea: dict) -> str:
    """A title reduced to comparable words: case and punctuation set aside."""
    return re.sub(r"[^a-z0-9]+", " ", idea["title"].lower()).strip()


def _same_title(one: dict, two: dict) -> bool:
    """True when the two titles say the same thing.

    Used only to catch a remix that changed nothing - never to decide that a
    project is good, or that two different titles mean two different projects.
    """
    return _title_key(one) == _title_key(two)


def _remix_gate(original, adapted, mode, exclude, restrictions) -> str:
    """The mode's own deterministic gate. Returns "" when the mode is honoured.

    Kept minimal on purpose: material-priority for use_it_up is NOT checked,
    because no string match can prove the model used those materials first.
    """
    if mode == "zero_buy" and adapted["additional_items"]:
        return "The zero-buy version still lists things to buy."
    if mode == "simplify" and adapted["difficulty"].strip().lower().startswith("hard"):
        return "The simplified version still calls itself hard to do."

    banned = _significant(exclude or "")
    if mode == "remove_material":
        for item in adapted["materials_used"]:
            if banned & _significant(item):
                return f"The version without '{exclude}' still uses {item}."
        # The prose counts too: a remix can name the material in its
        # instructions while keeping it out of the list. The title is NOT
        # scanned - an honest adaptation often names what it removed
        # ("Cardboard tray, no tape"), and "no tape" is not a use of tape.
        prose = " ".join([adapted["description"], adapted["why_it_fits"],
                          *adapted["materials_used"]]).lower()
        for word in banned:
            if re.search(rf"\b{re.escape(word)}", prose):
                return (f"The version without '{exclude}' still needs it: "
                        f"'{word}' appears in the instructions.")

    # A remix must not reintroduce one of the user's own hard restrictions, and
    # this is the same table the guide is checked against.
    reason = _restriction_error(" ".join([
        adapted["title"], adapted["description"], adapted["why_it_fits"],
        *adapted["materials_used"], *adapted["additional_items"],
    ]), restrictions)
    if reason:
        return f"The remix {reason}."

    # Superficial adaptation, and only where the mode HAD something to change.
    # An unchanged title is not itself a fault: a removal that really happened
    # may well keep the name, and use_it_up and simplify can leave the name
    # alone and still be real adaptations. This only fires for zero-buy over an
    # original that bought something - measured returning the original's own
    # title with an empty materials list. remove_material needs no such rule:
    # a reply that removed nothing still names the material and is rejected
    # above with a message that says exactly what is still there.
    if mode == "zero_buy" and original["additional_items"] \
            and _same_title(original, adapted):
        return ("The zero-buy version has the same title as the original, "
                "which already needed purchases.")
    return ""


def remix_idea(
    idea, mode, materials, time_minutes, budget, skill_level, restrictions,
    preference, focus=None, exclude=None
) -> GenerateResult:
    """Ask for ONE adapted idea. One request, no retries, no repair call."""
    if mode not in _REMIX_MODES:
        return GenerateResult(error=f"'{mode}' isn't a remix mode.")
    if mode == "remove_material" and not (exclude or "").strip():
        return GenerateResult(error="Name the material to remove first.")

    payload, error = _post(
        build_remix_prompt(idea, mode, materials, time_minutes, budget,
                           skill_level, restrictions, preference,
                           focus=focus, exclude=exclude),
        _remix_schema(budget, mode),
    )
    if error:
        return GenerateResult(error=error)

    adapted = payload.get("idea")
    if not isinstance(adapted, dict):
        return GenerateResult(error=f"{MODEL} didn't return one adapted idea.")
    # Every field, type, time and budget rule from the ideas engine, unchanged.
    clean, error = _validate({"ideas": [adapted]}, time_minutes, budget, count=1,
                             inventory=_significant(materials))
    if error:
        return GenerateResult(error=error)
    error = _remix_gate(idea, clean[0], mode, exclude, restrictions)
    if error:
        return GenerateResult(error=error)
    return GenerateResult(ok=True, ideas=clean)


def _extract_json_object(text: str):
    """Pull the outermost JSON object out of a possibly messy reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    blob = text[start : end + 1]
    # Cheap second chance: a 1B loves single quotes.
    unquoted = re.sub(r"'([^'\\\n]*)'", r'"\1"', blob)
    for candidate in (blob, unquoted):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _already_owned(items, inventory: set[str]) -> str:
    """The first purchase that is really something the user already has, or "".

    A "buy this" list that names a thing from their own inventory is not a
    purchase, and showing it as one is the app quietly lying about money. Fail
    closed: name it and reject.
    """
    for item in items:
        if inventory and _covered_by_inventory(item, inventory):
            return item
    return ""


def _validate(payload, time_minutes, budget, count=_IDEA_COUNT, inventory=None
              ) -> tuple[list, str]:
    """Return (clean_ideas, "") or ([], reason_to_show_the_user).

    Constraints are requirements, so a violation is a rejection - never a silent
    clamp, filter or fix. `count` is 3 for the ideas engine and 1 for a remix,
    so both share every field, type, time and budget rule.
    """
    ideas = payload.get("ideas")
    if not isinstance(ideas, list):
        return [], f"{MODEL} didn't return an 'ideas' list."
    if not ideas:
        return [], f"{MODEL} returned no ideas."
    if len(ideas) != count:
        return [], f"{MODEL} returned {len(ideas)} ideas instead of exactly {count}."

    clean = []
    for number, idea in enumerate(ideas, start=1):
        if not isinstance(idea, dict):
            return [], f"Idea {number} isn't a JSON object."
        missing = [name for name in _IDEA_FIELDS if name not in idea]
        if missing:
            return [], f"Idea {number} is missing {', '.join(missing)}."

        title = idea["title"]
        if not isinstance(title, str) or not title.strip():
            return [], f"Idea {number} has no title."
        title = title.strip()

        minutes = _as_int(idea["estimated_minutes"])
        if minutes is None:
            return [], f"Idea '{title}' has an unreadable estimated_minutes."
        if not 1 <= minutes <= time_minutes:
            return [], (f"Idea '{title}' claims {minutes} minutes but only "
                        f"{time_minutes} are available.")

        texts = {}
        for name in _TEXT_FIELDS:
            value = idea[name]
            if not isinstance(value, str):
                return [], f"Idea '{title}' has a non-text '{name}'."
            texts[name] = value.strip()

        lists = {}
        for name in _LIST_FIELDS:
            value = idea[name]
            if not isinstance(value, list):
                return [], f"Idea '{title}' has a non-list '{name}'."
            # Only real strings: coercing would turn null into a literal "None".
            lists[name] = [
                item.strip() for item in value
                if isinstance(item, str) and item.strip()
            ]

        # A project that names no materials is not buildable from anything, so
        # an empty list is missing information, not a valid "nothing needed".
        if not lists["materials_used"]:
            return [], f"Idea '{title}' doesn't say what it's made from."

        if budget == "zero" and lists["additional_items"]:
            return [], (f"Idea '{title}' needs things to buy, but the budget "
                        "is ₹0.")
        owned = _already_owned(lists["additional_items"], inventory or set())
        if owned:
            return [], (f"Idea '{title}' lists '{owned}' as something to buy, "
                        "but that is already in your materials.")

        clean.append({**texts, **lists, "estimated_minutes": minutes})

    if count > 1 and len({idea["title"].lower() for idea in clean}) != count:
        return [], f"{MODEL} returned three near-identical ideas. Try again."
    return clean, ""


def _restriction_error(text: str, restrictions: str) -> str:
    """Why this text breaks a hard restriction ("breaks your ... because ...")

    Scans every field the caller passes in - not just the steps - because a
    guide can name a banned tool in its materials list and trip no step
    otherwise. Returns the exact word that broke the rule, so the message can
    tell the user what to argue with rather than just that it failed. Empty
    string means the text is clean, which is a fact about these rules only -
    never about the parts of a restriction no rule can check.
    """
    for phrase, forbidden, label in _RESTRICTION_RULES:
        if not re.search(phrase, restrictions or "", re.I):
            continue
        hit = re.search(forbidden, text, re.I)
        if hit:
            return (f"breaks your '{label}' restriction - '{hit.group(0).strip()}'"
                    f" is banned anywhere in it")
    return ""


def _validate_guide(payload, materials, time_minutes, budget, restrictions):
    """Return (clean_guide, "") or ({}, reason_to_show_the_user).

    Same contract as _validate: every rule is a requirement, and breaking one
    rejects the whole guide instead of quietly trimming or fixing it.
    """
    missing = [name for name in _GUIDE_FIELDS if name not in payload]
    if missing:
        return {}, f"The build guide is missing {', '.join(missing)}."

    texts = {}
    for name in _GUIDE_TEXT_FIELDS:
        value = payload[name]
        if not isinstance(value, str):
            return {}, f"The build guide has a non-text '{name}'."
        # title is required; purpose is decorative, so a blank one is dropped
        # rather than throwing away an otherwise valid guide.
        if name == "title" and not value.strip():
            return {}, "The build guide has no 'title'."
        texts[name] = value.strip()

    minutes = _as_int(payload["estimated_minutes"])
    if minutes is None:
        return {}, "The build guide has an unreadable estimated_minutes."
    if not 1 <= minutes <= time_minutes:
        return {}, (f"The guide claims {minutes} minutes but only "
                    f"{time_minutes} are available.")

    lists = {}
    for name in _GUIDE_LIST_FIELDS:
        value = payload[name]
        if not isinstance(value, list):
            return {}, f"The build guide has a non-list '{name}'."
        # Only real strings: coercing would turn null into a literal "None".
        lists[name] = [
            item.strip() for item in value
            if isinstance(item, str) and item.strip()
        ]

    steps = payload["steps"]
    if not isinstance(steps, list):
        return {}, "The build guide has a non-list 'steps'."
    if not _STEP_MIN <= len(steps) <= _STEP_MAX:
        return {}, (f"The guide has {len(steps)} steps instead of "
                    f"{_STEP_MIN}-{_STEP_MAX}.")

    clean_steps = []
    for number, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return {}, f"Step {number} isn't a JSON object."
        for name in ("title", "instruction"):
            value = step.get(name)
            if not isinstance(value, str) or not value.strip():
                return {}, f"Step {number} has no '{name}'."
        # A tip is optional: missing or blank is fine, non-string means none.
        tip = step.get("tip")
        clean_steps.append({
            "title": step["title"].strip(),
            "instruction": step["instruction"].strip(),
            "tip": tip.strip() if isinstance(tip, str) else "",
        })

    if not lists["materials_used"]:
        return {}, "The build guide doesn't say what to build it from."

    if budget == "zero" and lists["tools_needed"]:
        return {}, "The guide needs tools to buy, but the budget is ₹0."

    inventory = _significant(materials)
    for item in lists["materials_used"]:
        if not _covered_by_inventory(item, inventory):
            return {}, (f"The guide uses '{item}', which isn't in the "
                        "materials you listed.")
    owned = _already_owned(lists["tools_needed"], inventory)
    if owned:
        return {}, (f"The guide asks you to buy '{owned}', which is already in "
                    "your materials.")

    # Every field a person would read is scanned, materials_used included: a
    # banned tool can sit in the materials list without ever reaching a step.
    scanned = " ".join([
        texts["title"],
        texts["purpose"],
        *lists["materials_used"],
        *lists["tools_needed"],
        *(word for step in clean_steps for word in step.values()),
    ]).lower()
    reason = _restriction_error(scanned, restrictions)
    if reason:
        return {}, f"The guide {reason}."

    hazard = _SAFETY_RE.search(scanned)
    if hazard:
        return {}, f"The guide suggests something unsafe ({hazard.group(0)})."

    return {**texts, **lists,
            "estimated_minutes": minutes, "steps": clean_steps}, ""