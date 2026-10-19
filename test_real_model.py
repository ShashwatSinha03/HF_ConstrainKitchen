"""Opt-in real-model diagnostics. NOT part of the unit test suite.

Nothing here runs on import, so `python -m unittest` can import this module
while discovering `test*.py` and still never touch Ollama - the whole script
lives behind `if __name__ == "__main__"`.

Run it by hand:

    source .venv/bin/activate
    python test_real_model.py

It prints a duration and a one-word failure CATEGORY per check. It never
prints a prompt, a model reply or your materials, and it writes no files
anywhere. Exit code 1 means Ollama is unreachable, the model is missing, or a
check failed.

The seven checks are the ones worth knowing about, and they are deliberately
not a benchmark: idea generation (twice), a build guide with no restrictions,
a build guide under "no sharp tools", a ₹100 idea against a known inventory,
and the zero-buy and remove-a-material remixes. One request each, ever.
"""

import time

import llm

# Safe, boring, cheap inputs: no restrictions, nothing to buy.
_SAFE = {
    "materials": "cardboard, tape, string",
    "time_minutes": 30,
    "budget": "zero",
    "skill_level": "Beginner",
    "restrictions": "",
    "preference": "Surprise me",
}
# A real budget against a known inventory: this is where a purchase that is
# really something the user already owns used to slip through.
_PAID = dict(_SAFE, materials="cardboard, masking tape, string", budget="100")
# The restriction that measured worst on this 1B: describing cardboard craft
# reaches for "cut" the way any craft description does.
_SHARP = dict(_SAFE, restrictions="no sharp tools")
_IDEA_CHECKS = 2  # a point-in-time sample, not a benchmark
# Used only when idea generation fails, so the later checks still run. These are
# hand-written ideas, not anything the model produced.
_STAND_IN_IDEA = {
    "title": "Cardboard pen stand",
    "description": "Roll two cardboard tubes and tape them into a pen stand.",
    "materials_used": ["cardboard tubes", "tape"],
    "additional_items": [],
    "estimated_minutes": 20,
    "difficulty": "easy",
    "why_it_fits": "Everything comes from the tubes already on the shelf.",
}

# First match wins, so the more specific phrase must come first.
_CATEGORIES = (
    ("didn't answer within", "timeout"),
    ("couldn't reach", "ollama_unreachable"),
    ("doesn't have", "model_missing"),
    ("http 4", "http_error"),
    ("http 5", "http_error"),
    ("instead of exactly 3", "wrong_idea_count"),
    ("returned 1 idea", "wrong_idea_count"),
    ("returned no ideas", "no_ideas"),
    ("near-identical", "duplicate_ideas"),
    ("adapted idea", "wrong_shape"),
    ("'ideas' list", "wrong_shape"),
    ("unexpected shape", "wrong_shape"),
    ("isn't a json object", "bad_field_type"),
    ("non-text", "bad_field_type"),
    ("non-list", "bad_field_type"),
    ("unreadable", "bad_field_type"),
    ("parseable json", "unparseable"),
    ("empty answer", "empty_answer"),
    ("ran out of tokens", "truncated"),
    ("is missing", "missing_field"),
    ("no title", "missing_field"),
    ("has no '", "missing_field"),
    ("minutes but only", "over_time_limit"),
    ("budget is", "purchase_at_zero_budget"),
    ("already in your materials", "owned_item_called_a_purchase"),
    ("same title as the original", "superficial_remix"),
    ("still lists things to buy", "mode_not_honoured"),
    ("still uses", "mode_not_honoured"),
    ("still needs it", "mode_not_honoured"),
    ("still calls itself hard", "mode_not_honoured"),
    ("steps instead of", "bad_step_count"),
    ("isn't in the materials", "material_not_in_inventory"),
    ("restriction", "restriction_broken"),
    ("unsafe", "unsafe"),
)


def categorise(error: str) -> str:
    """A short label for a failure. The message itself is never echoed."""
    text = error.lower()
    for needle, label in _CATEGORIES:
        if needle in text:
            return label
    return "other_failure"


def verdict(result, ok_detail: str) -> tuple[bool, str]:
    """(passed, category_or_detail) - never the model's own words."""
    if result.ok:
        return True, ok_detail
    return False, categorise(result.error)


def problems() -> list[str]:
    """Why the real model can't be measured, or [] when it can."""
    if not llm.is_running():
        return ["Ollama isn't answering on http://localhost:11434."]
    if not llm.has_model():
        return [f"Ollama is running, but {llm.MODEL} isn't installed."]
    return []


def setup_help(issues: list[str]) -> list[str]:
    """Actionable lines for a machine that isn't ready."""
    return issues + [
        "start it with:  ollama serve",
        f"fetch the model once with:  ollama pull {llm.MODEL}",
    ]


def time_check(call) -> tuple[object, float]:
    """Run one generation call. Returns (result, elapsed seconds)."""
    started = time.monotonic()
    result = call()
    return result, round(time.monotonic() - started, 1)


def diagnose(report=print) -> int:
    """Print the whole diagnostic. Returns the process exit code.

    Exactly one request per check - a failed check is never retried, and the
    guides and remixes are built from an idea an earlier check already paid for.
    The point is to show how this model behaves on this machine right now, not
    to produce a nice result, so nothing is run twice to improve the score.
    """
    issues = problems()
    if issues:
        report("Constraint Kitchen - real-model diagnostic")
        for line in setup_help(issues):
            report(f"  {line}")
        return 1

    report("Constraint Kitchen - real-model diagnostic")
    report(f"  model {llm.MODEL} on {llm.OLLAMA_URL}: reachable, present")
    report("")

    results = []
    idea = None
    for attempt in range(1, _IDEA_CHECKS + 1):
        result, elapsed = time_check(lambda: llm.generate_ideas(**_SAFE))
        results.append((f"idea generation {attempt}",
                        *verdict(result, f"{len(result.ideas)} ideas"), elapsed))
        if result.ok and idea is None:
            idea = result.ideas[0]

    paid, paid_elapsed = time_check(lambda: llm.generate_ideas(**_PAID))
    results.append(("ideas, ₹100 + inventory",
                    *verdict(paid, f"{len(paid.ideas)} ideas"), paid_elapsed))
    if paid.ok and paid.ideas:
        paid_idea = paid.ideas[0]
    else:
        paid_idea = None

    base = idea if idea is not None else _STAND_IN_IDEA
    remix_base = paid_idea if paid_idea is not None else base
    source = "a real idea" if idea is not None else "a stand-in idea"
    remix_source = ("a ₹100 idea" if paid_idea is not None else source)

    for name, call in (
        (f"build guide, no limits ({source})",
         lambda: llm.generate_build_guide(idea=base, **_SAFE)),
        ("build guide, no sharp tools",
         lambda: llm.generate_build_guide(idea=base, **_SHARP)),
        (f"remix zero-buy ({remix_source})",
         lambda: llm.remix_idea(idea=remix_base, mode="zero_buy", **_PAID)),
        (f"remix remove-material ({remix_source})",
         lambda: llm.remix_idea(idea=remix_base, mode="remove_material",
                                exclude="tape", **_PAID)),
    ):
        result, elapsed = time_check(call)
        results.append((name, *verdict(result, "valid"), elapsed))

    for name, ok, detail, elapsed in results:
        line = "PASS" if ok else f"FAIL [{detail}]"
        report(f"  {name:<34} {line} after {elapsed}s")
    failed = sum(1 for _, ok, _, _ in results if not ok)
    total = round(sum(elapsed for *_, elapsed in results), 1)
    report("")
    report(f"  {len(results) - failed} of {len(results)} checks passed "
           f"in {total}s.")
    report("  Point-in-time sample on this machine: speed and reliability "
           "depend on the")
    report("  model version and the hardware. A FAIL is a refusal, not a bug: "
           "nothing is retried.")
    report("  Nothing was written to disk.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(diagnose())