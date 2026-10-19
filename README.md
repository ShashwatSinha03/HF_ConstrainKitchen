# Constraint Kitchen 🍳

Turn what you have into what you can make.

## The problem

You already own the materials. Cardboard from a delivery, a leftover roll of
string, sticks, tape, notebooks you were going to recycle. What you *don't*
have is a short list of ideas that respect your situation: your time, your
budget, your tools, your skill level.

Most idea lists assume a trip to the shop. This one starts where you are.

## Why local AI

The ideas get generated on your own machine by a small local model (Ollama +
Gemma 3 1B). No account, no API key, no usage meter, no idea list leaving your
Mac. It works on a laptop with modest compute and it keeps working on a plane.

## Known limitations, up front

- **A build guide under `no sharp tools` is unreliable** — measured about 2 in 6
  on the real model. The gate fails closed: you get a plain-English error and
  press the button again.
- **The model is not deterministic.** Any check can fail on any run. Every
  pass-rate in this README is a sample from one machine, not a benchmark.
- **No budget ceiling is checked beyond ₹0** — the model's reply carries no
  price, so a ₹100/₹500 ceiling cannot be verified without inventing one.
- **Nothing checks buildability or real-world safety.** The restriction and
  hazard checks are keyword rules over words, not understanding.
- **Nothing is stored.** No accounts, no history, no sharing, no telemetry.

## Quick start

```bash
git clone <this repo>
cd ConstraintKitchen
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull gemma3:1b              # one-time, ~815 MB
ollama serve                       # only if it isn't already running
streamlit run app.py
```

Then open <http://localhost:8501>.

Needs **Python 3.10+**, **Ollama 0.6 or newer** (the app sends a JSON Schema in
the request's `format` field, which needs structured outputs), and about **2 GB
free disk** — roughly 1 GB for the virtual environment and 815 MB for the model.

Generation talks only to `http://localhost:11434` on your own machine. If Ollama
isn't running the app says so at the bottom of the page and everything else keeps
working — it never pulls or installs anything for you. On a headless machine
(SSH, CI, a container) launch with `streamlit run app.py --server.headless true`.

Longer form: **Prerequisites**, **Installation**, **Preparing Gemma 3 1B**,
**Diagnostics**, **Troubleshooting**.

## Sprint 5 scope (current) — Idea Engine + Build Mode + Constraint Remix

Tell the app what you have (**A. Materials**) and what you can give up
(**B. Time, C. Budget, D. Skill level, E. Restrictions, F. Preference**), press
**Find My Project ✨**, and it asks your local Gemma 3 1B for **exactly three
different project ideas** that fit those constraints. Each one arrives as a card
with a title, a short description, estimated time, difficulty, the materials it
uses, anything extra it needs, and a sentence on why it fits your situation.

### Build Mode — one idea, step by step

Every idea card has its own **Build This →** button. Pressing it on one card
asks the model for **one project's build guide** and nothing else: title,
purpose, estimated minutes, the materials used, the tools needed, **3–7 numbered
steps** (each with a short instruction and an optional tip), and safety notes
only when the project genuinely has a hazard.

Nothing is generated until you press a button. Opening the page, editing an
input, or clicking an unrelated control never triggers a generation — each
**Build This →** click is exactly one request. The guide stays on screen across
reruns; editing an input marks it stale with a caption instead of silently
looking current; pressing **Find My Project ✨** again retires the old guide; and
a failed build clears it and shows the reason, so an error never leaves the
previous project's steps on screen looking like they belong to the new choice.

### Constraint Remix — change one condition, keep the project

Every idea card also carries four compact remix actions: **♻️ Zero-buy**,
**Use-it-up**, **Simplify it** and **Remove a material** (the last one takes the
material to drop in a small box on the card). Each press sends **that card's
idea** — the original title, description, materials and `why_it_fits` — plus
your original constraints and a brief that says what must change and what must
stay the same, and asks for **one adapted idea**:

| Mode | What it asks for | Enforced in code |
| --- | --- | --- |
| **Zero-buy** | the same project with nothing bought at all | `additional_items` must be empty, and the request's grammar makes a non-empty list impossible to even write; and the result may not be the original's own title unchanged when the original did buy something |
| **Use-it-up** | the same project, prioritising the materials you name (defaults to everything you listed) | structure and every idea rule; **priority itself is not checked** |
| **Remove a material** | the same project without the material you name | the excluded material is rejected if it appears in `materials_used` **or in the remix's own instructions**; the title is not scanned, because "Cardboard tray, no tape" names what it removed rather than using it |
| **Simplify it** | the same project, easier: fewer parts, fewer steps | every idea rule, plus `difficulty` may not still call itself hard |

Every remix is also checked against **your own hard restrictions**, and rejected
if it reintroduces one. A remix may not be shown as a success while
`additional_items` is non-empty — that gate runs before anything is rendered.

**Enforced vs advisory.** Zero-buy and Remove-a-material are gates: a reply
that breaks them is **rejected**, your original idea stays on screen untouched,
and the reason is shown. Use-it-up is **advisory only** — no string match can
prove the model used those materials *first*, so the app says so under the
buttons and you should read the `Uses:` line. Simplify is deliberately thin: it
rejects an output that still claims to be hard, and otherwise trusts the model,
because any readability or keyword score would be a fake measurement.

A successful remix appears **inside the same card**, labelled, below the
original — the original is never replaced or hidden. Any build guide already on
screen was written for the *pre-remix* idea, so it is retired the moment a remix
succeeds and the card offers **Build This Remix →** for steps that match the new
version. A failed remix costs one request, changes nothing, and never triggers
a second call to "repair" the output. Pressing **Find My Project ✨** again
retires any old remix.

Not in this sprint (and deliberately so): project history or any persistence,
image upload or vision, charts, progress tracking, rankings, confidence scores,
model reasoning transcripts, a second "fix it up" call, and any network call
that isn't to `http://localhost:11434`.

## What the app checks, and what it only claims

The app states this in the interface as well as here, because the useful
version is the honest one. **A reply whose JSON validated is not thereby a safe,
buildable, affordable or compliant project.**

**Deterministically enforced** — a reply that breaks one of these is rejected
whole, never trimmed or repaired:

| Constraint | Enforced on | How |
| --- | --- | --- |
| exactly 3 distinct ideas, every field present and typed | ideas, remixes | schema + validator |
| a project says what it's made from | `materials_used` | grammar `minItems: 1` **and** validator — an empty list is missing information, not "nothing needed" |
| `estimated_minutes` within your time limit | everything | validator |
| ₹0 means no purchases | ideas, guides, remixes | grammar `maxItems: 0` **and** validator |
| a purchase is never something you already own | `additional_items`, `tools_needed` | every significant word must be covered by your inventory |
| a guide's materials are really yours | `materials_used` (guides only) | every significant word must be covered by your inventory |
| `no sharp tools` / `no glue` / `no power` / `no fire or heat` | guide title, purpose, **materials_used**, tools and every step field; every remix field | keyword rules from one shared table |
| hazardous instructions | title, purpose, materials, tools, steps | keyword blocklist |
| 3–7 steps, each with a real instruction | guides | validator |
| remove-a-material really removed it | `materials_used` **and** the remix's prose | word overlap |
| zero-buy really needs nothing | `additional_items` | empty, or rejected |
| a remix did not just repeat the original | title, **only when the mode had something to change** | comparison with the clicked idea |
| the remix did not reintroduce your restriction | every remix field | the same keyword table |

**Explicitly NOT enforced, and never presented as if it were:**

- **Any ₹100/₹500 ceiling.** The reply carries no price field of any kind, so
  there is no number to compare against a budget. The app says so under the
  budget control rather than inventing a model-estimated price, which would
  manufacture exactly the evidence it is supposed to be without. Only ₹0 is a
  real rule.
- **Restrictions no keyword can check** — "must fit on my desk", "indoor only",
  "no mess". The restrictions box splits your own words into
  *Checked for you: …* and *NOT checked — your call: …*, so a checkable rule is
  never confused with an uncheckable one.
- **Idea cards' material lists.** The inventory check runs on the *guide's*
  `materials_used`, not on an idea card's. A card can name a material you don't
  have; the guide for it cannot.
- **`safety_notes`.** The hazard blocklist and the restriction scan cover the
  guide's title, purpose, materials, tools and every step — *not* its safety
  notes. A hazard named only there passes every gate, so read them yourself.
- **`use_it_up` priority, and any judgement of whether an adaptation is
  actually better.** No string match proves either.
- **Buildability, structural soundness and real-world safety.** The hazard list
  is keywords, not a review.

## How generation works

One POST per click, no retries, no second "fix it up" call:

1. `llm.generate_ideas()`, `llm.generate_build_guide()` and `llm.remix_idea()`
   share one private `_post()` helper, which first checks `has_model()` — if
   `gemma3:1b` isn't installed it returns an error instead of posting anything.
2. It POSTs to `http://localhost:11434/api/generate` with `"format"` set to a
   **JSON Schema**. Ollama turns that schema into a grammar,
   so the reply cannot skip a required field, close an object early, or return
   fewer than three ideas / fewer than three steps — that is the single most
   important reason a 1B model is usable here. (Sprint 3 sent `"format": "json"`,
   which only forces *valid* JSON; against the real model it dropped the last
   two fields of every idea and failed the whole run. See **Real-model
   results**.) At ₹0 the schema also sets `maxItems: 0` on the purchase list, so
   a non-empty purchase is ungrammatical rather than merely forbidden.
3. Options are conservative: `temperature` 0.7, `top_p` 0.9, `num_predict` 1024,
   and `keep_alive: 10m` so the model stays resident between clicks.
4. The request is bounded by a **real 120-second wall-clock deadline**
   (`llm.GENERATE_TIMEOUT`), enforced by `_post()` itself rather than by
   `requests`. A socket timeout is not a total-request deadline for a buffered
   reply — measured on this machine, a generation once ran 420s with
   `timeout=120` and `requests` never raised. The request therefore goes out with
   `stream: true`, and `_read_stream()` checks the deadline on every NDJSON line
   it reads; on expiry it raises, the `with` block closes the connection, and the
   request ends. Sprint 4 bounded the *caller* with a daemon thread and
   `join(120)` instead: that left the worker running and the request open on the
   server after giving up, which is why its timeout tests needed a
   `time.sleep(0.6)` to let the stray thread die quietly. There is now no
   background thread and no shared dict between threads, and a regression test
   walks the thread stacks to prove it. No retry, no second request.
5. The reply is parsed tolerantly (markdown fences, surrounding prose, single
   quotes) and then validated hard.

The Build Mode and Remix prompts carry **only the idea you clicked plus your
original inputs** — never the other ideas, never any earlier run — and they
reuse the same time, budget and skill rule tables as the ideas prompt, so all
three prompts stay consistent. Two prompts also matter because this 1B copies
what it is shown: the ideas template shows **three** idea objects (a one-object
template produced one idea in 6 of 6 measured runs), and every template shows a
legal `estimated_minutes` placeholder rather than `0` (a `0` placeholder
produced "0 minutes" answers in 4 of 4 measured guide runs).

A hard restriction is additionally repeated into the prompt as **its own
numbered rule, quoted from the very table the validator uses**, naming both the
banned verb and a workable substitute:

```
Non-negotiable restrictions. Obey each one in EVERY field -
title, purpose, description, materials_used, tools_needed and every step
instruction, not just the words of the rule:
1. no sharp tools - REQUIRED: The words cut, knife, scissors, blade and saw are
   banned everywhere - not even to trim or shape. Build with folding, scoring a
   line with a blunt edge, tearing or punching instead. Good: "Fold the flap over
   and score it with a teaspoon." Bad: "Cut the cardboard to size."
```

That is measured, not stylistic: with the restriction buried in one line of the
brief, **0 of 10** real build guides under `no sharp tools` passed, because a 1B
describing cardboard craft reaches for "cut" the way any craft description does.
With the block above, **2 of 6** measured runs passed — an improvement, and
still not a fix. See **Real-model results**.

A build guide looks like this:

```json
{"title": "", "purpose": "", "estimated_minutes": 20,
 "materials_used": ["cardboard"], "tools_needed": [],
 "steps": [{"title": "", "instruction": "", "tip": ""}],
 "safety_notes": []}
```

The values above are shapes, but two are load-bearing: `materials_used` must
hold **at least one** item and `estimated_minutes` must be **1 or more within
your time limit**. An empty list and a `0` are both hard rejections.

### Validation is rejection, not repair

Constraints are requirements. If the reply is unparseable, isn't a list, doesn't
hold exactly 3 ideas, is missing a field, has a bad type, exceeds your time
limit, has an empty title, or — at ₹0 — lists anything extra to buy, the whole
response is **rejected** with a plain-English reason. Nothing is silently
clamped, filtered or patched up, and a rejected run never displays partial JSON
as if it had succeeded. Press the button again to retry. A rejection is a
legitimate outcome, not a bug: nothing here was loosened to raise the pass rate.

Build guides are held to the same contract, plus their own gates: required
fields and types present, `estimated_minutes` an integer within your time limit,
3–7 steps each with a real title and instruction, only real strings inside every
list (a `null` is dropped, never shown as the word "None"), `tools_needed` empty
at ₹0, no tool in `tools_needed` that you already own, every claimed
`materials_used` item covered by what you actually typed, your hard restrictions
obeyed, and no hazardous instruction.

**The restriction scan covers `materials_used`** as well as the title, purpose,
tools and every step field. Sprint 4 omitted it, so a guide could require
`scissors` in its materials list and pass while owning none of the steps that
used it — the only reason that defect was invisible is that such a guide was
also rejected for the unowned material, right up until you happened to own
scissors and set `no sharp tools`. That case is now rejected by name:

> The guide breaks your 'no sharp tools' restriction - 'cut' is banned anywhere
> in it.

On a violation the guide is **not** displayed, your selected idea and every
other piece of state are left exactly as they were, no retry is issued, and the
error names both the rule and the word that tripped it.

**The material check is all-of, not any-of.** A claimed material passes only when
**every** significant word of it appears in your inventory as a whole word, after
lowercasing and folding plurals (`tubes`↔`tube`, `glasses`↔`glass`). Sprint 4's
rule accepted a claim if *any one* significant word matched anywhere inside a
word you typed, so `cardboard tubes` passed on the word `cardboard` against an
inventory of `cardboard`, and `ink` passed inside `pink`. Direction matters: your
asking for less than you have is fine (`tape` against `masking tape`), asking for
more is not (`masking tape` against `tape`). An ambiguous claim is **rejected**,
because an unproven material is not a material.

This rejects more, and that is intended. Measured on the ten distinct material
strings a real run produced: nine were accepted under both rules, and `Cardboard
boxes` against an inventory of `cardboard, masking tape, string` flipped from
accepted to rejected. In a six-payload replay of a full real run, one payload
flipped — a remove-a-material remix that listed `String` as something to buy
while string was already in the inventory, which the "a purchase is never
something you own" rule now catches.

**A purchase is never something you already own.** At a paid budget the model
used to list `tape` as something to buy when tape was in the inventory (3 of 3
ideas in one measured run at ₹100). Anything in `additional_items` or
`tools_needed` that your inventory already covers is now rejected by name.

A remix is held to the **same idea contract** (all seven fields, correct types,
time limit, budget rule, inventory rule) plus its mode's own gate, exactly as
tabled above, plus your restrictions, plus a **superficial-change** check that
fires *only where the mode had something to change*: a zero-buy remix of an
original that bought something may not come back under the original's own title.
An unchanged title is otherwise allowed — a removal that genuinely happened may
keep the name, and `use_it_up` and `simplify` may legitimately not rename
anything.

## Diagnostics — measuring the real model

`python -m unittest` is **fully mocked**: it runs against a loopback stub HTTP
server and never contacts Ollama. To measure the actual model instead, run the
opt-in diagnostic by hand:

```bash
source .venv/bin/activate
ollama serve              # if it isn't already running
ollama pull gemma3:1b     # one-time download
python test_real_model.py
```

It first checks that Ollama answers and that `gemma3:1b` is installed; if
either is missing it prints the exact command to fix it and exits **1**. Then it
runs **seven real checks**, one request each, never a retry:

| Check | Inputs |
| --- | --- |
| idea generation ×2 | `cardboard, tape, string`, 30 min, ₹0, no restrictions |
| ideas with a real budget | `cardboard, masking tape, string`, ₹100 |
| build guide | the first idea from an earlier check, no restrictions |
| build guide under a restriction | the same idea, `no sharp tools` |
| remix zero-buy | the first ₹100 idea |
| remix remove-a-material | the same idea, excluding `tape` |

Each line prints a duration and a one-word failure **category**
(`wrong_idea_count`, `missing_field`, `unparseable`, `http_error`, `timeout`,
`restriction_broken`, `material_not_in_inventory`,
`owned_item_called_a_purchase`, `superficial_remix`, `mode_not_honoured`, …).
It **never prints a prompt, a model reply or your materials, and it writes no
file anywhere** — the diagnostic output is the only record, and there is no log
to leak into. A `FAIL` is a refusal, not a crash, and nothing is retried to
improve the line. Exit code is 1 if Ollama is missing or any check fails.

Because the diagnostic never prints payloads, a `PASS` tells you the reply
satisfied every gate — not what the reply said.

### Real-model results — measured, not assumed

Measured on this machine (Apple M3, 8 GB, Ollama 0.40.3, `gemma3:1b`, one
resident model), Sprint 5 code. **Two** full diagnostic runs of seven checks
each, both reported:

| Check | Run 1 | Run 2 |
| --- | --- | --- |
| idea generation ×2 | **2 / 2** — 12.8 s, 16.9 s | **2 / 2** — 18.1 s, 17.4 s |
| ideas, ₹100 + known inventory | **1 / 1** — 16.9 s | **1 / 1** — 18.3 s |
| build guide, no restrictions | **1 / 1** — 8.9 s | **1 / 1** — 10.6 s |
| build guide, `no sharp tools` | **1 / 1** — 11.7 s | **0 / 1** — 22.5 s, `restriction_broken` |
| remix zero-buy | **1 / 1** — 6.5 s | **1 / 1** — 6.1 s |
| remix remove-a-material | **1 / 1** — 4.8 s | **1 / 1** — 8.4 s |
| **total** | **7 of 7** in 78.5 s | **6 of 7** in 101.4 s |

**`no sharp tools` is still unreliable: about 2 of 6 measured**, against
**0 of 10** in Sprint 4. The restriction block in the prompt helps — it changed
a rule that never fired into one that usually fires — but it does not make the
mode reliable. A 1B describing cardboard craft still reaches for "cut", and when
it does, the gate rejects the guide and the user presses the button again. Read
this as "much better, still the weakest path in the app", not as fixed. Neither
run was repeated to improve the table; the second run is the one that failed, and
it is here for that reason.

In the two runs tabulated above, everything except `no sharp tools` passed —
**12 of 12**. That is not a guarantee: later untabulated runs have failed other
checks too (one scored 5 of 7, failing different cases). The model is not
deterministic, so treat every figure here as a sample, not a benchmark.

Latency: 4.8 s to 22.5 s per call on an idle machine, against a 120 s ceiling.
Sprint 4 measured **103.1 s** against the same ceiling, and the current code has
since been seen at roughly that figure once too. A 276.8 s observation under
heavy load could not be reproduced in eight
subsequent calls. **Expect the deadline to be reached under load** — treat 120 s
as the design ceiling, not a measured maximum.

The same scenarios on the pre-Sprint-5 code, same machine and model: the
unrestricted guide passed, and the `no sharp tools` guide was **rejected**
(`'cut'`), the ideas passed, and both remixes passed — with the remove-material
remix passing *while listing string as something to buy*, which Sprint 5 now
rejects.

Sprint 4's wider history, for context (69 real generations across its session,
including runs before each fix landed): **63 passed, 6 rejected**. Sprint 5's own
real-model runs: **14 generations, 13 passed, 1 rejected** in the two runs tabulated above, plus later runs that failed other checks (one scored 5 of 7).

These are point-in-time samples, not a benchmark: the numbers depend on this
hardware, on Ollama, and on the exact model build, and a warm model is far
faster than the first run after a restart. Two runs is not a sample worth
extrapolating from in either direction.

**Not guaranteed.** Read these as "usually works, occasionally needs another
press", not a promise:

- **A reply is still rejected whenever it breaks a rule**: too many or too few
  ideas, a time estimate over your limit, a duplicate title, a purchase at ₹0,
  a material you may not have, a step count outside 3–7, a restriction broken, a
  hazard, a remix that changed nothing. That is by design — press the button
  again. No validation gate was loosened anywhere in Sprint 5 to raise a pass
  rate.
- **`no sharp tools` in Build Mode remains the weakest path** — 2 of 6 measured.
- **Remove a material remains the weakest remix mode.** Sprint 4 measured 4 of 6;
  Sprint 5 tightened the gate to cover the prose as well as the materials list,
  which can only reject more, and measured 2 of 2 here.
- Structured output fixes *shape*, never *truth*: `estimated_minutes` is the
  model's claim about its own plan, not a measurement, and nothing here checks
  that a guide is genuinely buildable or that a remix really is simpler.
- **`use_it_up` priority cannot be verified by string matching** and is not
  checked at all.
- Nothing is persisted, shared or queued: everything lives in that one browser
  session, and the app never starts, pulls or installs anything for you.

## Troubleshooting

| Symptom | What it means |
| --- | --- |
| `ollama: command not found` | Ollama isn't installed. See <https://ollama.com>. |
| `ollama serve` → `address already in use` | It's already running. Skip it. |
| Every button says "Ollama isn't running" | Run `ollama serve` in a **separate terminal** and leave it there. |
| "Ollama is running, but gemma3:1b isn't pulled yet" | Run `ollama pull gemma3:1b` once (~815 MB). |
| Streamlit: `Address already in use` on 8501 | Another copy of the app is already running. |
| The first click feels slow | The model is being loaded into memory. Later clicks are faster. |
| `python3 -m venv` fails on Debian/Ubuntu | `sudo apt install python3-venv`. |
| Nothing happens and no browser opens | Headless machine: `streamlit run app.py --server.headless true`. |
| "The guide breaks your 'no sharp tools' restriction" | The model reached for a banned verb. The gate refuses by design — press again. |

## Prerequisites

- Python 3.10 or newer
- ~1 GB of free disk for the virtual environment
- [Ollama](https://ollama.com) with the `gemma3:1b` model, for generation.
  Measured on Ollama 0.40.3: **0.5.0 or newer** is needed, because the request's
  `format` field carries a JSON Schema (structured outputs) rather than the
  bare `"json"` mode.

## Installation

Installing Python deps needs internet access (to fetch `streamlit` and
`requests` from PyPI). Fetching the model is a one-time download. Generating
ideas needs neither.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparing Gemma 3 1B

```bash
ollama pull gemma3:1b   # one-time download, ~800 MB
ollama serve            # keeps the local API on http://localhost:11434
```

The status line at the bottom of the app then reads *Ollama is running with
gemma3:1b ready.* If Ollama is down the app says so and explains these two
commands — **the app never starts, pulls or installs anything for you**, and the
input form stays fully usable.

## Launch

```bash
streamlit run app.py
```

Then open http://localhost:8501.

The app binds to `localhost` only (see `.streamlit/config.toml`), so it is not
reachable from other machines on your network.

### Runtime network behaviour

After installation the only network traffic the app ever makes is to
`http://localhost:11434` — loopback Ollama, on this same machine:

- a read-only status probe (`/api/version`, `/api/tags`) run once on first load and
  re-probed on each click, with a 1.5s timeout, and
- one `/api/generate` POST per button click — **Find My Project ✨**, one
  **Build This →**, **Build This Remix →**, or one of the four remix buttons —
  which is where your materials and constraints are sent so the model can
  answer.

Both stay on loopback. Nothing is stored: ideas, the one build guide and the one
remix live in Streamlit's session state for that browser session and are gone
when it ends. There is no database, no account, no diagnostic log and no
analytics (`browser.gatherUsageStats = false`).

## Project layout

```
constraint-kitchen/
├── app.py                  # Streamlit UI: constraints, cards, Build Mode, Remix
├── llm.py                  # local Ollama probes, prompts, generation, validation
├── test_llm.py             # probe, generation, validation, remix + diagnostics tests
├── test_app.py             # Streamlit behaviour tests via AppTest (stub server)
├── test_real_model.py      # opt-in REAL-model diagnostic (never run by unittest)
├── .streamlit/config.toml  # loopback-only bind, telemetry off
├── requirements.txt        # streamlit, requests
├── README.md
└── .gitignore
```

## Tests

```bash
source .venv/bin/activate
python -m unittest          # 233 tests, all mocked
```

Every one of those tests is mocked: they run against a local stub HTTP server
that mimics Ollama's API and never start, contact or require a real Ollama
install. The full suite has been run with `ollama serve` stopped and passes
unchanged. `test_real_model.py` is the only file that touches a real model, it
holds no test cases, and all of its work sits behind an `if __name__ == "__main__"`
guard — so test discovery may import it and still never runs it. Use
`python test_real_model.py` to measure the real model (see **Diagnostics**).

### What is verified, and what is not

**Verified here, by the automated tests** (against a stub that mimics Ollama's
API):

- the three probes behave correctly, including when Ollama is absent;
- a generation request carries model `gemma3:1b`, a JSON-Schema `format` that
  requires every field (and forbids a purchase at ₹0), `stream: true`, the
  conservative options, and every user constraint in the prompt — and the Build
  Mode and Remix prompts carry only the clicked idea;
- **exactly one** `/api/generate` request happens per `generate_ideas()`,
  `generate_build_guide()` or `remix_idea()` call, and none happens at all when
  the model isn't installed, on a rerun, on typing in a remix input, or after a
  rejected reply;
- connection failure, timeout, non-2xx, empty reply, truncated reply and
  unparseable reply all return a readable message and never raise;
- **a timed-out request leaves no thread running** — the test walks the live
  thread stacks and asserts none is executing `llm.py`. Against Sprint 4's
  daemon-thread implementation that test fails, naming the leaked worker;
- every validation rule accepts or rejects the right payloads, including fenced
  and single-quoted JSON;
- restrictions are enforced over the **materials list** as well as the steps,
  in the wordings people actually type (`no sharp tools`, `must not use sharp
  tools`, `don't use sharp tools`, `no knives or scissors`, …), that an obeying
  guide is still accepted, and that the error names both the rule and the word;
- uncheckable restrictions are **named as unchecked** rather than treated as
  satisfied, and each budget control states what it does and does not enforce;
- the material rule is all-of-not-any-of: a shared word, a word inside a longer
  word, and plural spelling all behave as documented; a purchase you already own
  is rejected by name; and ₹0 stays the only enforced ceiling, with no price
  field anywhere in the schema to imply otherwise;
- each remix gate fires: zero-buy with a purchase, remove-a-material still
  naming it in the list *or* in the instructions, simplify still hard, a remix
  re-shrinking a restriction, and a zero-buy remix repeating the original title
  when the original bought something — each with no repair call, while a
  legitimate adaptation that keeps the title is still accepted;
- the Streamlit app warns on empty materials without generating, renders three
  cards on success, keeps them across reruns, does not regenerate when inputs
  change, and clears stale ideas when a later run fails;
- Build Mode gives each card its own button and its own key, costs exactly one
  request per click, shows that idea's guide, keeps it across reruns, marks it
  stale when you edit an input, replaces it when you pick another idea or press
  **Find My Project ✨** again, and turns every failure into a readable message
  with the cards still usable — including a restriction refusal, which shows
  neither guide nor partial data and leaves the cards intact;
- each of the four remix modes builds its own prompt from the selected idea and
  the original constraints; a failed remix keeps the original idea *and* an
  existing guide; a successful remix retires that guide and offers a separate
  build for the remixed version; and the diagnostic reports unreachable Ollama, a
  missing model, a clean run and a failure **without printing model output and
  without writing a single file**.

**Verified against the real model too**, on the machine described in **Real-model
results**: the requests above really do come back complete, mostly inside every
gate and inside the 120 s deadline. 13 of 14 generations passed in the two runs
tabulated above, and
the one failure was a deliberate refusal of a guide that broke `no sharp tools`.
Those samples are single-run and machine-specific.

**Not verified by anything, here or there:**

- **real-model reliability as a guarantee.** A 1B model can still produce output
  that trips the hard validation — too many or too few ideas, a time estimate
  over your limit, a duplicate title, a step count outside 3–7, a material that
  isn't really in your inventory, a mode it didn't honour. That is a designed
  outcome (the app refuses rather than shows you something that breaks your
  constraints), but it means pressing the button again may be needed.
- **material-priority compliance for use-it-up** — no string match can prove it,
  so nothing checks it;
- prompt *quality*: whether the ideas are genuinely distinct, whether a remix
  really is simpler, and whether the generated steps genuinely build something,
  are judgements neither a stub nor a pass-rate can make.

### Enforced vs. advisory

The full table is at the top of this file, under **What the app checks, and what
it only claims**. In short:

- *Ideas*: exactly three; three distinct titles; every field present and
  correctly typed; `estimated_minutes` between 1 and your time limit; at ₹0 an
  empty `additional_items`; and nothing listed for purchase that your inventory
  already covers.
- *Build guides*: every field present and correctly typed; `estimated_minutes`
   an integer between 1 and your time limit; 3–7 steps, each with a non-empty
   title and instruction (a tip is optional, and non-strings in any list are
   dropped rather than shown as `"None"`); `tools_needed` empty at ₹0; no tool in
   `tools_needed` you already own; **every** `materials_used` item covered, word
   for word, by what you typed; your hard restrictions obeyed across the title,
   purpose, materials, tools and every step field; and a **hazard blocklist**
   (mains, live wires, soldering, blowtorch, heat gun, open flame, button
   batteries, bleach/ammonia/solvent/paint thinner/acetone, fireworks,
   projectiles, load-bearing claims) absent from the same fields.

   Restriction rules fire on a negation (`no`, `without`, `avoid`, `not`,
   `never`, `nothing`, `don't`) within two words of the topic, so `no sharp
   tools`, `must not use sharp tools` and `no knives or scissors` all reach the
   one rule. They are still keyword rules: `no cutting implements` and
   `glue-free` are **not** detected, so a guide breaking those is shown
   unchecked.
- *Remixes*: the same idea rules, plus your restrictions, plus — for zero-buy an
  empty `additional_items` and a grammar that makes one ungrammatical; for
   remove-a-material the excluded word absent from `materials_used` and from the
   remix's prose; for simplify a `difficulty` that no longer says hard; and for
   every mode a title that may not be the original's own when the mode had
   something to change.

What remains **advisory only**, and is not guaranteed by any of the above:

- **Material availability.** The all-of rule is stronger than Sprint 4's, but it
  is still a spelling check: the same object under another name ("cling film" vs
  "plastic wrap") is rejected, and it cannot tell whether you really own three
  intact shoeboxes, whether "cardboard" means a cereal box or a sheet of foam
  board, or whether a claimed supply is the right size. Nothing checks that a
  returned guide is actually for the card you clicked — the UI shows which
  project it was built for so a mismatch is visible rather than silent.
- **Idea cards' material lists** are not inventory-checked at all; only the
  guide's are.
- **Real-world safety.** The blocklist is a keyword list, not a safety review.
  It cannot judge whether a step is safe *in your situation*, and it is blind to
  hazards nobody wrote down.
- **Structural soundness and buildability.** Nothing here verifies that a guide
  holds together, holds weight, or actually produces a working object — and the
  time estimate is the model's claim about its own plan, not a measurement.
- **Non-word restrictions.** "Must fit on my desk", "indoor only", "no mess",
  "child-safe" and anything else that isn't a keyword are prompt instructions
  with no check behind them, and the app names them as unchecked rather than
  implying they were met.
- **Prices.** For ₹100/₹500 budgets, nothing verifies that a purchase genuinely
  costs that much, because the reply carries no price at all.
- **Remix quality.** Each remix is checked for the one thing it promises, and
  for not re-breaking a restriction. Nothing checks that an adaptation is a
  *good* one, that **use-it-up** used those materials first, or that **simplify
  it** is actually easier to build.
- **Which idea a guide belongs to.** Nothing inspects a guide's contents to
  confirm it matches the card; the UI shows which project it was built for, and
  any remix retires the guide built for the pre-remix idea, so a mismatch is
  visible rather than silent.

Read the cards and the steps before you build anything.

## License

MIT — see [LICENSE](LICENSE). Everything runs on your own machine; there is no
account, no key and nothing to sign up for.