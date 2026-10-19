"""Constraint Kitchen - Sprint 5: constraints in, ideas out, Build Mode, Remix."""

import streamlit as st

from llm import (MODEL, checked_restrictions, generate_build_guide,
                 generate_ideas, remix_idea, status as llm_status,
                 unchecked_restrictions)

BUDGET_LABELS = {
    "zero": "₹0 — Use only what I have",
    "100": "Up to ₹100",
    "500": "Up to ₹500",
    "flexible": "Flexible",
}
TIME_CHOICES = ["15 minutes", "30 minutes", "60 minutes"]
# Constraint Remix: one press = one request against that card's idea.
REMIX_BUTTONS = (
    ("zero_buy", "Zero-buy"),
    ("use_it_up", "Use-it-up"),
    ("simplify", "Simplify it"),
    ("remove_material", "Remove a material"),
)
REMIX_LABELS = {
    "zero_buy": "Zero-buy",
    "use_it_up": "Use-it-up",
    "simplify": "Simplified",
    "remove_material": "Without a material",
}

# What this app can actually promise, and what it cannot. Stated in the UI and
# in the README because the honest version is the useful one: a word check that
# passed is evidence about those words and nothing else. Saying "compliant"
# after a JSON validated would be the whole app lying.
WHAT_IS_CHECKED = (
    "Checked: the JSON shape, the time limit, no purchases at ₹0, that a "
    "purchase is never something you already own, and the hard restrictions "
    "named above. On a build guide, also that every material is one you "
    "actually listed."
)
WHAT_IS_NOT_CHECKED = (
    "Not checked: that a project is buildable or safe beyond a keyword list, "
    "that it is a good idea, that an idea card's material list is really "
    "yours, or that a purchase respects a ₹100/₹500 ceiling."
)


def restriction_note(restrictions):
    """Split the user's own words into what is enforced and what is not.

    Anything not in the checkable table - "must fit on my desk", "indoor
    only" - is named as unchecked rather than quietly treated as satisfied.
    """
    if not restrictions.strip():
        return ""
    parts = []
    checked = checked_restrictions(restrictions)
    if checked:
        parts.append("Checked for you: " + ", ".join(checked) + ".")
    unchecked = unchecked_restrictions(restrictions)
    if unchecked:
        parts.append("NOT checked — your call: " + "; ".join(unchecked) + ".")
    return " ".join(parts)


def budget_note(budget):
    """One honest line about what the budget control does.

    Only ₹0 is enforced. The schema carries no price field at all, so ₹100
    and ₹500 are an instruction to the model and there is no number in the
    reply to compare against a ceiling. Adding a model-estimated price would
    manufacture the exact evidence the app is supposed to be without, so the
    ceiling is reported as unchecked instead of faked.
    """
    if budget == "zero":
        return "₹0 is enforced — any purchase at all is rejected."
    if budget == "flexible":
        return "Flexible adds no ceiling, and none is checked."
    return (f"{BUDGET_LABELS[budget]} is NOT checked — the app has no prices, "
            "so nothing confirms the total stays under it.")

st.markdown(
    """<style>
    .stApp { background-color: #fdf8f1; }
    h1 { letter-spacing: -0.02em; color: #3b2f27; }
    p, label { color: #5a4a3d; }
    .stTextArea textarea { border-radius: 12px; border-color: #e0d0bd; padding: 12px; }
    </style>""",
    unsafe_allow_html=True,
)

# Defaults are seeded once per session, never overwritten on rerun.
st.session_state.setdefault("materials", "")
st.session_state.setdefault("time", "30 minutes")
st.session_state.setdefault("budget", "zero")
st.session_state.setdefault("skill", "Beginner")
st.session_state.setdefault("restrictions", "")
st.session_state.setdefault("preference", "Surprise me")
st.session_state.setdefault("ideas", None)
st.session_state.setdefault("ideas_for", None)
st.session_state.setdefault("guide", None)
st.session_state.setdefault("guide_ctx", None)  # (idea title, input snapshot)
st.session_state.setdefault("remix", None)      # {"mode": ..., "idea": {...}}
st.session_state.setdefault("remix_for", None)  # which card's idea was remixed

st.title("Constraint Kitchen 🍳")
st.subheader("Turn what you have into what you can make.")
st.caption(
    "No shopping list. No fancy equipment. Just your materials, "
    "your constraints, and a little creativity."
)

# Local AI status is probed at most once per session; the app works either way.
if "ollama_status" not in st.session_state:
    st.session_state["ollama_status"] = llm_status()

with st.container(border=True):
    st.markdown("**A. Materials**")
    st.text_area(
        "What's lying around?",
        key="materials",
        placeholder="Cardboard, string, tape, old notebooks, wooden sticks...",
        height=140,
    )

left, right = st.columns(2)
with left:
    st.markdown("**B. Available time**")
    st.selectbox(
        "Available time",
        TIME_CHOICES,
        key="time",
        label_visibility="collapsed",
    )
with right:
    st.markdown("**C. Budget**")
    st.selectbox(
        "Budget",
        list(BUDGET_LABELS),
        format_func=lambda value: BUDGET_LABELS[value],
        key="budget",
        label_visibility="collapsed",
    )
    st.caption(budget_note(st.session_state["budget"]))

left, right = st.columns(2)
with left:
    st.markdown("**D. Skill level**")
    st.selectbox(
        "Skill level",
        ["Beginner", "Intermediate", "Advanced"],
        key="skill",
        label_visibility="collapsed",
    )
with right:
    st.markdown("**E. Optional restrictions**")
    st.text_area(
        "Anything else we should know?",
        key="restrictions",
        placeholder="No sharp tools, no glue, must fit on my desk...",
        height=80,
        label_visibility="collapsed",
    )
    note = restriction_note(st.session_state["restrictions"])
    if note:
        st.caption(note)

with st.container(border=True):
    st.markdown("**F. Project preference**")
    st.selectbox(
        "Project preference",
        ["Surprise me", "Useful", "Decorative", "Fun", "Educational"],
        key="preference",
        label_visibility="collapsed",
    )

INPUT_KEYS = ("materials", "time", "budget", "skill", "restrictions", "preference")


def current_inputs():
    return tuple(st.session_state[key] for key in INPUT_KEYS)


def user_constraints():
    """The same six inputs for every generation call - ideas and guides alike."""
    return {
        "materials": st.session_state["materials"].strip(),
        "time_minutes": int(st.session_state["time"].split()[0]),
        "budget": st.session_state["budget"],
        "skill_level": st.session_state["skill"],
        "restrictions": st.session_state["restrictions"].strip(),
        "preference": st.session_state["preference"],
    }


def ollama_not_ready_error():
    """One wording for both buttons, so the fix is identical everywhere."""
    return (f"{st.session_state['ollama_status'][1]} Start it with "
            f"`ollama serve`, then fetch the model once with "
            f"`ollama pull {MODEL}`. Everything else on this page keeps "
            f"working without it.")


def run_build(idea, retry_hint="Press Build This again to retry."):
    """One Build This press = exactly one request for that one idea."""
    # Clear first: a failed build must never leave the previous guide on
    # screen looking like it belongs to this idea.
    st.session_state["guide"] = None
    st.session_state["guide_ctx"] = None
    st.session_state["ollama_status"] = llm_status()
    if not st.session_state["ollama_status"][0]:
        st.error(ollama_not_ready_error())
        return
    with st.spinner(f"Writing the steps for {idea['title']} — one pass, "
                    f"give it a moment..."):
        result = generate_build_guide(idea=idea, **user_constraints())
    if result.ok:
        st.session_state["guide"] = result.guide
        # Snapshot the constraints the IDEAS were made under, not the ones on
        # screen now: building from a stale card must not produce a guide that
        # looks current.
        st.session_state["guide_ctx"] = (
            idea["title"], st.session_state["ideas_for"]
        )
    else:
        st.error(f"{result.error} {retry_hint}")


def run_remix(number, idea, mode, exclude=""):
    """One remix press = exactly one request, against this card's idea only."""
    st.session_state["remix_for"] = number
    # A failed press must never leave an older remix looking current.
    st.session_state["remix"] = None
    st.session_state["ollama_status"] = llm_status()
    if not st.session_state["ollama_status"][0]:
        st.error(ollama_not_ready_error())
        return
    with st.spinner(f"Adapting {idea['title']} — one pass, give it a moment..."):
        result = remix_idea(idea=idea, mode=mode, exclude=exclude,
                            **user_constraints())
    if not result.ok:
        # No repair call, no partial remix: the original idea stays exactly as
        # it was, and so does any guide already on screen.
        st.error(f"{result.error} The original idea is unchanged — press the "
                 f"remix button again to retry.")
        return
    st.session_state["remix"] = {"mode": mode, "idea": result.ideas[0]}
    # Any guide on screen was built for the pre-remix idea. Retire it rather
    # than let it look like it belongs to the new version.
    if st.session_state["guide"] is not None:
        st.session_state["guide"] = None
        st.session_state["guide_ctx"] = None


def show_idea(idea, heading="", note=""):
    """One idea's body - the original card and its remix share it exactly."""
    if heading:
        st.markdown(heading)
    st.caption(f"{idea['estimated_minutes']} min · {idea['difficulty']}")
    st.write(idea["description"])
    st.markdown(f"**Uses:** {', '.join(idea['materials_used']) or '—'}")
    if idea["additional_items"]:
        st.markdown(f"**Also needs:** {', '.join(idea['additional_items'])}")
    st.markdown(f"**Why it fits:** {idea['why_it_fits']}")
    if note:
        st.caption(note)


if st.button("Find My Project ✨", type="primary", width="stretch"):
    # Clear first: a failed attempt must never leave older ideas on screen.
    st.session_state["ideas"] = None
    st.session_state["ideas_for"] = None
    # A new set of ideas retires any old guide and any old remix: card 1 may
    # now be a different project, and neither a guide nor a remix may appear to
    # belong to it.
    st.session_state["guide"] = None
    st.session_state["guide_ctx"] = None
    st.session_state["remix"] = None
    st.session_state["remix_for"] = None
    if not st.session_state["materials"].strip():
        st.warning(
            "Start with whatever's in front of you — even a handful of scraps "
            "is plenty to work with."
        )
    else:
        st.session_state["ollama_status"] = llm_status()
        if not st.session_state["ollama_status"][0]:
            st.error(ollama_not_ready_error())
        else:
            with st.spinner(f"{MODEL} is thinking — the first click loads the "
                            f"model, so give it a moment..."):
                result = generate_ideas(**user_constraints())
            if result.ok:
                st.session_state["ideas"] = result.ideas
                st.session_state["ideas_for"] = current_inputs()
            else:
                st.error(f"{result.error} Click Find My Project again to retry.")

ideas = st.session_state["ideas"]
if ideas:
    st.subheader("Three ideas from your materials")
    st.caption(f"{WHAT_IS_CHECKED} {WHAT_IS_NOT_CHECKED}")
    if st.session_state["ideas_for"] != current_inputs():
        st.caption(
            "These predate your last edit — press Find My Project ✨ again."
        )
    for number, idea in enumerate(ideas, start=1):
        with st.container(border=True):
            show_idea(idea, heading=f"**{number}. {idea['title']}**")
            # One button and one key per card: card 2 can only ever build card 2.
            if st.button("Build This →", key=f"build_{number}",
                          width="stretch"):
                run_build(idea)

            # Constraint Remix: four compact actions on this card's own idea.
            st.caption("♻️ Remix — one request per press, this idea only:")
            pressed, excluded = None, ""
            for column, (mode, label) in zip(st.columns(4), REMIX_BUTTONS):
                with column:
                    if mode == "remove_material":
                        excluded = st.text_input(
                            "Material to remove", key=f"exclude_{number}",
                            placeholder="e.g. tape", label_visibility="collapsed",
                        )
                    if st.button(label, key=f"remix_{mode}_{number}",
                                 width="stretch"):
                        pressed = mode
            st.caption(
                "Use-it-up asks the model to prioritise those materials — no "
                "check can confirm it did, so read the list. Every other mode "
                "is checked against its own rule only; whether the result is "
                "really better is not something a word check can tell you."
            )
            if pressed is not None:
                run_remix(number, idea, pressed, excluded)

            remix = st.session_state["remix"]
            if remix and st.session_state["remix_for"] == number:
                st.markdown(
                    f"♻️ **{REMIX_LABELS[remix['mode']]} remix of idea "
                    f"{number}**"
                )
                show_idea(
                    remix["idea"],
                    heading=f"**{remix['idea']['title']}**",
                    note="This is the adapted version. The original is "
                         "unchanged above it, and any build guide on this page "
                         "was written for the original — press Build This Remix "
                         "below for steps that match this one.",
                )
                if st.button("Build This Remix →", key=f"build_remix_{number}",
                             width="stretch"):
                    run_build(
                        remix["idea"],
                        retry_hint="Press Build This Remix again to retry.",
                    )

guide = st.session_state["guide"]
if guide:
    st.subheader("BUILD MODE — step by step")
    st.caption(f"{WHAT_IS_CHECKED} {WHAT_IS_NOT_CHECKED}")
    if st.session_state["guide_ctx"][1] != current_inputs():
        st.caption("These steps predate your last edit — press Build This "
                   "again to refresh them.")
    st.markdown(f"### {guide['title']}")
    st.caption(f"For: {st.session_state['guide_ctx'][0]}")
    if guide["purpose"]:
        st.write(guide["purpose"])
    st.caption(f"About {guide['estimated_minutes']} min")
    st.markdown(f"**Materials:** {', '.join(guide['materials_used']) or '—'}")
    st.markdown(
        f"**Tools needed:** "
        f"{', '.join(guide['tools_needed']) or 'None required'}"
    )
    for number, step in enumerate(guide["steps"], start=1):
        st.markdown(f"**{number}. {step['title']}**")
        st.markdown(step["instruction"])
        if step["tip"]:
            st.caption(f"Tip: {step['tip']}")
    if guide["safety_notes"]:
        st.markdown("**Safety**")
        for note in guide["safety_notes"]:
            st.markdown(f"- {note}")

st.caption(f"Local AI — {st.session_state['ollama_status'][1]}")