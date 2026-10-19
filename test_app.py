"""Streamlit behaviour tests for the Idea Engine.

These drive app.py through streamlit.testing.v1.AppTest against the same
loopback Ollama stub the llm tests use, so no real model or Ollama install is
involved. They verify wiring, session state and display - not model quality.
"""

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import llm
from test_llm import (Stub, StubServer, generate_response, make_guide,
                      make_idea, make_three_ideas)

APP = Path(__file__).parent / "app.py"
TITLES = ("Cardboard pen stand", "Desk tidying tray", "Book mark tassel")
BUILD_BUTTON = "Build This →"
REMIX_BUTTON = "Build This Remix →"
REMIX_ACTIONS = {"Zero-buy": "zero_buy", "Use-it-up": "use_it_up",
                 "Simplify it": "simplify",
                 "Remove a material": "remove_material"}


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.stub = StubServer()
        self.at = AppTest.from_file(str(APP), default_timeout=30).run()

    def tearDown(self):
        self.stub.close()

    def fill(self, **overrides):
        values = {"materials": "cardboard, tape, string"}
        values.update(overrides)
        for key, value in values.items():
            self.at.session_state[key] = value
        return self.at.run()

    def click_generate(self):
        return self.at.button[0].click().run()

    def build_buttons(self):
        return [button for button in self.at.button
                if button.label == BUILD_BUTTON]

    def click_build(self, number):
        """Click card `number`'s own Build This button (1-based, like the card)."""
        button = self.build_buttons()[number - 1]
        self.assertEqual(button.key, f"build_{number}",
                         "each card must own its button")
        return button.click().run()

    def screen(self):
        """Every piece of visible text, for 'is this idea on screen' assertions."""
        chunks = []
        for elements in (self.at.markdown, self.at.text, self.at.subheader,
                         self.at.caption, self.at.error):
            chunks.extend(element.value for element in elements)
        chunks.extend(button.label for button in self.at.button)
        return "\n".join(chunks)

    def posts(self):
        return Stub.generate_posts()

    def show_ideas(self, **overrides):
        """Reach the state where three cards are on screen."""
        self.fill(**overrides)
        return self.click_generate()

    # --- the controls and defaults Sprint 1 shipped -------------------------

    def test_controls_and_defaults_unchanged(self):
        self.assertEqual(self.at.title[0].value, "Constraint Kitchen 🍳")
        self.assertEqual(self.at.button[0].label, "Find My Project ✨")
        self.assertEqual(self.at.session_state["materials"], "")
        self.assertEqual(self.at.session_state["time"], "30 minutes")
        self.assertEqual(self.at.session_state["budget"], "zero")
        self.assertEqual(self.at.session_state["skill"], "Beginner")
        self.assertEqual(self.at.session_state["restrictions"], "")
        self.assertEqual(self.at.session_state["preference"], "Surprise me")
        self.assertEqual(len(self.at.text_area), 2)  # materials + restrictions
        self.assertEqual(len(self.at.selectbox), 4)  # time, budget, skill, preference

    # --- the button ---------------------------------------------------------

    def test_empty_materials_warns_without_generating(self):
        self.assertTrue(self.click_generate().warning)
        self.assertEqual(self.posts(), [])
        self.assertEqual(self.at.session_state["ideas"], None)

    def test_successful_generation_shows_three_idea_cards(self):
        before = len(self.at.container)
        self.fill()
        self.click_generate()
        self.assertEqual(self.at.error, [], "a good run must show no error")
        screen = self.screen()
        for title in TITLES:
            self.assertIn(title, screen)
        self.assertIn("20 min · easy", screen)  # estimated time + difficulty
        self.assertIn("**Uses:** cardboard tubes, tape", screen)
        self.assertIn("**Why it fits:**", screen)
        self.assertEqual(len(self.at.container) - before, 3, "one card per idea")
        self.assertEqual(len(self.posts()), 1)

    def test_additional_items_only_shown_when_needed(self):
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["a strip of felt", "glue dots"]
        Stub.generate = generate_response(json.dumps({"ideas": ideas}))
        self.fill(budget="100")
        self.click_generate()
        screen = self.screen()
        self.assertIn("**Also needs:** a strip of felt, glue dots", screen)
        self.assertEqual(screen.count("**Also needs:**"), 1)

    def test_results_survive_a_plain_rerun(self):
        self.fill()
        self.click_generate()
        self.assertEqual(len(self.posts()), 1)
        self.at.run()  # Streamlit re-executes the script on any interaction
        for title in TITLES:
            self.assertIn(title, self.screen())
        self.assertEqual(len(self.posts()), 1, "a rerun must not regenerate")

    def test_changing_inputs_does_not_regenerate(self):
        self.fill()
        self.click_generate()
        before = len(self.posts())
        self.at.session_state["time"] = "60 minutes"
        self.at.session_state["preference"] = "Decorative"
        self.at.session_state["restrictions"] = "no glue"
        self.at.run()
        self.assertEqual(len(self.posts()), before)
        for title in TITLES:
            self.assertIn(title, self.screen())

    def test_failure_clears_stale_results(self):
        self.fill()
        self.click_generate()
        self.assertIn(TITLES[0], self.screen())
        Stub.generate = generate_response(text="I am not going to answer that.")
        self.click_generate()
        screen = self.screen()
        self.assertTrue(self.at.error)
        self.assertNotIn(TITLES[0], screen)
        self.assertNotIn(TITLES[1], screen)
        self.assertEqual(self.at.session_state["ideas"], None)

    def test_validation_failure_shows_reason_not_json(self):
        ideas = make_three_ideas()
        ideas[0]["estimated_minutes"] = 45  # breaks the 30-minute limit
        Stub.generate = generate_response(json.dumps({"ideas": ideas}))
        self.fill()
        self.click_generate()
        self.assertTrue(self.at.error)
        self.assertIn("only 30 are available", self.at.error[0].value)
        self.assertNotIn("estimated_minutes\": 45", self.screen())

    # --- Ollama is optional -------------------------------------------------

    def test_missing_model_explains_how_to_install(self):
        Stub.body = {"models": []}
        self.fill()
        self.click_generate()
        self.assertIn("ollama pull gemma3:1b", self.at.error[0].value)
        self.assertIn("ollama serve", self.at.error[0].value)
        self.assertEqual(self.posts(), [], "nothing is generated without a model")
        self.assertEqual(len(self.at.text_area), 2)  # inputs still usable
        self.assertEqual(self.at.session_state["materials"], "cardboard, tape, string")

    def test_ollama_down_leaves_the_form_usable(self):
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        self.fill()
        self.click_generate()
        self.assertTrue(self.at.error)
        self.assertIn("Ollama", self.at.error[0].value)
        self.assertEqual(len(self.at.selectbox), 4)
        self.assertEqual(len(self.at.text_area), 2)
        self.assertFalse(self.at.exception)
        self.assertEqual(self.posts(), [])


class BuildModeTest(AppTestCase):
    """Build Mode: one click on one card, one request, one guide."""

    def guide_for(self, label, **overrides):
        guide = make_guide(**overrides)
        guide["title"] = f"Guide for {label}"
        return guide

    def test_every_card_has_its_own_build_action(self):
        self.show_ideas()
        buttons = self.build_buttons()
        self.assertEqual(len(buttons), 3, "one Build This per idea card")
        self.assertEqual([button.key for button in buttons],
                         ["build_1", "build_2", "build_3"])
        self.assertEqual(self.at.button[0].label, "Find My Project ✨")
        self.assertEqual(len(self.posts()), 1, "cards alone generate nothing")

    def test_build_this_shows_that_ideas_guide(self):
        self.show_ideas()
        Stub.generate = generate_response(
            json.dumps(self.guide_for("card 1"))
        )
        self.click_build(1)
        self.assertEqual(self.at.error, [], "a good build must show no error")
        screen = self.screen()
        self.assertIn("BUILD MODE", screen)
        self.assertIn("### Guide for card 1", screen)
        self.assertIn("A small holder that keeps pens upright on a desk.", screen)
        self.assertIn("About 20 min", screen)
        self.assertIn("**Materials:** cardboard, tape", screen)
        self.assertIn("**Tools needed:** None required", screen)
        self.assertIn("**1. Flatten the tubes**", screen)
        self.assertIn("Tip: Two passes of tape hold much better than one.", screen)
        self.assertNotIn("Guide for card 2", screen)
        self.assertEqual(len(self.posts()), 2, "one request per action, total")
        self.assertEqual(self.at.session_state["guide_ctx"][0], TITLES[0])

    def test_guide_survives_reruns_and_input_edits(self):
        self.show_ideas()
        Stub.generate = generate_response(
            json.dumps(self.guide_for("card 2"))
        )
        self.click_build(2)
        posts = len(self.posts())
        self.at.run()  # a bare rerun keeps the guide
        self.assertIn("### Guide for card 2", self.screen())
        # A real widget interaction, the kind a browser user makes.
        self.at.selectbox[0].set_value("60 minutes").run()
        self.at.run()
        self.assertEqual(len(self.posts()), posts, "a rerun must not regenerate")
        self.assertIn("### Guide for card 2", self.screen())
        self.assertIn("These steps predate your last edit", self.screen())

    def test_no_build_happens_until_the_button_is_clicked(self):
        self.show_ideas()
        self.assertEqual(self.at.session_state["guide"], None)
        self.assertIsNone(self.at.session_state["guide_ctx"])
        self.assertEqual(len(self.posts()), 1)
        self.at.run()  # page load must be free
        self.assertEqual(len(self.posts()), 1)
        self.assertNotIn("BUILD MODE", self.screen())

    def test_selecting_another_idea_replaces_the_guide(self):
        self.show_ideas()
        Stub.generate = generate_response(
            json.dumps(self.guide_for("card 1"))
        )
        self.click_build(1)
        self.assertIn("### Guide for card 1", self.screen())
        Stub.generate = generate_response(
            json.dumps(self.guide_for("card 3"))
        )
        self.click_build(3)
        screen = self.screen()
        self.assertIn("### Guide for card 3", screen)
        self.assertNotIn("### Guide for card 1", screen)
        self.assertEqual(self.at.session_state["guide_ctx"][0], TITLES[2])
        self.assertEqual(len(self.posts()), 3)
        prompt = self.posts()[2]["prompt"]
        self.assertIn(TITLES[2], prompt)
        self.assertNotIn(TITLES[0], prompt, "only the clicked idea is sent")

    def test_new_ideas_retire_the_old_guide(self):
        self.show_ideas()
        Stub.generate = generate_response(
            json.dumps(self.guide_for("card 1"))
        )
        self.click_build(1)
        self.assertIn("### Guide for card 1", self.screen())
        Stub.generate = generate_response(json.dumps({"ideas": make_three_ideas()}))
        self.click_generate()
        self.assertNotIn("BUILD MODE", self.screen())
        self.assertEqual(self.at.session_state["guide"], None)

    def test_failed_build_clears_the_guide_and_says_why(self):
        self.show_ideas()
        Stub.generate = generate_response(
            json.dumps(self.guide_for("card 1"))
        )
        self.click_build(1)
        self.assertIn("### Guide for card 1", self.screen())
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Heat it over a blowtorch until soft."
        Stub.generate = generate_response(json.dumps(guide))
        self.click_build(1)
        self.assertTrue(self.at.error)
        self.assertIn("unsafe", self.at.error[0].value)
        self.assertIn("Press Build This again to retry.", self.at.error[0].value)
        self.assertNotIn("### Guide for card 1", self.screen())
        self.assertEqual(self.at.session_state["guide"], None)
        for title in TITLES:
            self.assertIn(title, self.screen(), "the cards survive a bad build")

    def test_validation_failure_is_human_readable(self):
        self.show_ideas()
        guide = make_guide()
        guide["estimated_minutes"] = 45  # breaks the 30-minute limit
        Stub.generate = generate_response(json.dumps(guide))
        self.click_build(1)
        self.assertTrue(self.at.error)
        self.assertIn("only 30 are available", self.at.error[0].value)
        self.assertNotIn("estimated_minutes\": 45", self.screen())

    def test_tools_line_only_lists_real_tools(self):
        self.show_ideas(budget="100")
        guide = self.guide_for("card 1", tools=[None, "a stapler"])
        Stub.generate = generate_response(json.dumps(guide))
        self.click_build(1)
        self.assertIn("**Tools needed:** a stapler", self.screen())

    def test_safety_notes_are_shown_when_present(self):
        self.show_ideas()
        guide = self.guide_for("card 1")
        guide["safety_notes"] = [None, "Sand the cut edge before you touch it"]
        Stub.generate = generate_response(json.dumps(guide))
        self.click_build(1)
        screen = self.screen()
        self.assertIn("**Safety**", screen)
        self.assertIn("- Sand the cut edge before you touch it", screen)
        self.assertNotIn("- None", screen, "a null note must be dropped")
        self.assertEqual(self.at.session_state["guide"]["safety_notes"],
                         ["Sand the cut edge before you touch it"])

    def test_build_without_a_model_explains_how_to_install(self):
        self.show_ideas()
        Stub.body = {"models": []}
        self.click_build(1)
        self.assertIn("ollama pull gemma3:1b", self.at.error[0].value)
        self.assertIn("ollama serve", self.at.error[0].value)
        self.assertEqual(len(self.posts()), 1, "nothing generated without a model")
        self.assertEqual(len(self.build_buttons()), 3, "retry is still possible")

    def test_ollama_going_down_leaves_the_cards_usable(self):
        self.show_ideas()
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        self.click_build(1)
        self.assertTrue(self.at.error)
        self.assertIn("Ollama", self.at.error[0].value)
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.posts()), 1)
        for title in TITLES:
            self.assertIn(title, self.screen())


class RemixTest(AppTestCase):
    """Constraint Remix: one press, one request, on the card you pressed."""

    def remix_buttons(self, label):
        return [button for button in self.at.button if button.label == label]

    def click_remix(self, label, number):
        """Press card `number`'s own remix button (1-based, like the card)."""
        buttons = self.remix_buttons(label)
        self.assertEqual(len(buttons), 3, f"one {label} per idea card")
        mode = REMIX_ACTIONS[label]
        self.assertEqual(buttons[number - 1].key, f"remix_{mode}_{number}",
                         "each card must own its remix button")
        return buttons[number - 1].click().run()

    def answer_remix(self, **overrides):
        """A well-adapted idea: no purchases, and nothing about tape in it."""
        adapted = make_idea("Cardboard tray, no tape", 15)
        adapted["description"] = "Fold cardboard scraps into a shallow tray."
        adapted["why_it_fits"] = "Uses only what is already on the shelf."
        adapted.update(overrides)
        Stub.generate = generate_response(json.dumps({"idea": adapted}))
        return adapted

    def test_every_card_offers_the_four_remix_actions(self):
        self.show_ideas()
        for label, mode in REMIX_ACTIONS.items():
            keys = [button.key for button in self.remix_buttons(label)]
            self.assertEqual(keys, [f"remix_{mode}_{n}" for n in (1, 2, 3)])
        self.assertEqual(len(self.posts()), 1, "cards alone generate nothing")

    def test_cards_alone_never_remix(self):
        self.show_ideas()
        self.at.run()
        self.at.text_input[0].set_value("tape").run()
        self.assertEqual(len(self.posts()), 1, "no request without a press")

    def test_a_remix_appears_under_its_own_card(self):
        self.show_ideas()
        adapted = self.answer_remix()
        self.click_remix("Zero-buy", 2)
        self.assertEqual(self.at.error, [], "a good remix must show no error")
        screen = self.screen()
        self.assertIn("Zero-buy remix of idea 2", screen)
        self.assertIn(adapted["title"], screen)
        self.assertIn("Roll two cardboard tubes and tape them into a pen stand",
                      screen)
        for title in TITLES:
            self.assertIn(title, screen, "every original card survives")
        self.assertEqual(screen.count("adapted"), 1, "one remix at a time")
        self.assertEqual(len(self.posts()), 2, "one request per press")
        self.assertEqual(self.at.session_state["remix_for"], 2)
        self.assertEqual(self.at.session_state["remix"]["mode"], "zero_buy")

    def test_a_remix_acts_on_the_pressed_card_only(self):
        self.show_ideas()
        self.answer_remix()
        self.click_remix("Simplify it", 3)
        prompt = self.posts()[-1]["prompt"]
        self.assertIn(TITLES[2], prompt)
        self.assertNotIn(TITLES[0], prompt)
        self.assertNotIn(TITLES[1], prompt)

    def test_the_exclude_box_is_per_card_and_generates_nothing(self):
        self.show_ideas()
        self.at.text_input[1].set_value("tape").run()
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(self.at.text_input[1].key, "exclude_2")

    def test_remove_material_without_a_material_says_so_and_posts_nothing(self):
        self.show_ideas()
        self.click_remix("Remove a material", 1)
        self.assertIn("Name the material to remove", self.at.error[0].value)
        self.assertEqual(self.at.session_state["remix"], None)
        self.assertEqual(len(self.posts()), 1)
        self.assertIn(TITLES[0], self.screen(), "the original is untouched")

    def test_remove_material_sends_what_the_user_typed(self):
        self.show_ideas()
        self.answer_remix(materials_used=["cardboard tubes", "string"])
        self.at.text_input[0].set_value("tape").run()
        self.click_remix("Remove a material", 1)
        prompt = self.posts()[-1]["prompt"]
        self.assertIn("without this material: tape", prompt)
        self.assertEqual(self.at.session_state["remix"]["mode"],
                         "remove_material")

    def test_a_refused_remix_keeps_the_original_and_the_guide(self):
        self.show_ideas(budget="100")   # so the zero-buy gate is what refuses
        Stub.generate = generate_response(json.dumps(make_guide()))
        self.click_build(1)
        self.assertIn("### Cardboard pen stand, step by step", self.screen())
        Stub.generate = generate_response(
            json.dumps({"idea": make_idea("Felt tray", 15,
                                          extra=["a strip of felt"])}))
        self.click_remix("Zero-buy", 1)
        self.assertIn("still lists things to buy", self.at.error[0].value)
        self.assertIn("The original idea is unchanged", self.at.error[0].value)
        self.assertEqual(self.at.session_state["remix"], None)
        for title in TITLES:
            self.assertIn(title, self.screen(), "the cards survive")
        self.assertIn("### Cardboard pen stand, step by step", self.screen(),
                      "an existing guide must survive a failed remix")
        self.assertEqual(len(self.posts()), 3, "no repair call after a refusal")

    def test_a_successful_remix_retires_the_guide_it_was_built_for(self):
        self.show_ideas()
        Stub.generate = generate_response(json.dumps(make_guide()))
        self.click_build(1)
        self.assertIn("### Cardboard pen stand, step by step", self.screen())
        self.answer_remix()
        self.click_remix("Zero-buy", 1)
        self.assertNotIn("BUILD MODE", self.screen())
        self.assertEqual(self.at.session_state["guide"], None)
        self.assertIsNone(self.at.session_state["guide_ctx"])
        self.assertIn("was written for the original", self.screen())

    def test_a_zero_buy_remix_never_renders_with_a_purchase_list(self):
        self.show_ideas(budget="100")
        self.answer_remix(additional_items=["a strip of felt"])
        self.click_remix("Zero-buy", 1)
        self.assertTrue(self.at.error)
        self.assertIn("still lists things to buy", self.at.error[0].value)
        self.assertNotIn("**Also needs:** a strip of felt", self.screen())
        self.assertIsNone(self.at.session_state["remix"])

    def test_the_remixed_idea_can_be_built_explicitly(self):
        self.show_ideas()
        adapted = self.answer_remix()
        self.click_remix("Zero-buy", 1)
        Stub.generate = generate_response(json.dumps(make_guide()))
        button = [b for b in self.at.button if b.label == REMIX_BUTTON][0]
        self.assertEqual(button.key, "build_remix_1")
        button.click().run()
        self.assertIn("### Cardboard pen stand, step by step", self.screen())
        self.assertEqual(self.at.session_state["guide_ctx"][0],
                         adapted["title"])
        prompt = self.posts()[-1]["prompt"]
        self.assertIn(adapted["title"], prompt)
        self.assertNotIn(TITLES[0], prompt)

    def test_new_ideas_retire_the_old_remix(self):
        self.show_ideas()
        self.answer_remix()
        self.click_remix("Zero-buy", 1)
        self.assertIn("Cardboard tray, no tape", self.screen())
        Stub.generate = generate_response(json.dumps({"ideas": make_three_ideas()}))
        self.click_generate()
        self.assertNotIn("Cardboard tray, no tape", self.screen())
        self.assertEqual(self.at.session_state["remix"], None)
        self.assertIsNone(self.at.session_state["remix_for"])

    def test_the_use_it_up_caveat_is_visible(self):
        self.show_ideas()
        self.assertIn("no check can confirm it did", self.screen())

    def test_remix_without_a_model_explains_how_to_install(self):
        self.show_ideas()
        Stub.body = {"models": []}
        self.click_remix("Simplify it", 1)
        self.assertIn("ollama pull gemma3:1b", self.at.error[0].value)
        self.assertEqual(len(self.posts()), 1, "nothing generated without it")
        self.assertEqual(self.at.session_state["remix"], None)

    def test_ollama_going_down_leaves_the_cards_usable(self):
        self.show_ideas()
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        self.click_remix("Zero-buy", 1)
        self.assertIn("Ollama", self.at.error[0].value)
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.posts()), 1)
        for title in TITLES:
            self.assertIn(title, self.screen())


class HonestyTest(AppTestCase):
    """Sprint 5: what the app checks is stated, and what it cannot check is
    never presented as satisfied."""

    def test_the_restriction_box_says_what_is_and_is_not_checked(self):
        self.fill(restrictions="no sharp tools, no glue, must fit on my desk")
        screen = self.screen()
        self.assertIn("Checked for you: no sharp tools, no glue.", screen)
        self.assertIn("NOT checked", screen)
        self.assertIn("must fit on my desk", screen)
        self.assertEqual(self.posts(), [], "reading the box generates nothing")

    def test_an_uncheckable_restriction_is_never_shown_as_satisfied(self):
        # "indoor only" cannot be word-checked, so the app may not imply that
        # it was. It has to be named as out of reach.
        self.fill(restrictions="indoor only")
        self.assertNotIn("Checked for you", self.screen())
        self.assertIn("NOT checked", self.screen())
        self.assertIn("indoor only", self.screen())

    def test_no_restrictions_means_no_claim_in_either_direction(self):
        self.assertNotIn("Checked for you", self.screen())
        self.assertNotIn("NOT checked", self.screen())

    def test_each_budget_says_what_it_can_and_cannot_do(self):
        self.assertIn("₹0 is enforced", self.screen())
        self.fill(budget="100")
        self.assertIn("NOT checked", self.screen())
        self.assertIn("no prices", self.screen())
        self.fill(budget="500")
        self.assertIn("₹500 is NOT checked", self.screen())
        self.fill(budget="flexible")
        self.assertIn("adds no ceiling", self.screen())

    def test_results_state_both_what_is_checked_and_what_is_not(self):
        self.show_ideas()
        screen = self.screen()
        self.assertIn("Checked: the JSON shape", screen)
        self.assertIn("Not checked:", screen)
        self.assertIn("₹100/₹500 ceiling", screen)
        self.assertNotIn("all inside your constraints", screen,
                         "that headline overclaimed what is verified")

    def test_a_refused_restricted_guide_is_hidden_and_says_what_tripped_it(self):
        self.show_ideas(restrictions="no sharp tools")
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Cut the tape to length with scissors."
        Stub.generate = generate_response(json.dumps(guide))
        self.click_build(1)
        self.assertTrue(self.at.error)
        error = self.at.error[0].value
        self.assertIn("no sharp tools", error)
        self.assertIn("'cut'", error.lower())
        self.assertNotIn("BUILD MODE", self.screen())
        self.assertIsNone(self.at.session_state["guide"])
        for title in TITLES:
            self.assertIn(title, self.screen(), "the cards survive a refusal")
        self.assertEqual(len(self.posts()), 2, "no repair call after a refusal")

    def test_a_restricted_guide_that_obeys_is_shown(self):
        self.show_ideas(restrictions="no sharp tools")
        guide = make_guide()
        guide["title"] = "Guide for a no-sharp-tools build"
        Stub.generate = generate_response(json.dumps(guide))
        self.click_build(1)
        self.assertEqual(self.at.error, [], "an obeying guide must be shown")
        self.assertIn("### Guide for a no-sharp-tools build", self.screen())

    def test_a_banned_tool_in_the_materials_list_is_refused(self):
        # The user owns scissors here, so the inventory check passes and only
        # the restriction scan can catch this - exactly the Sprint 4 defect.
        self.show_ideas(restrictions="no sharp tools",
                        materials="cardboard, tape, string, scissors")
        Stub.generate = generate_response(
            json.dumps(make_guide(materials=["cardboard", "scissors"])))
        self.click_build(1)
        self.assertTrue(self.at.error)
        self.assertIn("no sharp tools", self.at.error[0].value)
        self.assertNotIn("BUILD MODE", self.screen())

    def test_an_owned_material_is_never_shown_as_a_purchase(self):
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["tape"]  # they already have tape
        Stub.generate = generate_response(json.dumps({"ideas": ideas}))
        self.fill(budget="100")
        self.click_generate()
        self.assertTrue(self.at.error)
        self.assertIn("already in your materials", self.at.error[0].value)
        self.assertNotIn("**Also needs:** tape", self.screen())
        self.assertIsNone(self.at.session_state["ideas"])

    def test_stale_results_are_marked_and_never_silently_rebuilt(self):
        self.show_ideas(restrictions="no sharp tools")
        Stub.generate = generate_response(json.dumps(make_guide()))
        self.click_build(1)
        posts = len(self.posts())
        self.at.session_state["restrictions"] = "no sharp tools, no glue"
        self.at.run()
        self.assertIn("These steps predate your last edit", self.screen())
        self.assertEqual(len(self.posts()), posts, "no silent regeneration")


if __name__ == "__main__":
    unittest.main()