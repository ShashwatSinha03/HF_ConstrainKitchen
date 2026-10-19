"""Tests for the Ollama probes and the one-shot idea generator.

Everything runs against a throwaway localhost stub, so no real Ollama install is
needed and nothing outside this machine is ever contacted. Real-model output is
NOT exercised here; these tests cover our request shape and our validation.
"""

import json
import re
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import llm
import test_real_model


def make_idea(title="Cardboard pen stand", minutes=20, extra=None):
    """One well-formed idea, ready to be JSON-serialised into the schema."""
    return {
        "title": title,
        "description": "Roll two cardboard tubes and tape them into a pen stand.",
        "materials_used": ["cardboard tubes", "tape"],
        "additional_items": [] if extra is None else extra,
        "estimated_minutes": minutes,
        "difficulty": "easy",
        "why_it_fits": "Everything comes from the tubes already on the shelf.",
    }


def tape_free(idea):
    """An idea that genuinely does without tape.

    Sprint 5: the remove-material gate now reads the prose as well as the
    materials list, because a remix can describe taping something together
    while leaving "tape" out of materials_used. These two fields are the only
    prose on the fixture, so replacing them is enough to make it honest.
    """
    idea = dict(idea, description="Fold cardboard scraps into a shallow tray.",
                why_it_fits="Uses only what is already on the shelf.")
    return idea


def make_three_ideas():
    return [
        make_idea("Cardboard pen stand", 20),
        make_idea("Desk tidying tray", 25),
        make_idea("Book mark tassel", 10),
    ]


# A pool of safe, boring actions: no hazard words, no restriction triggers, so a
# guide built from them passes every rule without any test having to excuse it.
_STEP_POOL = (
    {"title": "Flatten the tubes",
     "instruction": "Press each cardboard tube flat with the palm of your hand.",
     "tip": ""},
    {"title": "Roll one back",
     "instruction": "Roll the flattened cardboard back into a tube shape.",
     "tip": ""},
    {"title": "Tape the seam",
     "instruction": "Wrap tape around the seam to hold the roll in place.",
     "tip": "Two passes of tape hold much better than one."},
    {"title": "Add the base",
     "instruction": "Tape a cardboard circle to the bottom of the tube.",
     "tip": ""},
    {"title": "Shape the rim",
     "instruction": "Press the top edge flat so no paper sticks up.",
     "tip": ""},
    {"title": "Stand it up",
     "instruction": "Stand the tube on a flat surface and press the base down.",
     "tip": ""},
    {"title": "Test it",
     "instruction": "Drop a pen in to check the holder stands up on its own.",
     "tip": ""},
)


def make_guide(steps=4, minutes=20, tools=None, materials=None):
    """One well-formed build guide, ready to be JSON-serialised.

    `steps` is a count rather than a list so every shape test stays a one-liner;
    the pool repeats past its end and each step is copied, never shared.

    Sprint 5: the default materials are ones the tests' inventory really
    contains. They used to say "cardboard tubes" against an inventory of
    "cardboard, tape", which only passed while _covered_by_inventory accepted
    one shared word as proof - the fixture was leaning on the leniency it was
    meant to help test.
    """
    return {
        "title": "Cardboard pen stand, step by step",
        "purpose": "A small holder that keeps pens upright on a desk.",
        "estimated_minutes": minutes,
        "materials_used": ["cardboard", "tape"] if materials is None
        else materials,
        "tools_needed": [] if tools is None else tools,
        "steps": [dict(_STEP_POOL[i % len(_STEP_POOL)]) for i in range(steps)],
        "safety_notes": [],
    }


def generate_response(text=None):
    """A canned non-streamed /api/generate envelope."""
    if text is None:
        text = json.dumps({"ideas": make_three_ideas()})
    return {"model": llm.MODEL, "response": text, "done": True}


class Stub(BaseHTTPRequestHandler):
    """Serves /api/tags, /api/version and POST /api/generate; logs POST bodies."""

    body = {"models": [{"name": llm.MODEL}]}
    generate = {}
    generate_status = 200
    delay = 0.0
    posts = []
    # Optional replies consumed one per POST, in order - for the real-model
    # diagnostic, which asks a different question on every call.
    sequence = []

    @classmethod
    def reset(cls, body=None, generate=None, status=200):
        cls.body = {"models": [{"name": llm.MODEL}]} if body is None else body
        cls.generate = generate_response() if generate is None else generate
        cls.generate_status = status
        cls.delay = 0.0
        cls.posts = []
        cls.sequence = []

    @classmethod
    def generate_posts(cls):
        return [payload for path, payload in cls.posts if path == "/api/generate"]

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # client already gave up (timeout test)

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        self._send(Stub.body)

    def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        Stub.posts.append((self.path, json.loads(raw or b"{}")))
        if Stub.delay:
            time.sleep(Stub.delay)
        if Stub.sequence:
            self._send(Stub.sequence.pop(0))
            return
        self._send(Stub.generate, Stub.generate_status)

    def log_message(self, *args):
        pass  # keep test output clean


class StubServer:
    """Runs the stub on a loopback port and points llm.OLLAMA_URL at it."""

    def __init__(self):
        Stub.reset()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        self._real_url = llm.OLLAMA_URL
        host, port = self.server.server_address
        llm.OLLAMA_URL = f"http://{host}:{port}"

    def close(self):
        llm.OLLAMA_URL = self._real_url
        self.server.shutdown()
        self.server.server_close()


def threads_inside_llm():
    """Names of non-main threads with a live frame inside llm.py.

    This is the honest form of "did the call leave something running". The test
    stub serves requests on threads of its own, so counting threads would only
    measure the stub; walking the frame stack measures the thing under test.
    """
    frames = sys._current_frames()
    leaked = []
    for thread in threading.enumerate():
        if thread is threading.main_thread():
            continue
        frame = frames.get(thread.ident)
        while frame is not None:
            if frame.f_code.co_filename == llm.__file__:
                leaked.append(thread.name)
                break
            frame = frame.f_back
    return leaked


class LlmProbeTest(unittest.TestCase):
    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def test_is_running_against_live_stub(self):
        self.assertTrue(llm.is_running())

    def test_is_running_false_when_nothing_listens(self):
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        self.assertFalse(llm.is_running())

    def test_has_model_exact_and_latest_tag(self):
        Stub.body = {"models": [{"name": "gemma3:1b"}]}
        self.assertTrue(llm.has_model())
        Stub.body = {"models": [{"name": "gemma3:1b:latest"}]}
        self.assertTrue(llm.has_model())

    def test_has_model_false_when_absent_or_unrelated(self):
        Stub.body = {"models": [{"name": "llama3:8b"}]}
        self.assertFalse(llm.has_model())
        Stub.body = {"models": []}
        self.assertFalse(llm.has_model())

    def test_malformed_body_degrades_without_raising(self):
        for bad in ("oops", [1, 2], {"models": "oops"}, {"models": [{"name": 7}]}):
            Stub.body = bad
            self.assertFalse(llm.has_model(), f"should not raise on {bad!r}")

    def test_status_reports_model_missing(self):
        Stub.body = {"models": []}
        ready, message = llm.status()
        self.assertFalse(ready)
        self.assertIn("isn't pulled yet", message)

    def test_status_reports_ready(self):
        Stub.body = {"models": [{"name": "gemma3:1b"}]}
        ready, message = llm.status()
        self.assertTrue(ready)
        self.assertIn("ready", message)

    def test_status_reports_ollama_down(self):
        llm.OLLAMA_URL = "http://127.0.0.1:9"
        ready, message = llm.status()
        self.assertFalse(ready)
        self.assertIn("isn't running", message)


class GenerateTest(unittest.TestCase):
    def generate(self, **overrides):
        kwargs = {
            "materials": "cardboard, tape",
            "time_minutes": 30,
            "budget": "zero",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        return llm.generate_ideas(**kwargs)

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def test_valid_three_ideas_are_returned(self):
        result = self.generate()
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.ideas), 3)
        self.assertEqual(
            [idea["title"] for idea in result.ideas],
            ["Cardboard pen stand", "Desk tidying tray", "Book mark tassel"],
        )
        self.assertEqual(result.ideas[0]["estimated_minutes"], 20)
        self.assertIsInstance(result.ideas[1]["additional_items"], list)

    def test_request_carries_model_json_format_and_single_call(self):
        self.assertTrue(self.generate().ok)
        posts = Stub.generate_posts()
        self.assertEqual(len(posts), 1, "exactly one request per click")
        body = posts[0]
        self.assertEqual(body["model"], "gemma3:1b")
        # Sprint 4: `format` is Ollama's structured-outputs JSON Schema rather
        # than the bare "json" mode. Same intent - the reply must be JSON, and
        # every field is now required by the grammar - but measured 3/3 vs 0/3
        # against the real model. This run is at ₹0, so the grammar also makes
        # a purchase ungrammatical.
        ideas = body["format"]
        self.assertEqual(ideas["required"], ["ideas"])
        items = ideas["properties"]["ideas"]
        self.assertEqual((items["minItems"], items["maxItems"]), (3, 3))
        self.assertEqual(items["items"]["required"], list(llm._IDEA_FIELDS))
        extras = items["items"]["properties"]["additional_items"]
        self.assertEqual(extras["type"], "array")
        self.assertEqual(extras["maxItems"], 0, "at ₹0 a purchase is impossible")
        # Sprint 5: this used to assert `stream is False`. That asserted how the
        # transport was implemented, not a requirement. Streaming is now what
        # makes the 120s ceiling real - the deadline runs while reading Ollama's
        # NDJSON and gives up by closing the connection - so the honest
        # assertion is that we stream, and that we send nothing else.
        self.assertIs(body["stream"], True)
        self.assertEqual(body["options"]["num_predict"], llm.NUM_PREDICT)
        self.assertEqual(body["keep_alive"], "10m")

    def test_prompt_carries_every_constraint(self):
        Stub.generate = generate_response(
            json.dumps({"ideas": [make_idea(f"Quick thing {i}", 10) for i in range(3)]})
        )
        result = self.generate(
            materials="bamboo skewers, rubber bands",
            time_minutes=15,
            budget="500",
            skill_level="Advanced",
            restrictions="no sharp tools, nothing that flies",
            preference="Decorative",
        )
        self.assertTrue(result.ok, result.error)
        prompt = Stub.generate_posts()[0]["prompt"]
        for fragment in (
            "bamboo skewers, rubber bands",
            "15 minutes",
            "500",
            "Advanced",
            "no sharp tools, nothing that flies",
            "Decorative",
        ):
            self.assertIn(fragment, prompt)

    def test_missing_model_never_posts(self):
        Stub.body = {"models": [{"name": "llama3:8b"}]}
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("ollama pull gemma3:1b", result.error)
        self.assertEqual(Stub.posts, [], "no generation request should be sent")

    def test_empty_response_text(self):
        Stub.generate = generate_response(text="")
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("empty answer", result.error)
        self.assertEqual(result.ideas, [])

    def test_missing_response_key(self):
        Stub.generate = {"done": True}
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("empty answer", result.error)

    def test_unexpected_envelope_shape(self):
        Stub.generate = [1, 2, 3]
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("unexpected shape", result.error)

    def test_unparseable_text(self):
        Stub.generate = generate_response(text="Sorry, I cannot help with that.")
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("parseable JSON", result.error)

    def test_truncated_reply_says_so(self):
        Stub.generate = generate_response(text='{"ideas": [{"title": "ha')
        Stub.generate["done_reason"] = "length"
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("ran out of tokens", result.error)

    def test_non_2xx_response(self):
        Stub.reset(generate=generate_response(), status=500)
        result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("HTTP 500", result.error)

    def test_timeout(self):
        real_timeout = llm.GENERATE_TIMEOUT
        llm.GENERATE_TIMEOUT = 0.05
        Stub.delay = 0.5
        try:
            result = self.generate()
        finally:
            llm.GENERATE_TIMEOUT = real_timeout
            Stub.delay = 0.0
        self.assertFalse(result.ok)
        self.assertIn("didn't answer within", result.error)

    def test_a_timed_out_request_leaves_nothing_running(self):
        # Sprint 5: the old _post spawned a daemon thread and abandoned it on
        # the deadline, so a timed-out call left a live worker - and a `got`
        # dict shared between two threads - to finish on its own. That is why
        # the timeout tests needed a time.sleep(0.6) to let it die quietly.
        # _post now streams and gives up by closing the connection, so nothing
        # survives the call: no thread, no dict, no late reply.
        real_timeout = llm.GENERATE_TIMEOUT
        llm.GENERATE_TIMEOUT = 0.05
        Stub.delay = 0.5
        started = time.monotonic()
        try:
            result = self.generate()
        finally:
            elapsed = time.monotonic() - started
            llm.GENERATE_TIMEOUT = real_timeout
            Stub.delay = 0.0
        self.assertFalse(result.ok)
        self.assertLess(elapsed, 0.4, "the call must return at the deadline")
        self.assertEqual(threads_inside_llm(), [],
                         "a timed-out generation must leave no worker thread")

    def test_connection_failure(self):
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        with mock.patch.object(llm, "has_model", return_value=True):
            result = self.generate()
        self.assertFalse(result.ok)
        self.assertIn("Couldn't reach Ollama", result.error)


class ValidationTest(unittest.TestCase):
    """Constraints are requirements: violations are rejected, never repaired."""

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def result_for(self, ideas, **overrides):
        kwargs = {
            "materials": "cardboard, tape",
            "time_minutes": 30,
            "budget": "zero",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        Stub.generate = generate_response(text=json.dumps({"ideas": ideas}))
        return llm.generate_ideas(**kwargs)

    def test_markdown_fenced_json_is_accepted(self):
        payload = json.dumps({"ideas": make_three_ideas()})
        Stub.generate = generate_response(
            text=f"Sure! Here you go:\n```json\n{payload}\n```\nHope this helps."
        )
        result = llm.generate_ideas("cardboard", 30, "zero", "Beginner", "", "Fun")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.ideas), 3)

    def test_single_quoted_json_is_accepted(self):
        text = "{'ideas': [" + ", ".join(
            json.dumps(idea).replace('"', "'") for idea in make_three_ideas()
        ) + "]}"
        Stub.generate = generate_response(text=text)
        result = llm.generate_ideas("cardboard", 30, "zero", "Beginner", "", "Fun")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.ideas), 3)

    def test_missing_ideas_key_rejected(self):
        Stub.generate = generate_response(text=json.dumps({"projects": []}))
        result = llm.generate_ideas("cardboard", 30, "zero", "Beginner", "", "Fun")
        self.assertFalse(result.ok)
        self.assertIn("ideas", result.error)

    def test_ideas_wrong_type_or_empty_rejected(self):
        for payload in ({"ideas": "three of them"}, {"ideas": []}):
            Stub.generate = generate_response(text=json.dumps(payload))
            result = llm.generate_ideas("cardboard", 30, "zero", "Beginner", "", "Fun")
            self.assertFalse(result.ok, payload)

    def test_wrong_number_of_ideas_rejected(self):
        for count in (2, 4):
            ideas = make_three_ideas()[:count] if count < 3 else make_three_ideas() + [
                make_idea("Extra thing", 5)
            ]
            result = self.result_for(ideas)
            self.assertFalse(result.ok, f"{count} ideas should be rejected")
            self.assertIn("instead of exactly 3", result.error)

    def test_missing_required_field_rejected(self):
        idea = make_idea()
        del idea["why_it_fits"]
        result = self.result_for([idea, *make_three_ideas()[1:]])
        self.assertFalse(result.ok)
        self.assertIn("why_it_fits", result.error)

    def test_wrong_field_types_rejected(self):
        broken = [
            ("estimated_minutes", "soon"),
            ("estimated_minutes", True),
            ("materials_used", "cardboard"),
            ("description", 12),
            ("additional_items", {"a": 1}),
        ]
        for name, value in broken:
            idea = make_idea()
            idea[name] = value
            result = self.result_for([idea, *make_three_ideas()[1:]])
            self.assertFalse(result.ok, f"{name}={value!r} should be rejected")

    def test_empty_title_rejected(self):
        for title in ("", "   ", None):
            idea = make_idea()
            idea["title"] = title
            result = self.result_for([idea, *make_three_ideas()[1:]])
            self.assertFalse(result.ok, f"title={title!r} should be rejected")
            self.assertIn("title", result.error)

    def test_over_time_limit_rejected(self):
        result = self.result_for(make_three_ideas(), time_minutes=15)
        self.assertFalse(result.ok)
        self.assertIn("only 15 are available", result.error)

    def test_zero_budget_rejects_purchases(self):
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["a hot glue gun"]
        result = self.result_for(ideas, budget="zero")
        self.assertFalse(result.ok)
        self.assertIn("budget is ₹0", result.error)

    def test_zero_budget_accepts_empty_purchase_lists(self):
        result = self.result_for(make_three_ideas(), budget="zero")
        self.assertTrue(result.ok, result.error)
        self.assertTrue(all(not idea["additional_items"] for idea in result.ideas))

    def test_non_zero_budget_allows_purchases(self):
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["a strip of felt", "  "]
        result = self.result_for(ideas, budget="100")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.ideas[0]["additional_items"], ["a strip of felt"])

    def test_duplicate_titles_rejected(self):
        ideas = make_three_ideas()
        ideas[1]["title"] = ideas[0]["title"]
        ideas[2]["title"] = ideas[0]["title"]
        result = self.result_for(ideas)
        self.assertFalse(result.ok)
        self.assertIn("near-identical", result.error)

    def test_zero_or_negative_minutes_rejected(self):
        for bad in (0, -5):
            ideas = make_three_ideas()
            ideas[0]["estimated_minutes"] = bad
            self.assertFalse(self.result_for(ideas).ok, f"{bad} should reject")

    def test_null_list_items_are_dropped_not_stringified(self):
        ideas = make_three_ideas()
        ideas[0]["materials_used"] = ["cardboard", None, "tape"]
        result = self.result_for(ideas)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.ideas[0]["materials_used"], ["cardboard", "tape"])
        self.assertNotIn("None", result.ideas[0]["materials_used"])


class BuildGuideTest(unittest.TestCase):
    """One click, one POST, one validated guide. The transport is shared."""

    def guide(self, **overrides):
        kwargs = {
            "idea": make_idea(),
            "materials": "cardboard, tape",
            "time_minutes": 30,
            "budget": "zero",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        return llm.generate_build_guide(**kwargs)

    def setUp(self):
        self.stub = StubServer()
        Stub.generate = generate_response(json.dumps(make_guide()))

    def tearDown(self):
        self.stub.close()

    def test_valid_guide_is_returned(self):
        result = self.guide()
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.error, "")
        self.assertEqual(result.guide["estimated_minutes"], 20)
        self.assertEqual(len(result.guide["steps"]), 4)
        self.assertEqual(result.guide["steps"][2]["title"], "Tape the seam")
        self.assertEqual(result.guide["tools_needed"], [])
        self.assertEqual(result.ideas, [], "guides never populate ideas")

    def test_request_asks_for_json_and_costs_exactly_one_post(self):
        self.assertTrue(self.guide().ok)
        posts = Stub.generate_posts()
        self.assertEqual(len(posts), 1, "exactly one request per Build This")
        body = posts[0]
        self.assertEqual(body["model"], "gemma3:1b")
        # The guide's grammar: JSON with every required field, 3-7 steps, and
        # (this run is at ₹0) no tool to buy.
        guide = body["format"]
        self.assertEqual(guide["required"], list(llm._GUIDE_FIELDS))
        self.assertEqual(guide["properties"]["steps"]["minItems"], llm._STEP_MIN)
        self.assertEqual(guide["properties"]["steps"]["maxItems"], llm._STEP_MAX)
        tools = guide["properties"]["tools_needed"]
        self.assertEqual(tools["type"], "array")
        self.assertEqual(tools["maxItems"], 0, "at ₹0 a tool to buy is impossible")
        self.assertIs(body["stream"], True)
        self.assertEqual(body["options"]["num_predict"], llm.NUM_PREDICT)
        self.assertEqual(body["keep_alive"], "10m")

    def test_prompt_carries_the_selected_idea_and_original_constraints(self):
        selected = make_idea("Desk tidying tray", 25)
        Stub.generate = generate_response(
            json.dumps(make_guide(minutes=15, materials=["bamboo skewers"]))
        )
        result = self.guide(
            idea=selected,
            materials="bamboo skewers, rubber bands",
            time_minutes=15,
            budget="500",
            skill_level="Advanced",
            restrictions="no sharp tools, nothing that flies",
            preference="Decorative",
        )
        self.assertTrue(result.ok, result.error)
        prompt = Stub.generate_posts()[0]["prompt"]
        for fragment in (
            "Desk tidying tray",  # the idea that was clicked
            "bamboo skewers, rubber bands",  # the ORIGINAL materials
            "15 minutes",
            "500",
            "Advanced",
            "no sharp tools, nothing that flies",
            "Decorative",
        ):
            self.assertIn(fragment, prompt)
        self.assertNotIn("Cardboard pen stand", prompt, "no sibling idea")
        self.assertNotIn("Book mark tassel", prompt, "no history either")

    def test_missing_model_never_posts(self):
        Stub.body = {"models": [{"name": "llama3:8b"}]}
        result = self.guide()
        self.assertFalse(result.ok)
        self.assertIn("ollama pull gemma3:1b", result.error)
        self.assertEqual(Stub.posts, [], "no guide request without a model")

    def test_empty_response(self):
        Stub.generate = generate_response(text="")
        result = self.guide()
        self.assertFalse(result.ok)
        self.assertIn("empty answer", result.error)
        self.assertEqual(result.guide, {})

    def test_unparseable_text(self):
        Stub.generate = generate_response(text="I'd rather not.")
        result = self.guide()
        self.assertFalse(result.ok)
        self.assertIn("parseable JSON", result.error)

    def test_non_2xx_response(self):
        Stub.reset(generate=generate_response(json.dumps(make_guide())), status=503)
        result = self.guide()
        self.assertFalse(result.ok)
        self.assertIn("HTTP 503", result.error)

    def test_timeout(self):
        real_timeout = llm.GENERATE_TIMEOUT
        llm.GENERATE_TIMEOUT = 0.05
        Stub.delay = 0.5
        try:
            result = self.guide()
        finally:
            llm.GENERATE_TIMEOUT = real_timeout
            Stub.delay = 0.0
        self.assertFalse(result.ok)
        self.assertIn("didn't answer within", result.error)

    def test_connection_failure(self):
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        with mock.patch.object(llm, "has_model", return_value=True):
            result = self.guide()
        self.assertFalse(result.ok)
        self.assertIn("Couldn't reach Ollama", result.error)


class GuideValidationTest(unittest.TestCase):
    """A guide that breaks a constraint is rejected whole, never repaired."""

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def result_for(self, guide, **overrides):
        kwargs = {
            "idea": make_idea(),
            "materials": "cardboard, tape",
            "time_minutes": 30,
            "budget": "zero",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        Stub.generate = generate_response(text=json.dumps(guide))
        return llm.generate_build_guide(**kwargs)

    def test_default_guide_is_accepted(self):
        result = self.result_for(make_guide())
        self.assertTrue(result.ok, result.error)

    def test_fenced_json_around_prose_is_accepted(self):
        payload = json.dumps(make_guide())
        Stub.generate = generate_response(
            text=f"Here you go:\n```json\n{payload}\n```\nGood luck!"
        )
        result = llm.generate_build_guide(
            make_idea(), "cardboard, tape", 30, "zero", "Beginner", "", "Useful"
        )
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.guide["steps"]), 4)

    def test_incomplete_json_rejected(self):
        for payload in ('{"title": "pen stand"', "{}", "not json at all"):
            Stub.generate = generate_response(text=payload)
            result = llm.generate_build_guide(
                make_idea(), "cardboard, tape", 30, "zero", "Beginner", "", "Fun"
            )
            self.assertFalse(result.ok, payload)

    def test_missing_required_field_rejected(self):
        for field in ("purpose", "steps", "safety_notes", "tools_needed"):
            guide = make_guide()
            del guide[field]
            result = self.result_for(guide)
            self.assertFalse(result.ok, field)
            self.assertIn(field, result.error)

    def test_wrong_field_types_rejected(self):
        broken = [
            ("title", 12),
            # A blank purpose is deliberately tolerated (see
            # test_blank_purpose_is_tolerated_but_blank_title_is_not), but a
            # non-string one is still a malformed field.
            ("purpose", None),
            ("materials_used", "cardboard"),
            ("tools_needed", {"a": 1}),
            ("safety_notes", "mind the edges"),
            ("steps", {"1": "flatten"}),
            ("steps", ["flatten the tubes"]),
            ("estimated_minutes", True),
            ("estimated_minutes", "soon"),
        ]
        for name, value in broken:
            guide = make_guide()
            guide[name] = value
            result = self.result_for(guide)
            self.assertFalse(result.ok, f"{name}={value!r} should be rejected")

    def test_step_shape_is_enforced_and_tip_is_optional(self):
        guide = make_guide()
        for step in guide["steps"]:
            del step["tip"]
        result = self.result_for(guide)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.guide["steps"][0]["tip"], "")

        for mutate in (
            lambda step: step.pop("instruction"),
            lambda step: step.update(title="   "),
            lambda step: step.update(instruction=None),
            lambda step: step.update(title=7),
        ):
            guide = make_guide()
            mutate(guide["steps"][0])
            result = self.result_for(guide)
            self.assertFalse(result.ok, f"{mutate} should be rejected")

    def test_non_string_tip_is_dropped_not_shown(self):
        guide = make_guide()
        guide["steps"][0]["tip"] = 42
        result = self.result_for(guide)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.guide["steps"][0]["tip"], "")

    def test_non_dict_step_rejected(self):
        guide = make_guide()
        guide["steps"][1] = "now roll it back"
        result = self.result_for(guide)
        self.assertFalse(result.ok)
        self.assertIn("Step 2", result.error)

    def test_step_count_outside_three_to_seven_rejected(self):
        for count in (0, 2, 8):
            result = self.result_for(make_guide(steps=count))
            self.assertFalse(result.ok, f"{count} steps should be rejected")
            self.assertIn("steps instead of 3-7", result.error)

    def test_step_count_boundaries_accepted(self):
        for count in (3, 7):
            result = self.result_for(make_guide(steps=count))
            self.assertTrue(result.ok, f"{count} steps should be accepted")

    def test_non_positive_minutes_rejected(self):
        for bad in (0, -5):
            result = self.result_for(make_guide(minutes=bad))
            self.assertFalse(result.ok, f"{bad} minutes should be rejected")

    def test_duration_over_the_limit_rejected(self):
        result = self.result_for(make_guide(minutes=45))
        self.assertFalse(result.ok)
        self.assertIn("only 30 are available", result.error)

    def test_null_items_are_dropped_not_stringified(self):
        guide = make_guide(
            tools=[None],
            materials=["cardboard", None, 7, {"a": 1}, "  ", "tape"],
        )
        guide["safety_notes"] = [None, "Mind the cut edges"]
        guide["steps"][0]["tip"] = None
        result = self.result_for(guide, materials="cardboard, tape")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.guide["materials_used"], ["cardboard", "tape"])
        self.assertEqual(result.guide["tools_needed"], [])
        self.assertEqual(result.guide["safety_notes"], ["Mind the cut edges"])
        self.assertEqual(result.guide["steps"][0]["tip"], "")
        self.assertNotIn("None", str(result.guide))

    def test_zero_budget_rejects_tools_to_buy(self):
        result = self.result_for(make_guide(tools=["a glue gun"]))
        self.assertFalse(result.ok)
        self.assertIn("budget is ₹0", result.error)

    def test_non_zero_budget_allows_tools(self):
        result = self.result_for(make_guide(tools=["  ", "a stapler"]), budget="100")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.guide["tools_needed"], ["a stapler"])

    def test_unavailable_material_rejected(self):
        result = self.result_for(
            make_guide(materials=["cardboard", "glue gun"]), materials="cardboard, tape"
        )
        self.assertFalse(result.ok)
        self.assertIn("isn't in the materials you listed", result.error)

    def test_fuzzy_but_reasonable_materials_accepted(self):
        result = self.result_for(
            make_guide(materials=["a piece of brown paper", "masking tape"]),
            materials="old brown paper bags, masking tape, cardboard",
        )
        self.assertTrue(result.ok, result.error)

    def test_no_glue_restriction_rejected(self):
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Smear glue along the seam and press."
        result = self.result_for(guide, restrictions="no glue")
        self.assertFalse(result.ok)
        self.assertIn("no glue", result.error)

    def test_no_sharp_restriction_rejected(self):
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Cut the tape to length with scissors."
        result = self.result_for(guide, restrictions="No sharp tools please")
        self.assertFalse(result.ok)
        self.assertIn("no sharp tools", result.error)

    def test_no_power_restriction_rejected(self):
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Stick a battery pack to the base."
        result = self.result_for(guide, restrictions="no electric tools")
        self.assertFalse(result.ok)
        self.assertIn("no power", result.error)

    def test_no_fire_restriction_rejected(self):
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Warm it over a candle until soft."
        result = self.result_for(guide, restrictions="nothing here, without heat")
        self.assertFalse(result.ok)
        self.assertIn("no fire or heat", result.error)

    def test_the_same_guide_passes_when_the_restriction_is_not_stated(self):
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Cut the tape to length with scissors."
        self.assertTrue(self.result_for(guide).ok, "restrictions are opt-in")

    def test_ordinary_craft_words_are_not_hazard_terms(self):
        guide = make_guide()
        guide["steps"][0]["instruction"] = (
            "Use a sharp knife to slice the cardboard along a straight edge."
        )
        result = self.result_for(guide)
        self.assertTrue(result.ok, "cutting tools are normal craft, not hazards")

    def test_hazardous_instructions_rejected(self):
        hazards = (
            "Plug the holder into the mains and leave it overnight.",
            "Solder the two wires together for a firm hold.",
            "Soften the tube with a blowtorch before bending.",
            "Keep an open flame under the cardboard to soften it.",
            "Add a splash of bleach to the water in the base.",
            "Hang the shelf on a load-bearing bracket you built yourself.",
            "Pack gunpowder into the tube for a louder bang.",
        )
        for hazard in hazards:
            guide = make_guide()
            guide["steps"][1]["instruction"] = hazard
            result = self.result_for(guide)
            self.assertFalse(result.ok, f"{hazard!r} should be rejected")
            self.assertIn("unsafe", result.error)

    def test_hazard_in_tools_or_title_is_also_caught(self):
        in_tools = make_guide()
        in_tools["tools_needed"] = ["a mains drill"]  # only legal with a real budget
        self.assertFalse(self.result_for(in_tools, budget="500").ok)
        self.assertIn("unsafe", self.result_for(in_tools, budget="500").error)
        in_title = make_guide()
        in_title["title"] = "Mains-powered drying rack"
        self.assertFalse(self.result_for(in_title).ok, "title must be scanned too")
        in_purpose = make_guide()
        in_purpose["purpose"] = "A holder for a blowtorch while you work."
        self.assertFalse(self.result_for(in_purpose).ok, "purpose must be scanned")

    def test_adjectives_and_innocent_phrases_are_not_hazards(self):
        """The tightened patterns must not reject ordinary craft wording."""
        for adjective in ("a thinner layer of tape", "use a thinner strip of card"):
            guide = make_guide()
            guide["steps"][0]["instruction"] = f"Apply {adjective}."
            self.assertTrue(self.result_for(guide).ok, f"{adjective!r} rejected")
        guide = make_guide()
        guide["steps"][0]["instruction"] = "Plug the gap between the tubes with card."
        self.assertTrue(self.result_for(guide).ok, "'plug the gap' is not electricity")
        guide = make_guide()
        guide["steps"][0]["instruction"] = "Let it cool - it stays hot for a while."
        self.assertTrue(
            self.result_for(guide, restrictions="no heat").ok,
            "a safety note saying 'hot' must survive a no-heat restriction",
        )

    def test_button_batteries_are_a_hazard(self):
        guide = make_guide()
        guide["steps"][0]["instruction"] = "Insert the button batteries to test it."
        self.assertFalse(self.result_for(guide).ok)

    def test_blank_purpose_is_tolerated_but_blank_title_is_not(self):
        blank = make_guide()
        blank["purpose"] = ""
        self.assertTrue(self.result_for(blank).ok, "purpose is decorative")
        untitled = make_guide()
        untitled["title"] = "   "
        self.assertFalse(self.result_for(untitled).ok, "title is required")


class RuleTableTest(unittest.TestCase):
    """The rule tables are indexed by UI values - a silent miss is the risk."""

    def setUp(self):
        # Importing app.py executes the whole Streamlit script, which probes
        # Ollama on import. Point it at a closed port so this unit test never
        # does real loopback I/O (and never hangs if a machine has Ollama up).
        self._real_url = llm.OLLAMA_URL
        llm.OLLAMA_URL = "http://127.0.0.1:9"

    def tearDown(self):
        llm.OLLAMA_URL = self._real_url

    def test_every_ui_choice_has_a_rule(self):
        from app import BUDGET_LABELS, TIME_CHOICES

        for label in TIME_CHOICES:
            self.assertIn(int(label.split()[0]), llm._TIME_RULES)
        for key in BUDGET_LABELS:
            self.assertIn(key, llm._BUDGET_RULES)
        for level in ("Beginner", "Intermediate", "Advanced"):
            self.assertIn(level, llm._SKILL_RULES)

    def test_prompt_uses_the_rule_not_the_fallback(self):
        prompt = llm.build_prompt("cardboard", 15, "zero", "Beginner", "", "Fun")
        self.assertIn(llm._TIME_RULES[15], prompt)
        self.assertIn(llm._BUDGET_RULES["zero"], prompt)
        self.assertIn(llm._SKILL_RULES["Beginner"], prompt)
        self.assertNotIn("finishable in one sitting", prompt)

    def test_guide_prompt_reuses_the_same_rule_tables(self):
        prompt = llm.build_guide_prompt(
            make_idea(), "cardboard", 15, "zero", "Beginner", "", "Fun"
        )
        self.assertIn(llm._TIME_RULES[15], prompt)
        self.assertIn(llm._BUDGET_RULES["zero"], prompt)
        self.assertIn(llm._SKILL_RULES["Beginner"], prompt)
        self.assertNotIn("finishable in one sitting", prompt)
        self.assertIn('"steps": [{"title": "", "instruction": "", "tip": ""}]',
                      prompt)

    def test_guide_restriction_and_safety_tables_are_valid_regexes(self):
        for phrase, forbidden, label in llm._RESTRICTION_RULES:
            re.compile(phrase)   # a broken pattern raises here, which is the test
            re.compile(forbidden)
            self.assertTrue(label, "every rule needs a human label")
        for term in llm._SAFETY_TERMS:
            re.compile(term)
        self.assertTrue(llm._SAFETY_RE.search("Hang it off the mains socket"))
        # Every restriction rule must fire on its own trigger phrase, or the
        # table entry is dead weight that looks like coverage.
        samples = {
            "no sharp tools": "cut it with scissors",
            "no glue": "glue it shut",
            "no power": "plug it in",
            "no fire or heat": "hold it over the flame",
        }
        for _phrase, forbidden, label in llm._RESTRICTION_RULES:
            self.assertIn(label, samples, "every rule needs a firing sample")
            self.assertTrue(
                re.search(forbidden, samples[label], re.I),
                f"{label} never fires on its own sample",
            )


class RemixTest(unittest.TestCase):
    """Constraint Remix: one selected idea, one mode, one request."""

    IDEA = make_idea("Cardboard pen stand", 20)
    OTHER = make_idea("Desk tidying tray", 25)

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def answer(self, idea):
        Stub.generate = generate_response(text=json.dumps({"idea": idea}))

    def remix(self, mode, idea=None, reply=None, **overrides):
        kwargs = {
            "idea": self.IDEA if idea is None else idea,
            "mode": mode,
            "materials": "cardboard tubes, tape, string",
            "time_minutes": 30,
            "budget": "100",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        if reply is not None:
            self.answer(reply)
        return llm.remix_idea(**kwargs)

    def prompt(self):
        return Stub.generate_posts()[-1]["prompt"]

    # --- the prompt: the selected idea, the original constraints, the mode ---

    def test_each_mode_builds_its_own_prompt(self):
        expectations = {
            "zero_buy": "nothing bought at all",
            "use_it_up": "get used up first",
            "remove_material": "without this material: glue",
            "simplify": "beginner version",
        }
        for mode, fragment in expectations.items():
            with self.subTest(mode=mode):
                extra = {"remove_material": {"exclude": "glue"}}.get(mode, {})
                self.assertTrue(self.remix(mode, reply=self.IDEA,
                                           **extra).ok)
                prompt = self.prompt()
                self.assertIn(fragment, prompt)
                # The original constraints travel with the remix.
                for constraint in ("cardboard tubes, tape, string", "30 minutes",
                                   "100", "Beginner", "Surprise me"):
                    self.assertIn(constraint, prompt)

    def test_prompt_carries_the_selected_idea_and_says_what_stays_the_same(self):
        self.assertTrue(self.remix("zero_buy", reply=self.IDEA).ok)
        prompt = self.prompt()
        self.assertIn(self.IDEA["title"], prompt)
        self.assertIn(self.IDEA["description"], prompt)
        self.assertIn("What must change:", prompt)
        self.assertIn("What must stay the same:", prompt)
        self.assertIn(self.IDEA["why_it_fits"], prompt)

    def test_only_the_given_idea_is_sent(self):
        self.assertTrue(self.remix("zero_buy", idea=self.IDEA,
                                   reply=self.OTHER).ok)
        prompt = self.prompt()
        self.assertNotIn(self.OTHER["title"], prompt)

    def test_asked_for_exactly_one_adapted_idea(self):
        self.assertTrue(self.remix("zero_buy", reply=self.IDEA).ok)
        prompt = self.prompt()
        self.assertIn('{"idea": {', prompt)
        self.assertNotIn('"ideas"', prompt)

    def test_use_it_up_focus_reaches_the_prompt(self):
        self.assertTrue(self.remix("use_it_up", focus="old newspaper, jar lids",
                                   reply=self.IDEA).ok)
        self.assertIn("old newspaper, jar lids", self.prompt())
        self.assertIn("materials_used must be items from that list",
                      self.prompt())

    def test_use_it_up_falls_back_to_the_materials_when_no_focus_is_given(self):
        self.assertTrue(self.remix("use_it_up", reply=self.IDEA).ok)
        self.assertIn("get used up first: cardboard tubes, tape, string",
                      self.prompt())

    def test_remove_material_names_the_banned_material_twice(self):
        taped = tape_free(make_idea("Cardboard tray", 15))
        taped["materials_used"] = ["cardboard tubes", "string"]
        self.assertTrue(self.remix("remove_material", exclude="tape",
                                   reply=taped).ok)
        prompt = self.prompt()
        self.assertEqual(prompt.count("'tape' is not available"), 1)
        self.assertIn("Never name it in materials_used", prompt)
        self.assertIn("without this material: tape", prompt)

    # --- one press, one request --------------------------------------------

    def test_one_request_per_remix_and_none_without_one(self):
        self.answer(self.IDEA)
        self.assertTrue(self.remix("zero_buy").ok)
        self.assertEqual(len(Stub.generate_posts()), 1)
        self.assertEqual(Stub.generate_posts()[0]["model"], llm.MODEL)
        self.assertIs(Stub.generate_posts()[0]["stream"], True)

    def test_missing_material_never_posts(self):
        result = self.remix("remove_material", exclude="   ")
        self.assertFalse(result.ok)
        self.assertIn("Name the material to remove", result.error)
        self.assertEqual(Stub.posts, [])

    def test_unknown_mode_never_posts(self):
        result = self.remix("make_it_better")
        self.assertFalse(result.ok)
        self.assertIn("isn't a remix mode", result.error)
        self.assertEqual(Stub.posts, [])

    def test_missing_model_never_posts(self):
        Stub.body = {"models": [{"name": "llama3:8b"}]}
        self.assertFalse(self.remix("zero_buy").ok)
        self.assertEqual(Stub.posts, [])

    # --- the grammar: what must be empty cannot be anything else ------------

    def test_zero_buy_grammar_forbids_a_purchase_even_at_a_real_budget(self):
        self.assertTrue(self.remix("zero_buy", reply=self.IDEA).ok)
        extras = Stub.generate_posts()[0]["format"]["properties"]["idea"][
            "properties"]["additional_items"]
        self.assertEqual(extras["maxItems"], 0)

    def test_other_modes_may_still_buy_at_a_real_budget(self):
        self.assertTrue(self.remix("simplify", reply=self.IDEA).ok)
        extras = Stub.generate_posts()[0]["format"]["properties"]["idea"][
            "properties"]["additional_items"]
        self.assertNotIn("maxItems", extras)

    # --- the gates: rejection, never repair ---------------------------------

    def test_a_valid_remix_returns_exactly_one_idea(self):
        adapted = make_idea("Tape-free cardboard tray", 15)
        result = self.remix("zero_buy", reply=adapted)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.ideas), 1)
        self.assertEqual(result.ideas[0]["title"], "Tape-free cardboard tray")
        self.assertEqual(result.guide, {})

    def test_zero_buy_rejects_a_non_empty_purchase_list(self):
        adapted = make_idea("Felt-lined tray", 15, extra=["a strip of felt"])
        result = self.remix("zero_buy", reply=adapted)
        self.assertFalse(result.ok)
        self.assertIn("still lists things to buy", result.error)
        self.assertEqual(result.ideas, [])
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_remove_material_rejects_a_reply_that_still_uses_it(self):
        result = self.remix("remove_material", exclude="tape", reply=self.IDEA)
        self.assertFalse(result.ok)
        self.assertIn("still uses", result.error)
        self.assertIn("tape", result.error)
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_remove_material_rejects_a_compound_mention_of_it(self):
        adapted = make_idea("Tray", 15, extra=None)
        adapted["materials_used"] = ["cardboard tubes", "masking tape"]
        result = self.remix("remove_material", exclude="tape", reply=adapted)
        self.assertFalse(result.ok)
        self.assertIn("still uses masking tape", result.error)

    def test_remove_material_accepts_a_reply_without_it(self):
        adapted = tape_free(make_idea("Cardboard tray", 15))
        adapted["materials_used"] = ["cardboard tubes", "string"]
        result = self.remix("remove_material", exclude="tape", reply=adapted)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.ideas[0]["materials_used"],
                         ["cardboard tubes", "string"])

    def test_remove_material_rejects_it_in_the_instructions_not_the_list(self):
        # The materials list is clean; the prose still says "tape it shut".
        # An absent keyword is not proof of absence, so this must reject too.
        adapted = tape_free(make_idea("Cardboard tray", 15))
        adapted["materials_used"] = ["cardboard tubes", "string"]
        adapted["description"] = "Fold the cardboard, then tape it shut."
        result = self.remix("remove_material", exclude="tape", reply=adapted)
        self.assertFalse(result.ok)
        self.assertIn("still needs it", result.error)
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_simplify_rejects_an_output_that_still_calls_itself_hard(self):
        adapted = make_idea("Hard tray", 15)
        adapted["difficulty"] = "hard"
        result = self.remix("simplify", reply=adapted)
        self.assertFalse(result.ok)
        self.assertIn("still calls itself hard", result.error)

    def test_simplify_accepts_easy_and_medium(self):
        for level in ("easy", "Easy", "medium"):
            with self.subTest(level=level):
                adapted = make_idea("Simple tray", 15)
                adapted["difficulty"] = level
                result = self.remix("simplify", reply=adapted)
                self.assertTrue(result.ok, result.error)

    def test_use_it_up_priority_is_not_string_checked(self):
        # Documented limitation: a reply that ignores the focus is accepted,
        # because no word match can prove the model used those materials first.
        adapted = make_idea("Pen stand", 20)
        adapted["materials_used"] = ["cardboard tubes", "tape"]
        result = self.remix("use_it_up", focus="old newspaper",
                            reply=adapted)
        self.assertTrue(result.ok, result.error)

    # --- the same idea rules still apply ------------------------------------

    def test_malformed_json_is_rejected_without_a_repair_call(self):
        Stub.generate = generate_response(text="Sorry, I cannot do that.")
        result = self.remix("zero_buy")
        self.assertFalse(result.ok)
        self.assertIn("parseable JSON", result.error)
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_a_reply_that_is_not_one_idea_is_rejected(self):
        Stub.generate = generate_response(
            text=json.dumps({"ideas": make_three_ideas()})
        )
        result = self.remix("zero_buy")
        self.assertFalse(result.ok)
        self.assertIn("didn't return one adapted idea", result.error)

    def test_two_ideas_where_one_was_asked_for_is_rejected(self):
        Stub.generate = generate_response(text=json.dumps(
            {"idea": [self.IDEA, self.OTHER]}))
        result = self.remix("zero_buy")
        self.assertFalse(result.ok)
        self.assertIn("didn't return one adapted idea", result.error)

    def test_missing_and_mistyped_fields_are_rejected(self):
        cases = {
            "difficulty": {"difficulty": None},
            "estimated_minutes": {"estimated_minutes": "soon"},
            "materials_used": {"materials_used": "cardboard"},
            "title": {"title": "   "},
        }
        for field, change in cases.items():
            with self.subTest(field=field):
                before = len(Stub.generate_posts())
                adapted = make_idea()
                del adapted[field]
                adapted.update(change)
                result = self.remix("zero_buy", reply=adapted)
                self.assertFalse(result.ok)
                self.assertEqual(len(Stub.generate_posts()), before + 1,
                                 "one request and no repair call")

    def test_the_original_time_and_budget_rules_still_apply(self):
        over = make_idea("Slow tray", 45)
        self.assertIn("only 30 are available",
                      self.remix("zero_buy", reply=over).error)
        bought = make_idea("Felt tray", 15, extra=["felt"])
        self.assertIn("budget is", self.remix("zero_buy", budget="zero",
                                              reply=bought).error)

    def test_transport_failures_come_back_as_messages(self):
        Stub.delay, saved = 0.5, llm.GENERATE_TIMEOUT
        llm.GENERATE_TIMEOUT = 0.05
        try:
            result = self.remix("zero_buy")
        finally:
            Stub.delay, llm.GENERATE_TIMEOUT = 0.0, saved
        self.assertFalse(result.ok)
        self.assertIn("didn't answer within", result.error)
        llm.OLLAMA_URL = "http://127.0.0.1:9"
        with mock.patch.object(llm, "has_model", return_value=True):
            result = self.remix("zero_buy")
        self.assertIn("Couldn't reach Ollama", result.error)
        with mock.patch.object(llm, "has_model", return_value=True), \
                mock.patch.object(llm.requests, "post",
                                  side_effect=ValueError("bad body")):
            result = self.remix("zero_buy")
        self.assertIn("couldn't take that request", result.error)
        self.assertIn("ValueError", result.error)


class RestrictionEnforcementTest(unittest.TestCase):
    """A restriction is a rule, not a wish. Sprint 5: it is checked over every
    field of the guide - the materials list included - and the wording people
    actually type all reach the same rule."""

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def result_for(self, guide, **overrides):
        kwargs = {
            "idea": make_idea(),
            # The user owns scissors here on purpose: the materials check must
            # pass so that ONLY the restriction scan can catch the defect.
            "materials": "cardboard, tape, scissors",
            "time_minutes": 30,
            "budget": "zero",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        Stub.generate = generate_response(text=json.dumps(guide))
        return llm.generate_build_guide(**kwargs)

    def cutting_guide(self):
        guide = make_guide()
        guide["steps"][1]["instruction"] = "Cut the tape to length with scissors."
        return guide

    # --- the defect: materials_used was never scanned ------------------------

    def test_a_banned_tool_in_the_materials_list_is_caught(self):
        guide = make_guide(materials=["cardboard", "scissors"])
        result = self.result_for(guide, restrictions="no sharp tools")
        self.assertFalse(result.ok, "the materials list is part of the guide")
        self.assertIn("no sharp tools", result.error)

    def test_a_banned_tool_in_the_tools_list_is_caught(self):
        guide = make_guide(tools=["a craft knife"])
        result = self.result_for(guide, budget="100", restrictions="no sharp tools")
        self.assertFalse(result.ok)
        self.assertIn("no sharp tools", result.error)

    def test_a_banned_tool_in_the_title_is_caught(self):
        guide = make_guide()
        guide["title"] = "Scissors and tape desk tidy"
        self.assertFalse(self.result_for(guide, restrictions="no sharp tools").ok)

    def test_a_banned_glue_material_is_caught(self):
        guide = make_guide(materials=["cardboard", "glue"])
        self.assertFalse(self.result_for(guide, restrictions="no glue").ok)

    # --- and the rule still holds over the steps ----------------------------

    def test_the_error_names_the_restriction_and_what_tripped_it(self):
        result = self.result_for(self.cutting_guide(),
                                 restrictions="no sharp tools")
        self.assertFalse(result.ok)
        self.assertIn("'no sharp tools'", result.error)
        self.assertIn("'cut'", result.error.lower(),
                      "the user must be told what to argue with")
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_a_compliant_guide_under_the_same_restriction_is_accepted(self):
        # The default step pool folds, rolls and tapes. No banned verb in it.
        result = self.result_for(make_guide(), restrictions="no sharp tools")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.guide["steps"]), 4)

    # --- wording variants, one shared rule -----------------------------------

    def test_the_usual_wordings_all_reach_the_same_rule(self):
        wordings = (
            "no sharp tools", "No Sharp Tools", "without sharp tools",
            "avoid sharp objects", "must not use sharp tools", "nothing sharp",
            "don't use sharp tools", "no knives or scissors",
        )
        for wording in wordings:
            with self.subTest(wording=wording):
                result = self.result_for(self.cutting_guide(),
                                         restrictions=wording)
                self.assertFalse(result.ok, f"{wording!r} was not enforced")

    def test_variants_of_the_other_rules(self):
        cases = {
            "without glue": "Glue the flap to the side.",
            "must not use adhesives": "Use an adhesive to stick them.",
            "no batteries": "Add the batteries to the base.",
            "nothing with fire": "Hold it over the flame.",
        }
        for wording, instruction in cases.items():
            with self.subTest(wording=wording):
                guide = make_guide()
                guide["steps"][0]["instruction"] = instruction
                self.assertFalse(self.result_for(guide,
                                                 restrictions=wording).ok)

    # --- the honest part: what cannot be checked is not claimed ---------------

    def test_word_checkable_restrictions_are_listed_as_checked(self):
        self.assertEqual(
            llm.checked_restrictions("no sharp tools, no glue, nothing loud"),
            ["no sharp tools", "no glue"])

    def test_uncheckable_restrictions_are_listed_rather_than_claimed(self):
        restrictions = "no sharp tools, no glue, must fit on my desk, indoor only"
        self.assertEqual(llm.checked_restrictions(restrictions),
                         ["no sharp tools", "no glue"])
        self.assertEqual(llm.unchecked_restrictions(restrictions),
                         ["must fit on my desk", "indoor only"])

    def test_no_restriction_means_nothing_is_claimed_in_either_direction(self):
        self.assertEqual(llm.checked_restrictions(""), [])
        self.assertEqual(llm.unchecked_restrictions(""), [])

    # --- the prompt and the gate come from one table -------------------------

    def test_every_rule_has_prompt_guidance_to_quote(self):
        for _phrase, _forbidden, label in llm._RESTRICTION_RULES:
            self.assertIn(label, llm._RESTRICTION_GUIDE,
                          f"{label} is enforced but never explained")

    def test_the_guide_prompt_quotes_the_rule_with_a_worked_example(self):
        prompt = llm.build_guide_prompt(
            make_idea(), "cardboard", 30, "zero", "Beginner", "no sharp tools",
            "Fun")
        self.assertIn("no sharp tools", prompt)
        self.assertIn(llm._RESTRICTION_GUIDE["no sharp tools"], prompt)
        self.assertIn("Non-negotiable restrictions", prompt)
        self.assertIn("Bad:", prompt, "the banned verb has to be named")

    def test_the_prompt_carries_no_restriction_block_when_there_is_nothing(self):
        prompt = llm.build_guide_prompt(
            make_idea(), "cardboard", 30, "zero", "Beginner", "", "Fun")
        self.assertNotIn("Non-negotiable restrictions", prompt,
                         "a 1B is length-sensitive; do not pad the prompt")


class InventoryTest(unittest.TestCase):
    """What the user already owns is evidence. One shared word is not, and a
    word hiding inside a longer one never is."""

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def covered(self, item, inventory="cardboard, masking tape, string"):
        return llm._covered_by_inventory(item, llm._significant(inventory))

    def guide_for(self, guide, **overrides):
        kwargs = {
            "idea": make_idea(),
            "materials": "cardboard, masking tape, string",
            "time_minutes": 30,
            "budget": "100",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        Stub.generate = generate_response(text=json.dumps(guide))
        return llm.generate_build_guide(**kwargs)

    # --- one shared word proves nothing --------------------------------------

    def test_a_shared_word_is_not_evidence_of_the_whole_material(self):
        # "cardboard" said nothing about tubes, and "glue" says nothing about a
        # glue gun. Under Sprint 4's any() rule both passed.
        self.assertFalse(self.covered("cardboard tubes", "cardboard"))
        self.assertFalse(self.covered("glue gun", "cardboard, glue"))

    def test_a_word_hiding_inside_a_longer_one_is_not_a_match(self):
        # "ink" is inside "pink"; "ap" inside "tape". Sprint 4's substring test
        # called both a match and waved them through.
        self.assertFalse(self.covered("ink", "pink fabric"))
        self.assertFalse(self.covered("ap glue", "tape"))

    def test_every_significant_word_must_be_present(self):
        self.assertTrue(self.covered("cardboard", "old brown paper, cardboard"))
        self.assertTrue(self.covered("masking tape", "tape, masking tape"))
        self.assertFalse(self.covered("masking tape", "tape"))

    def test_the_direction_of_the_claim_matters(self):
        # Asking for less than they have is fine; asking for more is not.
        self.assertTrue(self.covered("tape", "masking tape"))
        self.assertFalse(self.covered("masking tape", "tape"))

    def test_plural_folding_is_symmetric_and_only_that(self):
        self.assertTrue(self.covered("cardboard tubes", "cardboard tube, string"))
        self.assertTrue(self.covered("glass marbles", "marbles, glass"))
        self.assertFalse(self.covered("cardboard tubes", "cardboard"))

    def test_filler_words_carry_no_evidence_either_way(self):
        self.assertTrue(self.covered("a piece of cardboard", "cardboard"))
        self.assertTrue(self.covered("the string", "string, tape"))

    # --- in the guide --------------------------------------------------------

    def test_a_material_only_partly_owned_is_rejected(self):
        result = self.guide_for(make_guide(materials=["cardboard tubes"]))
        self.assertFalse(result.ok)
        self.assertIn("isn't in the materials you listed", result.error)

    def test_a_fully_owned_material_is_accepted(self):
        result = self.guide_for(make_guide(materials=["cardboard", "tape"]))
        self.assertTrue(result.ok, result.error)

    # --- inventory is never a purchase ---------------------------------------

    def test_a_purchase_the_user_already_owns_is_rejected(self):
        # Measured in Sprint 4: at ₹100 with cardboard and masking tape, all
        # three ideas listed 'tape' - which they already had - as a purchase.
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["tape"]
        Stub.generate = generate_response(text=json.dumps({"ideas": ideas}))
        result = llm.generate_ideas(
            materials="cardboard, masking tape", time_minutes=30, budget="100",
            skill_level="Beginner", restrictions="", preference="Fun")
        self.assertFalse(result.ok)
        self.assertIn("already in your materials", result.error)
        self.assertIn("'tape'", result.error)

    def test_a_genuine_purchase_is_still_allowed(self):
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["a strip of felt"]
        Stub.generate = generate_response(text=json.dumps({"ideas": ideas}))
        result = llm.generate_ideas(
            materials="cardboard, masking tape", time_minutes=30, budget="100",
            skill_level="Beginner", restrictions="", preference="Fun")
        self.assertTrue(result.ok, result.error)

    def test_the_guide_never_asks_the_user_to_buy_what_they_have(self):
        result = self.guide_for(make_guide(tools=["tape"]))
        self.assertFalse(result.ok)
        self.assertIn("already in your materials", result.error)


class BudgetSemanticsTest(unittest.TestCase):
    """₹0 is a rule. ₹100 and ₹500 are an instruction the app cannot check,
    and nothing here pretends otherwise."""

    def setUp(self):
        # Every class that generates needs the loopback stub. Without it the
        # suite would quietly post to whatever llm.OLLAMA_URL happens to be -
        # a real Ollama on this machine - and pass off the model's answer as a
        # test result.
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def test_zero_is_the_only_budget_the_grammar_enforces(self):
        zero = llm._ideas_schema("zero")["properties"]["ideas"]["items"]
        self.assertEqual(zero["properties"]["additional_items"]["maxItems"], 0)
        for budget in ("100", "500", "flexible"):
            with self.subTest(budget=budget):
                paid = llm._ideas_schema(budget)["properties"]["ideas"]["items"]
                self.assertNotIn("maxItems",
                                 paid["properties"]["additional_items"])

    def test_there_is_no_price_field_to_check_a_ceiling_against(self):
        # The honest reason a ₹100 ceiling cannot be enforced: the reply
        # carries no number of any kind. A model-estimated price would be
        # invented evidence, so the field simply does not exist.
        for schema in (llm._ideas_schema("100"), llm._guide_schema("100"),
                       llm._remix_schema("100", "simplify")):
            self.assertNotIn("price", json.dumps(schema).lower())
        self.assertEqual(
            set(llm._IDEA_FIELDS) | set(llm._GUIDE_FIELDS),
            {"title", "description", "purpose", "materials_used",
             "additional_items", "tools_needed", "estimated_minutes", "steps",
             "difficulty", "why_it_fits", "safety_notes"})

    def test_flexible_imposes_no_ceiling_and_no_zero_rule(self):
        ideas = make_three_ideas()
        ideas[0]["additional_items"] = ["a strip of felt", "a jar of paint"]
        Stub.generate = generate_response(text=json.dumps({"ideas": ideas}))
        result = llm.generate_ideas(
            materials="cardboard", time_minutes=30, budget="flexible",
            skill_level="Beginner", restrictions="", preference="Fun")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.ideas[0]["additional_items"]), 2)


class RemixIntegrityTest(unittest.TestCase):
    """Sprint 5: a remix has to have actually changed something, and must not
    smuggle back a prohibited requirement."""

    IDEA = make_idea("Cardboard pen stand", 20)

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def remix(self, mode, idea=None, reply=None, **overrides):
        kwargs = {
            "idea": self.IDEA if idea is None else idea,
            "mode": mode,
            "materials": "cardboard tubes, tape, string",
            "time_minutes": 30,
            "budget": "100",
            "skill_level": "Beginner",
            "restrictions": "",
            "preference": "Surprise me",
        }
        kwargs.update(overrides)
        if reply is not None:
            Stub.generate = generate_response(text=json.dumps({"idea": reply}))
        return llm.remix_idea(**kwargs)

    def bought(self, title="Felt tray"):
        return make_idea(title, 15, extra=["a strip of felt"])

    def test_a_zero_buy_remix_repeating_the_original_title_is_rejected(self):
        # Measured in Sprint 4: zero-buy returned the original's own title with
        # an empty materials list, and passed validation as a "version".
        result = self.remix("zero_buy", idea=self.bought("Felt tray"),
                            reply=make_idea("Felt tray", 15))
        self.assertFalse(result.ok)
        self.assertIn("same title as the original", result.error)
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_the_same_title_is_fine_when_the_mode_had_nothing_to_change(self):
        # Nothing needed buying, so nothing changed, and that is not a fault.
        result = self.remix("zero_buy", idea=self.IDEA, reply=self.IDEA)
        self.assertTrue(result.ok, result.error)

    def test_a_zero_buy_remix_that_really_changed_keeps_a_new_title(self):
        result = self.remix("zero_buy", idea=self.bought("Felt tray"),
                            reply=make_idea("Tape-free cardboard tray", 15))
        self.assertTrue(result.ok, result.error)

    def test_a_remix_that_removed_nothing_says_what_is_still_there(self):
        # remove_material needs no separate "nothing happened" rule: naming the
        # material is the proof that nothing was removed, and this is what the
        # user is told.
        result = self.remix("remove_material", exclude="tape", reply=self.IDEA)
        self.assertFalse(result.ok)
        self.assertIn("still uses tape", result.error)

    def test_a_real_removal_is_accepted_even_if_the_title_is_unchanged(self):
        adapted = tape_free(self.IDEA)
        adapted["materials_used"] = ["cardboard"]
        result = self.remix("remove_material", exclude="tape", reply=adapted)
        self.assertTrue(result.ok,
                        "the material really did go, so the name may stay")

    def test_a_remix_may_not_reintroduce_a_hard_restriction(self):
        adapted = tape_free(make_idea("Cardboard tray", 15))
        adapted["description"] = "Glue the two halves together and wrap it."
        result = self.remix("zero_buy", reply=adapted, restrictions="no glue")
        self.assertFalse(result.ok)
        self.assertIn("no glue", result.error)
        self.assertEqual(len(Stub.generate_posts()), 1, "no repair call")

    def test_every_mode_still_works_end_to_end(self):
        replies = {
            "zero_buy": (make_idea("Tape-free tray", 15), {}),
            "use_it_up": (make_idea("Paper and jar tray", 20), {}),
            "simplify": (make_idea("Simple cardboard tray", 10), {}),
            "remove_material": (dict(
                tape_free(make_idea("Cardboard tray", 15)),
                materials_used=["cardboard"]), {"exclude": "tape"}),
        }
        for mode, (reply, extra) in replies.items():
            with self.subTest(mode=mode):
                result = self.remix(mode, reply=reply, **extra)
                self.assertTrue(result.ok, f"{mode}: {result.error}")
                self.assertEqual(len(result.ideas), 1)


class DiagnosticsTest(unittest.TestCase):
    """The opt-in real-model diagnostic, driven by the stub, never the model."""

    def setUp(self):
        self.stub = StubServer()

    def tearDown(self):
        self.stub.close()

    def run_diagnostic(self):
        lines = []
        code = test_real_model.diagnose(lines.append)
        return code, "\n".join(lines)

    def canned_run(self, first=None, paid=None, remix=None):
        """One canned answer per check, in the order diagnose() asks.

        Seven checks: ideas, ideas, ₹100 ideas, guide, restricted guide, and one
        remix per mode. Each answer has to satisfy the gate for the check it is
        answering, so the two remixes differ - the remove-material one really
        does have to get rid of the tape.
        """
        return [
            generate_response(text=first or json.dumps(
                {"ideas": make_three_ideas()})),
            generate_response(text=json.dumps({"ideas": make_three_ideas()})),
            generate_response(text=paid or json.dumps(
                {"ideas": make_three_ideas()})),
            generate_response(text=json.dumps(make_guide())),
            generate_response(text=json.dumps(make_guide())),
            generate_response(text=json.dumps(
                {"idea": make_idea("Cardboard tray, no tape", 15)})),
            generate_response(text=json.dumps({"idea": remix})),
        ]

    def test_unreachable_ollama_is_reported_with_setup_help(self):
        llm.OLLAMA_URL = "http://127.0.0.1:9"  # discard port, closed
        code, printed = self.run_diagnostic()
        self.assertEqual(code, 1, "a missing Ollama is a non-zero exit")
        self.assertIn("ollama serve", printed)
        self.assertIn(f"ollama pull {llm.MODEL}", printed)
        self.assertNotIn("idea generation", printed)
        self.assertEqual(Stub.posts, [], "nothing is generated without Ollama")

    def test_missing_model_is_reported_with_the_pull_command(self):
        Stub.body = {"models": []}
        code, printed = self.run_diagnostic()
        self.assertEqual(code, 1)
        self.assertIn("isn't installed", printed)
        self.assertIn(f"ollama pull {llm.MODEL}", printed)
        self.assertEqual(Stub.posts, [])

    def test_a_clean_run_passes_every_check_and_exits_zero(self):
        Stub.sequence = self.canned_run(
            remix=dict(tape_free(make_idea("Cardboard tray", 15)),
                       materials_used=["cardboard"]))
        code, printed = self.run_diagnostic()
        self.assertEqual(code, 0, printed)
        self.assertEqual(printed.count("PASS"), 7)
        self.assertNotIn("FAIL [", printed, "no check line may report a failure")
        self.assertIn("7 of 7 checks passed", printed)
        self.assertEqual(len(Stub.generate_posts()), 7,
                         "exactly one request per check")

    def test_a_validation_failure_is_reported_as_a_category_only(self):
        Stub.sequence = self.canned_run(
            first=json.dumps({"ideas": [make_idea()]}),
            remix=dict(tape_free(make_idea("Cardboard tray", 15)),
                       materials_used=["cardboard"]))
        code, printed = self.run_diagnostic()
        self.assertEqual(code, 1)
        self.assertIn("FAIL [wrong_idea_count]", printed)
        self.assertIn("6 of 7 checks passed", printed)
        self.assertEqual(len(Stub.generate_posts()), 7, "no check is retried")

    def test_every_check_appears_in_the_report(self):
        # The diagnostic is the only place the real behaviour is visible, so a
        # check that silently stopped running would be a quiet loss of
        # coverage. Name them, then look for each name.
        Stub.sequence = self.canned_run(
            remix=dict(tape_free(make_idea("Cardboard tray", 15)),
                       materials_used=["cardboard"]))
        code, printed = self.run_diagnostic()
        self.assertEqual(code, 0, printed)
        for name in ("idea generation 1", "idea generation 2",
                     "ideas, ₹100 + inventory", "build guide, no limits",
                     "build guide, no sharp tools", "remix zero-buy",
                     "remix remove-material"):
            self.assertIn(name, printed)
        self.assertEqual(len(Stub.generate_posts()), 7)

    def test_nothing_raw_is_printed_and_no_file_is_written(self):
        sentinel = "SENTINEL-FROM-THE-MODEL"
        Stub.sequence = self.canned_run(
            first=json.dumps({"ideas": [make_idea(sentinel)]}),
            remix=dict(tape_free(make_idea("Cardboard tray", 15)),
                       materials_used=["cardboard"]))
        before = sorted(path.name for path in Path.cwd().iterdir())
        code, printed = self.run_diagnostic()
        after = sorted(path.name for path in Path.cwd().iterdir())
        self.assertEqual(code, 1)
        self.assertNotIn(sentinel, printed, "model output must never be shown")
        self.assertNotIn("cardboard", printed)
        self.assertEqual(before, after, "the diagnostic must write nothing")
        self.assertIn("FAIL [wrong_idea_count]", printed)

    def test_the_diagnostic_module_can_never_be_run_by_the_suite(self):
        # The invariant that keeps `python -m unittest` off the real model:
        # discovery imports test_real_model, and finds no test case in it.
        cases = [name for name, obj in vars(test_real_model).items()
                 if isinstance(obj, type) and issubclass(obj, unittest.TestCase)]
        self.assertEqual(cases, [], "the real-model module must hold no tests")
        source = Path(test_real_model.__file__).read_text()
        self.assertTrue(source.rstrip().endswith("raise SystemExit(diagnose())"),
                        "the real run must sit behind the __main__ guard")


class RemixGrammarTest(unittest.TestCase):
    """The grammars themselves: shape, required fields, and the ₹0 rules."""

    def test_every_response_carries_a_json_schema_with_required_fields(self):
        for schema in (llm._ideas_schema("100"), llm._guide_schema("100"),
                       llm._remix_schema("100", "simplify")):
            with self.subTest(required=schema["required"]):
                self.assertEqual(schema["type"], "object")
                self.assertTrue(schema["required"])
                for name in schema["required"]:
                    self.assertIn(name, schema["properties"])

    def test_zero_budget_schemas_forbid_the_purchase_fields(self):
        ideas = llm._ideas_schema("zero")["properties"]["ideas"]["items"]
        self.assertEqual(
            ideas["properties"]["additional_items"]["maxItems"], 0)
        self.assertEqual(
            llm._guide_schema("zero")["properties"]["tools_needed"]["maxItems"],
            0)
        self.assertEqual(
            llm._remix_schema("zero", "simplify")["properties"]["idea"][
                "properties"]["additional_items"]["maxItems"], 0)

    def test_a_paid_budget_leaves_the_purchase_fields_open(self):
        self.assertNotIn("maxItems", llm._ideas_schema("flexible")["properties"]
                         ["ideas"]["items"]["properties"]["additional_items"])
        self.assertNotIn("maxItems",
                         llm._guide_schema("500")["properties"]["tools_needed"])
        self.assertNotIn("maxItems", llm._remix_schema("100", "use_it_up")[
            "properties"]["idea"]["properties"]["additional_items"])

    def test_the_shown_minutes_placeholder_is_always_within_the_limit(self):
        for minutes in (15, 30, 60):
            with self.subTest(minutes=minutes):
                shown = llm._ideas_schema_text(min(minutes, 20))
                self.assertEqual(shown.count(f'"estimated_minutes": {min(minutes, 20)}'),
                                 3)
                self.assertEqual(shown.count('"estimated_minutes": 0'), 0)
                guide = llm._guide_schema_text(min(minutes, 20))
                self.assertIn(f'"estimated_minutes": {min(minutes, 20)}', guide)
                self.assertNotIn('"estimated_minutes": 0', guide)

    def test_the_ideas_template_shows_three_objects(self):
        shown = llm._ideas_schema_text(30)
        self.assertEqual(shown.count('"why_it_fits"'), llm._IDEA_COUNT)


class EmptyMaterialsTest(unittest.TestCase):
    """A project that names no materials is missing information, not valid."""

    def test_idea_without_materials_is_rejected(self):
        ideas = make_three_ideas()
        for one in ideas:
            one["materials_used"] = []
        ideas, error = llm._validate(
            {"ideas": ideas}, 30, "zero", inventory="cardboard, tape")
        self.assertFalse(ideas, "an idea with no materials must be rejected")
        self.assertIn("made from", error)

    def test_guide_without_materials_is_rejected(self):
        guide = make_guide()
        guide["materials_used"] = []
        cleaned, error = llm._validate_guide(guide, "cardboard, tape", 30, "zero", "")
        self.assertFalse(cleaned, "a guide with no materials must be rejected")
        self.assertIn("build it from", error)

    def test_materials_of_only_blank_entries_count_as_empty(self):
        guide = make_guide(materials=["  ", ""])
        cleaned, error = llm._validate_guide(guide, "cardboard, tape", 30, "zero", "")
        self.assertFalse(cleaned, "blank material entries are not materials")
        self.assertIn("build it from", error)

    def test_schema_forbids_an_empty_materials_list_at_the_grammar(self):
        # The validator would reject it anyway; this stops the model ever
        # wasting an inference call on one.
        for schema in (llm._guide_schema("zero"), llm._ideas_schema(30)):
            materials = schema["properties"].get("materials_used")
            if materials is None:
                materials = schema["properties"]["ideas"]["items"]["properties"]["materials_used"]
            self.assertEqual(materials.get("minItems"), 1)


if __name__ == "__main__":
    unittest.main()