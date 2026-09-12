"""Unit and integration tests using only the Python standard library."""

from __future__ import annotations

import io
import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .actions import parse_drawing_response, parse_forum_turn, parse_llm_response, parse_pixel_drawing
from .agent_runner import resumable_run, resume_saved_run, run_competing_teams, run_drawing, run_team_drawing
from .cli import mapping_actions
from .grid_image import write_grid_png
from .openrouter_client import OPENROUTER_URL, build_vision_request, chat_completion, request_preview, response_text
from .prompts import (
    DEFAULT_PROMPT_TEMPLATE_ID,
    apple_prompt,
    artwork_bounds,
    constructed_prompt,
    forum_response_schema,
    pixel_response_schema,
    prompt_template_records,
)
from .references import DEFAULT_OBJECTS, available_objects, reference_rows, save_reference, write_reference_png
from .server import create_server
from .store import CanvasError, CanvasStore
from .single_batch import _analyze_run, _jobs


class StoreTests(unittest.TestCase):
    def test_pixel_is_persisted_and_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canvas.json"
            store = CanvasStore(path)
            event = store.set_pixel(4, 7, "purple", "test")
            self.assertTrue(event["changed"])
            self.assertEqual(CanvasStore(path).snapshot()["pixels"][7][4], "purple")

    def test_invalid_coordinate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CanvasStore(Path(directory) / "canvas.json")
            with self.assertRaises(CanvasError):
                store.set_pixel(32, 0, "blue")

    def test_configurable_16_by_16_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CanvasStore(Path(directory) / "canvas.json", 16, 16)
            store.set_pixel(15, 15, "red")
            with self.assertRaises(CanvasError):
                store.set_pixel(16, 15, "red")

    def test_store_can_resize_and_clears_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CanvasStore(Path(directory) / "canvas.json", 16, 16)
            store.set_pixel(3, 4, "red")
            event = store.resize(24, "test")
            self.assertEqual((event["state"]["width"], event["state"]["height"]), (24, 24))
            self.assertEqual(sum(value is not None for row in event["state"]["pixels"] for value in row), 0)

    def test_store_can_restore_a_canvas_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canvas.json"
            store = CanvasStore(path, 16, 16)
            pixels = [[None for _ in range(16)] for _ in range(16)]
            pixels[4][3] = "green"
            event = store.replace_pixels(pixels, "resume")
            self.assertEqual(event["type"], "snapshot")
            self.assertEqual(event["state"]["pixels"][4][3], "green")
            self.assertEqual(CanvasStore(path, 16, 16).snapshot()["pixels"][4][3], "green")


class ParserTests(unittest.TestCase):
    def test_fenced_llm_response(self) -> None:
        actions = parse_llm_response('```json\n{"actions":[{"action":"place","x":3,"y":5,"color":"green"}]}\n```')
        self.assertEqual(actions[0], {"action": "place", "x": 3, "y": 5, "color": "green"})

    def test_action_limit(self) -> None:
        raw = json.dumps({"actions": [{"x": i, "y": 0, "color": "red"} for i in range(5)]})
        with self.assertRaises(CanvasError):
            parse_llm_response(raw)

    def test_mapping_actions_are_exactly_row_major(self) -> None:
        actions = list(mapping_actions())
        self.assertEqual(len(actions), 1024)
        self.assertEqual((actions[0]["x"], actions[0]["y"]), (0, 0))
        self.assertEqual((actions[31]["x"], actions[31]["y"]), (31, 0))
        self.assertEqual((actions[32]["x"], actions[32]["y"]), (0, 1))
        self.assertEqual((actions[-1]["x"], actions[-1]["y"]), (31, 31))

    def test_16_mapping_and_apple_response(self) -> None:
        actions = list(mapping_actions(16, 16))
        self.assertEqual(len(actions), 256)
        self.assertEqual((actions[15]["x"], actions[15]["y"]), (15, 0))
        self.assertEqual((actions[16]["x"], actions[16]["y"]), (0, 1))
        pixels = parse_pixel_drawing('{"pixels":[{"x":15,"y":15,"color":"red"}]}', 16)
        self.assertEqual(pixels[0]["color"], "red")

    def test_iterative_response_and_prompt_contract(self) -> None:
        actions, complete = parse_drawing_response('{"pixels":[{"x":1,"y":1,"color":"red"}]}', 16, iterative=True)
        self.assertEqual(len(actions), 1)
        self.assertFalse(complete)
        actions, complete = parse_drawing_response('{"pixels":[]}', 16, iterative=True)
        self.assertEqual(actions, [])
        self.assertTrue(complete)
        prompt = constructed_prompt(16, "apple", 8, 3)
        self.assertIn("This is step 3", prompt)
        self.assertIn("Change at most 8 pixels", prompt)
        self.assertIn("last saved step", prompt)
        baseline = constructed_prompt(16, "apple", 0, 1)
        self.assertNotIn("last saved step", baseline)
        self.assertNotIn("Change at most", baseline)
        schema = pixel_response_schema(16, 8)["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["pixels"]["maxItems"], 8)
        self.assertEqual(schema["properties"]["pixels"]["minItems"], 0)
        self.assertIn("erase", schema["properties"]["pixels"]["items"]["properties"]["color"]["enum"])
        self.assertNotIn("complete", schema["properties"])

    def test_drawing_response_can_erase_a_pixel(self) -> None:
        actions, complete = parse_drawing_response('{"pixels":[{"x":3,"y":4,"color":"erase"}]}', 16, iterative=True)
        self.assertEqual(actions, [{"action": "erase", "x": 3, "y": 4, "color": None}])
        self.assertFalse(complete)
        self.assertIn("Erasing counts toward the pixel limit", constructed_prompt(16, "apple", 8, 1))

    def test_forum_turn_contract_is_strict_and_conditional(self) -> None:
        colored = parse_forum_turn(
            '{"action":"color","pixels":[{"x":3,"y":4,"color":"red"}],"message":""}',
            16,
        )
        self.assertEqual((colored["kind"], colored["actions"][0]["color"], colored["complete"]), ("color", "red", False))
        written = parse_forum_turn('{"action":"write_team_forum","pixels":[],"message":" Use the upper left. "}', 16)
        self.assertEqual(written["message"], "Use the upper left.")
        with self.assertRaises(CanvasError):
            parse_forum_turn('{"action":"read_team_forum","pixels":[],"message":""}', 16)
        with self.assertRaises(CanvasError):
            parse_forum_turn('{"action":"read_public_forum","pixels":[],"message":""}', 16)
        with self.assertRaises(CanvasError):
            parse_forum_turn(
                '{"action":"read_public_forum","pixels":[],"message":""}',
                16,
                public_forum_enabled=True,
            )
        public = parse_forum_turn(
            '{"action":"write_public_forum","pixels":[],"message":"Avoid the middle."}',
            16,
            public_forum_enabled=True,
        )
        self.assertEqual(public["forum"], "public")
        with self.assertRaises(CanvasError):
            parse_forum_turn('{"action":"write_team_forum","pixels":[{"x":1,"y":1,"color":"red"}],"message":"hello"}', 16)

    def test_forum_prompt_and_schema_expose_one_action_per_turn(self) -> None:
        prompt = constructed_prompt(
            16,
            "apple",
            8,
            3,
            team_size=2,
            member_index=1,
            forum_enabled=True,
            public_forum_enabled=True,
            forum_posts={
                "team": [
                    {"round": index, "team": 1, "member": 2, "message": f"plan-{index}"}
                    for index in range(1, 7)
                ],
                "public": [],
            },
            forum_activity=[{"round": 2, "kind": "write_team_forum", "forum": "team", "message": "plan-2"}],
        )
        self.assertIn("All agents and forums begin this run from a clean state", prompt)
        self.assertIn("Choose exactly one action for this turn", prompt)
        self.assertNotIn("read_team_forum", prompt)
        self.assertIn("write_public_forum", prompt)
        self.assertNotIn("cannot communicate", prompt)
        self.assertIn("Communication is possible only through", prompt)
        self.assertIn("If progress stalls", prompt)
        self.assertIn("Make forum posts actionable", prompt)
        self.assertIn("complete history of every forum", prompt)
        self.assertIn("plan-1", prompt)
        self.assertIn("plan-2", prompt)
        self.assertIn("plan-6", prompt)
        private_schema = forum_response_schema(16, 8, False)["json_schema"]["schema"]
        public_schema = forum_response_schema(16, 8, True)["json_schema"]["schema"]
        self.assertNotIn("read_public_forum", private_schema["properties"]["action"]["enum"])
        self.assertNotIn("read_public_forum", public_schema["properties"]["action"]["enum"])
        self.assertIn("write_public_forum", public_schema["properties"]["action"]["enum"])
        before_post = constructed_prompt(
            16,
            "apple",
            8,
            1,
            team_size=2,
            member_index=1,
            forum_enabled=True,
        )
        self.assertIn("Before coloring any pixels, begin the run by writing one concise coordination post", before_post)
        self.assertNotIn("Before coloring any pixels, begin the run", prompt)

    def test_prompt_versions_and_custom_variables_render(self) -> None:
        records = {record["id"]: record for record in prompt_template_records()}
        self.assertEqual(set(records), {"forum_optional_v1", "forum_suggestions_v2a", "forum_protocol_v3a", "forum_directed_v2", "forum_conflict_v3"})
        common = {
            "size": 32, "subject": "fish", "step_size": 8, "step_number": 3,
            "team_size": 2, "member_index": 1, "competing_teams": 4, "team_index": 2,
            "team_objectives": ["apple", "orange", "tree", "fish"],
            "forum_enabled": True, "public_forum_enabled": True, "forum_posts": {}, "forum_activity": [],
        }
        private_common = dict(common)
        private_common.update(competing_teams=1, team_index=1, team_objectives=None, public_forum_enabled=False)
        optional = constructed_prompt(**private_common, prompt_template_id="forum_optional_v1")
        public_optional = constructed_prompt(**common, prompt_template_id="forum_optional_v1")
        suggested = constructed_prompt(**common, prompt_template_id="forum_suggestions_v2a")
        protocol = constructed_prompt(**private_common, prompt_template_id="forum_protocol_v3a")
        public_protocol = constructed_prompt(**common, prompt_template_id="forum_protocol_v3a")
        directed = constructed_prompt(**common, prompt_template_id="forum_directed_v2")
        latest = constructed_prompt(**common, prompt_template_id=DEFAULT_PROMPT_TEMPLATE_ID)
        self.assertNotIn("Before coloring any pixels", optional)
        self.assertNotIn("Competition context", optional)
        self.assertNotIn("Public forum", optional)
        self.assertNotIn("other team", optional.casefold())
        self.assertIn("Your only goal is to build one clear fish", optional)
        self.assertIn("There are 4 teams working on the same shared canvas", public_optional)
        self.assertIn("The team assignments are: Team 1: apple; Team 2: orange; Team 3: tree; Team 4: fish", public_optional)
        self.assertIn("A public forum also exists", public_optional)
        self.assertIn("Member 1 is your team's leader and designated public spokesperson", public_optional)
        self.assertIn("should not write directly to the public forum", public_optional)
        self.assertIn("`write_public_forum`", public_optional)
        self.assertNotIn("Forum suggestions:", public_optional)
        self.assertNotIn("Forum suggestions:", optional)
        self.assertIn("Forum suggestions:", suggested)
        self.assertNotIn("Forum use is optional", suggested)
        self.assertNotIn("Before coloring any pixels", suggested)
        self.assertIn("Member 1 can use the public forum to announce the team's intended region", suggested)
        self.assertIn("Round 0 is Member 1's only special lead turn", protocol)
        self.assertIn("From Round 1 onward, every member has the same role", protocol)
        self.assertIn("You must use `write_team_forum` instead of `color`", protocol)
        self.assertNotIn("checkpoint", protocol.casefold())
        self.assertIn("team leader and public spokesperson, must use `write_public_forum`", public_protocol)
        self.assertIn("If you are Member 1, you must use `write_public_forum`", public_protocol)
        self.assertIn("Member 1's only continuing distinction", public_protocol)
        self.assertIn("Do not write directly to the public forum", public_protocol)
        self.assertIn("most useful available forum", directed)
        self.assertIn("proposed drawing region to the public forum", latest)
        custom = constructed_prompt(
            **common,
            prompt_template_text="Team {{team_index}} of {{competing_teams}} draws {{subject}} in {{progress_label}} {{step_number}}.{{#public_forum}} Public enabled.{{/public_forum}}",
        )
        self.assertEqual(custom, "Team 2 of 4 draws fish in round 3. Public enabled.")
        with self.assertRaisesRegex(ValueError, "unknown placeholders"):
            constructed_prompt(**common, prompt_template_text="{{unknown_setting}}")

        solo_v1 = constructed_prompt(
            16,
            "apple",
            8,
            2,
            team_size=1,
            member_index=1,
            forum_enabled=True,
            reference_tag="apple",
            personal_history=[{"round": 1, "changes": [{"x": 2, "y": 3, "color": "red"}]}],
            forum_posts={"team": [
                {"round": index, "team": 1, "member": 1, "message": f"note-{index}"}
                for index in range(1, 8)
            ]},
            prompt_template_id="forum_optional_v1",
        )
        self.assertIn("You are the only agent working on this drawing", solo_v1)
        self.assertIn("complete history of your own edits", solo_v1)
        self.assertIn("Step 1: (2, 3)=red", solo_v1)
        self.assertIn("note-1", solo_v1)
        self.assertIn("note-7", solo_v1)
        self.assertNotIn("public forum", solo_v1.casefold())

    def test_team_prompt_is_conditional(self) -> None:
        solo = constructed_prompt(16, "apple", 8, 1, team_size=1, member_index=1)
        self.assertNotIn("team", solo.casefold())
        self.assertNotIn("member", solo.casefold())
        team = constructed_prompt(16, "apple", 8, 1, team_size=2, member_index=1)
        self.assertIn("There are 2 members in your team", team)
        self.assertIn("you are member 1", team)
        self.assertIn("act simultaneously", team)
        self.assertIn("This is round 1", team)

    def test_competing_prompt_names_team_count_and_artwork_bound(self) -> None:
        prompt = constructed_prompt(
            32,
            "orange",
            8,
            1,
            team_size=2,
            member_index=1,
            competing_teams=4,
            team_index=3,
            team_objectives=["apple", "pear", "orange", "grape"],
            artwork_size=16,
        )
        self.assertIn("You are member 1 of 2 in Team 3", prompt)
        self.assertIn("There are 4 teams drawing different images", prompt)
        self.assertIn("Team 1: apple; Team 2: pear; Team 3: orange; Team 4: grape", prompt)
        self.assertIn("may be intentional work for that team", prompt)
        self.assertIn("no larger than 16x16 cells", prompt)
        self.assertIn("Choose where to place", prompt)
        self.assertNotIn("centered", prompt.casefold())

    def test_assigned_center_sets_an_exact_region_and_schema_bounds(self) -> None:
        bounds = artwork_bounds(32, 16, (8, 8))
        self.assertEqual(bounds, (0, 15, 0, 15))
        prompt = constructed_prompt(32, "apple", 8, 1, artwork_size=16, center=(8, 8))
        self.assertIn("assigned center point is (8, 8)", prompt)
        self.assertIn("x=0..15, y=0..15", prompt)
        self.assertIn("exactly one apple", prompt)
        self.assertNotIn("Choose where to place", prompt)
        pixel = pixel_response_schema(32, 8, bounds)["json_schema"]["schema"]["properties"]["pixels"]["items"]["properties"]
        self.assertEqual((pixel["x"]["minimum"], pixel["x"]["maximum"]), (0, 15))
        self.assertEqual((pixel["y"]["minimum"], pixel["y"]["maximum"]), (0, 15))

    def test_personal_history_is_optional_and_compact(self) -> None:
        without_history = constructed_prompt(16, "apple", 8, 2, team_size=2, member_index=1)
        self.assertNotIn("Personal edit history:", without_history)
        empty_history = constructed_prompt(16, "apple", 8, 1, team_size=2, member_index=1, personal_history=[])
        self.assertIn("You have no earlier successful edits", empty_history)
        with_history = constructed_prompt(
            16,
            "apple",
            8,
            3,
            team_size=2,
            member_index=1,
            personal_history=[{"round": 1, "changes": [{"x": 4, "y": 5, "color": "red"}, {"x": 6, "y": 5, "color": "erase"}]}],
        )
        self.assertIn("Round 1: (4, 5)=red, (6, 5)=erase", with_history)
        self.assertIn("current canvas is authoritative", with_history)

    def test_object_description_and_reference_prompt(self) -> None:
        default_apple = constructed_prompt(16, "apple", 8, 1, team_size=2, member_index=1)
        editable_apple = constructed_prompt(16, "apple", 8, 1, team_size=2, member_index=1, description=DEFAULT_OBJECTS["apple"]["description"])
        self.assertEqual(default_apple, editable_apple)
        orange = constructed_prompt(16, "orange", 8, 1, description=DEFAULT_OBJECTS["orange"]["description"])
        self.assertIn("rounded orange body", orange)
        tree = constructed_prompt(16, "tree", 8, 1, description=DEFAULT_OBJECTS["tree"]["description"])
        fish = constructed_prompt(16, "fish", 8, 1, description=DEFAULT_OBJECTS["fish"]["description"])
        self.assertIn("broad green canopy", tree)
        self.assertIn("blue body", fish)
        referenced = constructed_prompt(16, "orange", 8, 1, description=DEFAULT_OBJECTS["orange"]["description"], reference_tag="orange")
        self.assertIn("second is the completed orange reference", referenced)
        self.assertNotIn("Reference target:", orange)
        rows = reference_rows("orange", 32)
        self.assertEqual((len(rows), len(rows[0])), (32, 32))
        self.assertIn("orange", {value for row in rows for value in row})
        tree = reference_rows("tree", 16)
        fish = reference_rows("fish", 16)
        self.assertTrue({"green", "orange", "black"}.issubset({value for row in tree for value in row}))
        self.assertTrue({"blue", "yellow", "black"}.issubset({value for row in fish for value in row}))

    def test_generated_canvas_can_replace_an_object_reference(self) -> None:
        rows = [[None for _ in range(16)] for _ in range(16)]
        rows[7][8] = "purple"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("harness.references.REFERENCE_ROOT", root), patch("harness.references.REFERENCE_INDEX", root / "saved_references.json"):
                tag, record = save_reference(subject="apple", description="A purple test apple.", rows=rows, source_run_id="drawing_test")
                self.assertEqual(tag, "apple")
                self.assertEqual(record["reference_source"], "saved_run")
                self.assertEqual(available_objects()["apple"]["description"], "A purple test apple.")
                self.assertEqual(reference_rows("apple", 16)[7][8], "purple")
                self.assertTrue((root / "apple.png").is_file())

    def test_grid_png_and_openrouter_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
            "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash-vision-exp",
            "OPENROUTER_PROVIDER": "deepinfra/fp8",
        }):
            image = write_grid_png(Path(directory) / "grid.png", 16)
            data = image.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", data[16:24]), (512, 512))
            payload = build_vision_request(apple_prompt(16), image, pixel_response_schema(16))
            self.assertEqual(payload["provider"]["order"], ["deepinfra/fp8"])
            self.assertFalse(payload["provider"]["allow_fallbacks"])
            budgeted = build_vision_request(apple_prompt(16), image, pixel_response_schema(16), reasoning="low", reasoning_max_tokens=1000)
            self.assertEqual(budgeted["reasoning"], {"max_tokens": 1000, "exclude": True})
            glm = build_vision_request(apple_prompt(16), image, pixel_response_schema(16), model="z-ai/glm-5.3-flash", provider="relace/fp4", reasoning="low", reasoning_max_tokens=500)
            self.assertEqual(glm["response_format"], {"type": "json_object"})
            self.assertEqual(glm["provider"]["only"], ["relace/fp4"])
            self.assertEqual(glm["provider"]["quantizations"], ["fp4"])
            self.assertTrue(glm["provider"]["allow_fallbacks"])
            streamlake = build_vision_request(apple_prompt(16), image, pixel_response_schema(16), model="z-ai/glm-5.3-flash", provider="streamlake/fp8", reasoning="low", reasoning_max_tokens=500)
            self.assertEqual(streamlake["response_format"], {"type": "json_object"})
            self.assertEqual(streamlake["provider"]["only"], ["streamlake/fp8"])
            self.assertEqual(streamlake["provider"]["quantizations"], ["fp8"])
            self.assertTrue(streamlake["provider"]["allow_fallbacks"])
            qwen = build_vision_request(apple_prompt(16), image, pixel_response_schema(16, 8), model="qwen/qwen3.8-27b", provider="reka/fp8", reasoning="low", reasoning_max_tokens=500)
            self.assertEqual(qwen["provider"]["order"], ["reka/fp8"])
            self.assertFalse(qwen["provider"]["allow_fallbacks"])
            self.assertEqual(qwen["response_format"]["type"], "json_schema")
            muse = build_vision_request(apple_prompt(16), image, pixel_response_schema(16, 8), model="meta/muse-spark-1.3-contributor", provider="meta", reasoning="low", reasoning_max_tokens=500)
            self.assertEqual(muse["model"], "meta/muse-spark-1.3-contributor")
            self.assertEqual(muse["provider"]["order"], ["meta"])
            self.assertFalse(muse["provider"]["allow_fallbacks"])
            self.assertEqual(muse["response_format"]["type"], "json_schema")
            luna = build_vision_request(apple_prompt(16), image, pixel_response_schema(16, 8), model="openai/gpt-5.6-luna", provider="openai/flex", reasoning="low", reasoning_max_tokens=500)
            self.assertEqual(luna["provider"]["order"], ["openai/flex"])
            self.assertEqual(luna["reasoning"], {"effort": "low", "exclude": True})
            self.assertNotIn("temperature", luna)
            self.assertEqual(luna["response_format"]["type"], "json_schema")
            self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertNotIn("base64,", request_preview(payload, image)["messages"][0]["content"][1]["image_url"]["url"])
            rows = [[None for _ in range(16)] for _ in range(16)]
            rows[4][7] = "red"
            populated = write_grid_png(Path(directory) / "populated.png", 16, rows=rows, colors={"red": "#ff0000"})
            self.assertNotEqual(image.read_bytes(), populated.read_bytes())
            reference = write_reference_png(Path(directory) / "reference.png", "apple", 16)
            with_reference = build_vision_request(apple_prompt(16), image, pixel_response_schema(16), reference_image_path=reference)
            self.assertEqual(len(with_reference["messages"][0]["content"]), 3)
            preview = request_preview(with_reference, image, reference)
            self.assertTrue(preview["messages"][0]["content"][2]["image_url"]["url"].startswith("<base64 PNG: reference.png"))

    def test_reasoning_limit_failure_is_explained(self) -> None:
        response = {"choices": [{"finish_reason": "length", "message": {"content": None}}], "usage": {"completion_tokens_details": {"reasoning_tokens": 12288}}}
        with self.assertRaisesRegex(RuntimeError, "12288 reasoning tokens"):
            response_text(response)

    def test_openrouter_429_is_retried_and_recorded(self) -> None:
        limited = HTTPError(
            OPENROUTER_URL,
            429,
            "rate limited",
            {"Retry-After": "1"},
            io.BytesIO(b'{"error":{"message":"temporary"}}'),
        )

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        success = Response(b'{"choices":[],"usage":{}}')
        with patch("harness.openrouter_client.urlopen", side_effect=[limited, success]), patch("harness.openrouter_client.sleep") as pause, patch("harness.openrouter_client.random.uniform", return_value=0.2):
            result = chat_completion({"model": "m"}, "key", max_rate_limit_retries=3)
        self.assertEqual(result["_client"]["attempts"], 2)
        self.assertEqual(result["_client"]["rate_limit_retries"], 1)
        self.assertEqual(result["_client"]["retry_events"][0]["delay_seconds"], 10.0)
        pause.assert_called_once_with(10.0)

    def test_openrouter_timeout_is_retried_and_recorded(self) -> None:
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        success = Response(b'{"choices":[],"usage":{}}')
        with patch("harness.openrouter_client.urlopen", side_effect=[TimeoutError("read timed out"), success]), patch("harness.openrouter_client.sleep") as pause, patch("harness.openrouter_client.random.uniform", return_value=0.2):
            result = chat_completion({"model": "m"}, "key", max_transient_retries=2)
        self.assertEqual(result["_client"]["attempts"], 2)
        self.assertEqual(result["_client"]["transient_retries"], 1)
        self.assertEqual(result["_client"]["retry_events"][0]["kind"], "timeout")
        pause.assert_called_once_with(1.2)

    def test_openrouter_uses_complete_requested_rate_limit_schedule(self) -> None:
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        limited = [
            HTTPError(OPENROUTER_URL, 429, "rate limited", {}, io.BytesIO(b'{"error":"limited"}'))
            for _ in range(7)
        ]
        success = Response(b'{"choices":[],"usage":{}}')
        with patch("harness.openrouter_client.urlopen", side_effect=[*limited, success]), patch("harness.openrouter_client.sleep") as pause:
            result = chat_completion({"model": "m"}, "key")
        self.assertEqual([call.args[0] for call in pause.call_args_list], [10.0, 10.0, 20.0, 20.0, 60.0, 60.0, 120.0])
        self.assertEqual(result["_client"]["rate_limit_retries"], 7)
        self.assertEqual(result["_client"]["attempts"], 8)

    def test_invalid_model_json_is_retried_immediately_and_recorded(self) -> None:
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        invalid = Response(b'{"choices":[{"message":{"content":"{\\"pixels\\":[{\\"x\\":1"}}]}')
        valid = Response(b'{"choices":[{"message":{"content":"{\\"pixels\\":[]}"}}]}')
        events: list[dict] = []
        with patch("harness.openrouter_client.urlopen", side_effect=[invalid, valid]), patch("harness.openrouter_client.sleep") as pause:
            result = chat_completion(
                {"model": "m"},
                "key",
                validate_response=lambda candidate: parse_drawing_response(response_text(candidate), 16, iterative=True),
                on_retry=events.append,
            )
        pause.assert_not_called()
        self.assertEqual(result["_client"]["attempts"], 2)
        self.assertEqual(result["_client"]["invalid_output_retries"], 1)
        self.assertEqual(events[0]["kind"], "invalid_output")
        self.assertEqual(events[0]["delay_seconds"], 0.0)
        self.assertIn("invalid_response", events[0])

    def test_iterative_runner_saves_each_step(self) -> None:
        responses = [
            {"model": "m", "provider": "p", "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.01}, "choices": [{"message": {"content": '{"pixels":[{"x":1,"y":1,"color":"red"}]}'}}]},
            {"model": "m", "provider": "p", "usage": {"prompt_tokens": 11, "completion_tokens": 3, "cost": 0.02}, "choices": [{"message": {"content": '{"pixels":[{"x":2,"y":1,"color":"red"}]}'}}]},
            {"model": "m", "provider": "p", "usage": {"prompt_tokens": 12, "completion_tokens": 2, "cost": 0.01}, "choices": [{"message": {"content": '{"pixels":[]}'}}]},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=responses):
            seen = []
            result = run_drawing(subject="apple", step_size=1, max_steps=4, size=16, model="m", provider="p", temperature=0.2, max_tokens=512, reasoning="disabled", on_step=seen.append)
            run = Path(result["run_path"])
            self.assertEqual((result["steps"], result["filled"], result["stop_reason"]), (3, 2, "model_complete"))
            self.assertEqual(result["usage"]["prompt_tokens"], 33)
            self.assertEqual(len(list((run / "snapshots").glob("*.png"))), 3)
            self.assertEqual(len(seen), 3)

    def test_team_runner_calls_members_and_applies_both(self) -> None:
        responses = [
            {"model": "glm", "provider": "relace", "usage": {"completion_tokens_details": {"reasoning_tokens": 1}}, "choices": [{"message": {"content": '{"pixels":[{"x":0,"y":1,"color":"red"}]}'}}]},
            {"model": "glm", "provider": "relace", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001, "completion_tokens_details": {"reasoning_tokens": 1}}, "choices": [{"message": {"content": '{"pixels":[{"x":1,"y":1,"color":"red"}]}'}}]},
            {"model": "glm", "provider": "relace", "usage": {"prompt_tokens": 11, "completion_tokens": 2, "cost": 0.002, "completion_tokens_details": {"reasoning_tokens": 2}}, "choices": [{"message": {"content": '{"pixels":[{"x":2,"y":1,"color":"green"}]}'}}]},
            {"model": "glm", "provider": "relace", "usage": {"completion_tokens_details": {"reasoning_tokens": 3}}, "choices": [{"message": {"content": '{"pixels":[]}'}}]},
            {"model": "glm", "provider": "relace", "usage": {"completion_tokens_details": {"reasoning_tokens": 4}}, "choices": [{"message": {"content": '{"pixels":[]}'}}]},
        ]
        members = [
            {"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500},
            {"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=responses):
            seen = []
            result = run_team_drawing(subject="apple", step_size=1, max_steps=10, size=16, members=members, temperature=0.2, max_tokens=512, on_step=seen.append)
            run = Path(result["run_path"])
            self.assertEqual((result["team_size"], result["rounds"], result["calls"]), (2, 3, 5))
            self.assertEqual((result["configured_rounds_executed"], result["opening_turns"]), (2, 1))
            self.assertEqual((result["filled"], result["stop_reason"]), (3, "team_complete"))
            self.assertEqual(result["usage"]["cost"], 0.003)
            self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 11)
            self.assertEqual(len(list((run / "snapshots").glob("*.png"))), 5)
            self.assertEqual({item["agent"] for item in seen}, {1, 2})
            saved_history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            self.assertEqual((saved_history[0]["round"], saved_history[0]["agent"], saved_history[0]["opening_turn"]), (0, 1, True))
            self.assertTrue(all(item["latency_seconds"] >= 0 for item in saved_history))
            self.assertTrue(all(item["request_started_utc"].endswith("Z") for item in saved_history))

    def test_team_runner_supports_batch_identity_request_guard_and_usage_events(self) -> None:
        response = {
            "id": "response-id",
            "model": "m",
            "provider": "p",
            "usage": {"cost": 0.001},
            "choices": [{"message": {"content": '{"pixels":[]}'}}],
        }

        class Guard:
            def __init__(self) -> None:
                self.calls = 0
                self.active = 0

            def __call__(self, _request: dict):
                return self

            def __enter__(self):
                self.calls += 1
                self.active += 1

            def __exit__(self, *_args):
                self.active -= 1

        guard = Guard()

        def respond(*_args, **_kwargs) -> dict:
            self.assertGreater(guard.active, 0)
            return json.loads(json.dumps(response))

        members = [
            {"model": "m", "provider": "p", "reasoning": "low", "reasoning_max_tokens": 500},
            {"model": "m", "provider": "p", "reasoning": "low", "reasoning_max_tokens": 500},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=respond):
            seen: list[dict] = []
            result = run_team_drawing(
                subject="apple",
                step_size=1,
                max_steps=1,
                size=16,
                members=members,
                temperature=0.2,
                max_tokens=512,
                on_step=seen.append,
                run_id_override="drawing_batch_test",
                request_guard=guard,
            )
            self.assertEqual(Path(result["run_path"]).name, "drawing_batch_test")
            self.assertEqual((guard.calls, len(seen)), (3, 3))
            self.assertTrue(all(item["tag"].startswith("round_") for item in seen))
            self.assertTrue(all(item["usage"]["cost"] == 0.001 for item in seen))

    def test_v1_batch_plan_has_40_runs_and_balanced_heterogeneous_leaders(self) -> None:
        config = json.loads((Path(__file__).parent.parent / "configs" / "single_v1_core.json").read_text(encoding="utf-8"))
        jobs = _jobs(config)
        self.assertEqual(len(jobs), 40)
        self.assertEqual(sorted(job["randomized_order"] for job in jobs), list(range(1, 41)))
        heterogeneous = [job for job in jobs if job["condition"] == "heterogeneous"]
        leader_counts = {name: sum(job["leader"] == name for job in heterogeneous) for name in ("GLM", "Gemini", "Qwen", "Luna")}
        self.assertLessEqual(max(leader_counts.values()) - min(leader_counts.values()), 1)

    def test_v1_batch_analysis_calculates_same_round_pixel_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "drawing_analysis_test"
            (run / "actions").mkdir(parents=True)
            actions = [
                {"action_kind": "color", "actions": [{"x": 1, "y": 1, "color": "red"}, {"x": 2, "y": 2, "color": "red"}]},
                {"action_kind": "color", "actions": [{"x": 1, "y": 1, "color": "green"}, {"x": 3, "y": 3, "color": "red"}]},
            ]
            for index, value in enumerate(actions, start=1):
                (run / "actions" / f"a{index}.json").write_text(json.dumps(value), encoding="utf-8")
            history = [
                {"round": 1, "display_round": 1, "agent": 1, "actions": "actions/a1.json", "pixels_changed": 2},
                {"round": 1, "display_round": 1, "agent": 2, "actions": "actions/a2.json", "pixels_changed": 2},
            ]
            (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
            (run / "result.json").write_text(json.dumps({"filled": 3}), encoding="utf-8")
            (run / "usage.json").write_text(json.dumps({"cost": 0.02}), encoding="utf-8")
            with patch("harness.single_batch.ROOT", root):
                metrics = _analyze_run(run)
            round_metrics = metrics["rounds"][0]
            self.assertEqual((round_metrics["unique_target_coordinates"], round_metrics["shared_target_coordinates"]), (3, 1))
            self.assertAlmostEqual(round_metrics["shared_coordinate_rate"], 1 / 3)
            self.assertAlmostEqual(round_metrics["overlapping_action_rate"], 1 / 4)
            self.assertEqual(round_metrics["different_color_conflicts"], 1)

    def test_team_runner_applies_model_eraser(self) -> None:
        responses = [
            {"model": "glm", "provider": "relace", "usage": {}, "choices": [{"message": {"content": '{"pixels":[{"x":3,"y":4,"color":"erase"}]}'}}]},
            {"model": "glm", "provider": "relace", "usage": {}, "choices": [{"message": {"content": '{"pixels":[]}'}}]},
        ]
        members = [{"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500}]
        initial = [[None for _ in range(16)] for _ in range(16)]
        initial[4][3] = "red"
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=responses):
            seen = []
            result = run_team_drawing(subject="apple", step_size=1, max_steps=2, size=16, members=members, temperature=0.2, max_tokens=512, initial_rows=initial, on_step=seen.append)
            self.assertEqual((result["filled"], result["changes"]), (0, 1))
            self.assertIsNone(seen[0]["actions"][0]["color"])

    def test_team_runner_persists_and_forwards_retry_events(self) -> None:
        invalid_response = {"choices": [{"message": {"content": "{broken"}}]}
        valid_response = {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[]}'}}]}

        def respond(_request: dict, _key: str, **kwargs) -> dict:
            kwargs["on_retry"]({
                "kind": "invalid_output",
                "retry": 1,
                "max_retries": 3,
                "delay_seconds": 0.0,
                "message": "model output was not valid JSON",
                "invalid_response": invalid_response,
            })
            return valid_response

        member = {"model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=respond):
            seen: list[dict] = []
            result = run_team_drawing(
                subject="apple",
                step_size=1,
                max_steps=1,
                size=16,
                members=[member],
                temperature=0.2,
                max_tokens=512,
                on_retry=seen.append,
            )
            run = Path(result["run_path"])
            saved = json.loads(next((run / "retries").glob("*.jsonl")).read_text(encoding="utf-8"))
            self.assertEqual((saved["round"], saved["team"], saved["member"], saved["opening_turn"]), (0, 1, 1, True))
            self.assertEqual(saved["kind"], "invalid_output")
            self.assertEqual(len(list((run / "responses").glob("*.invalid_output_*.json"))), 2)
            self.assertEqual(seen[0]["delay_seconds"], 0.0)

    def test_team_center_rejects_edits_outside_its_allocated_region(self) -> None:
        response = {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[{"x":15,"y":15,"color":"red"}]}'}}]}
        member = {"model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", return_value=response):
            result = run_team_drawing(
                subject="apple",
                step_size=1,
                max_steps=2,
                size=16,
                artwork_size=8,
                center=(4, 4),
                members=[member],
                temperature=0.2,
                max_tokens=512,
            )
            history = json.loads((Path(result["run_path"]) / "history.json").read_text(encoding="utf-8"))
            self.assertEqual((result["filled"], result["stop_reason"]), (0, "stalled"))
            self.assertEqual(history[0]["pixels_rejected_outside_region"], 1)

    def test_team_runner_keeps_personal_histories_separate(self) -> None:
        def respond(request: dict, _key: str, **_kwargs) -> dict:
            prompt = request["messages"][0]["content"][0]["text"]
            if "This is round 1" in prompt:
                pixel = '{"pixels":[{"x":1,"y":1,"color":"red"}]}' if "you are member 1" in prompt else '{"pixels":[{"x":2,"y":1,"color":"green"}]}'
            else:
                pixel = '{"pixels":[]}'
            return {"model": "glm", "provider": "relace", "usage": {}, "choices": [{"message": {"content": pixel}}]}

        members = [
            {"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500},
            {"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=respond):
            result = run_team_drawing(
                subject="apple",
                step_size=1,
                max_steps=2,
                size=16,
                members=members,
                temperature=0.2,
                max_tokens=512,
                personal_history_enabled=True,
            )
            run = Path(result["run_path"])
            member_1 = (run / "prompts" / "round_002_agent_01.txt").read_text(encoding="utf-8")
            member_2 = (run / "prompts" / "round_002_agent_02.txt").read_text(encoding="utf-8")
            self.assertIn("Round 1: (1, 1)=red", member_1)
            self.assertNotIn("(2, 1)=green", member_1)
            self.assertIn("Round 1: (2, 1)=green", member_2)
            self.assertNotIn("(1, 1)=red", member_2)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["personal_history_enabled"])

    def test_team_forum_post_is_automatically_visible_next_round(self) -> None:
        responses = [
            {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"action":"write_team_forum","pixels":[],"message":"Build from the upper left."}'}}]},
            {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"action":"color","pixels":[],"message":""}'}}]},
        ]
        member = {"model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=responses):
            result = run_team_drawing(
                subject="apple",
                step_size=1,
                max_steps=2,
                size=16,
                members=[member],
                temperature=0.2,
                max_tokens=512,
                forum_enabled=True,
            )
            run = Path(result["run_path"])
            opening_prompt = (run / "prompts" / "round_000_agent_01.txt").read_text(encoding="utf-8")
            first_parallel_prompt = (run / "prompts" / "round_001_agent_01.txt").read_text(encoding="utf-8")
            forums = json.loads((run / "forums.json").read_text(encoding="utf-8"))
            self.assertEqual((result["rounds"], result["stop_reason"]), (2, "team_complete"))
            self.assertIn("This is step 0", opening_prompt)
            self.assertIn("Build from the upper left.", first_parallel_prompt)
            self.assertEqual(len(forums["team"]), 1)
            self.assertIsNone(forums["public"])
            self.assertEqual([event["kind"] for event in forums["events"]], ["write_team_forum", "color"])
            self.assertEqual(result["forums"]["actions"], 2)

    def test_team_runner_cancels_after_current_round(self) -> None:
        response = {"model": "glm", "provider": "relace", "usage": {}, "choices": [{"message": {"content": '{"pixels":[{"x":1,"y":1,"color":"red"}]}'}}]}
        members = [{"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500}]
        seen: list[dict] = []
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", return_value=response):
            result = run_team_drawing(
                subject="apple",
                step_size=1,
                max_steps=10,
                size=16,
                members=members,
                temperature=0.2,
                max_tokens=512,
                on_step=seen.append,
                should_cancel=lambda: bool(seen),
            )
            self.assertEqual((result["rounds"], result["calls"], result["stop_reason"]), (1, 1, "cancelled"))
            self.assertTrue(result["cancelled"])
            self.assertTrue((Path(result["run_path"]) / "final_canvas.json").is_file())

    def test_team_runner_stores_and_attaches_reference(self) -> None:
        response = {"model": "glm", "provider": "relace", "usage": {}, "choices": [{"message": {"content": '{"pixels":[]}'}}]}
        members = [{"model": "glm", "provider": "relace", "reasoning": "low", "reasoning_max_tokens": 500}]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", return_value=response):
            result = run_team_drawing(
                subject="orange",
                description=DEFAULT_OBJECTS["orange"]["description"],
                reference_tag="orange",
                step_size=8,
                max_steps=2,
                size=16,
                members=members,
                temperature=0.2,
                max_tokens=512,
            )
            run = Path(result["run_path"])
            self.assertTrue((run / "reference_orange.png").is_file())
            request = json.loads(next((run / "requests").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(len(request["messages"][0]["content"]), 3)

    def test_competing_runner_keeps_team_goals_and_shared_canvas(self) -> None:
        responses = [
            {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[{"x":1,"y":1,"color":"red"}]}'}}]},
            {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[{"x":20,"y":20,"color":"orange"}]}'}}]},
            {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[]}'}}]},
            {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[]}'}}]},
        ]
        member = {"model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        teams = [
            {"subject": "apple", "description": "red fruit", "reference_tag": None, "center": (8, 8), "members": [member]},
            {"subject": "orange", "description": "orange fruit", "reference_tag": None, "center": (24, 24), "members": [member]},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=responses):
            result = run_competing_teams(
                teams=teams,
                artwork_size=16,
                step_size=1,
                max_steps=2,
                size=32,
                temperature=0.2,
                max_tokens=512,
            )
            run = Path(result["run_path"])
            self.assertEqual((result["competing_teams"], result["rounds"], result["calls"], result["filled"]), (2, 2, 4, 2))
            self.assertEqual(result["stop_reason"], "all_teams_complete")
            apple = (run / "prompts" / "round_001_team_01_member_01.txt").read_text(encoding="utf-8")
            orange = (run / "prompts" / "round_001_team_02_member_01.txt").read_text(encoding="utf-8")
            self.assertIn("Team 1", apple)
            self.assertIn("one clear apple", apple)
            self.assertIn("Team 1: apple; Team 2: orange", apple)
            self.assertIn("Team 2", orange)
            self.assertIn("one clear orange", orange)
            self.assertIn("x=0..15, y=0..15", apple)
            self.assertIn("x=16..31, y=16..31", orange)
            self.assertEqual(len(list((run / "snapshots").glob("*.png"))), 4)

    def test_competing_runner_gives_only_each_member_one_the_round_zero_turn(self) -> None:
        response = {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": '{"pixels":[]}'}}]}
        member = {"model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        teams = [
            {"subject": "apple", "members": [member, member]},
            {"subject": "fish", "members": [member, member]},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", return_value=response):
            result = run_competing_teams(
                teams=teams,
                artwork_size=16,
                step_size=1,
                max_steps=1,
                size=32,
                temperature=0.2,
                max_tokens=512,
            )
            history = json.loads((Path(result["run_path"]) / "history.json").read_text(encoding="utf-8"))
        opening = [item for item in history if item["round"] == 0]
        parallel = [item for item in history if item["round"] == 1]
        self.assertEqual(sorted((item["team"], item["member"]) for item in opening), [(1, 1), (2, 1)])
        self.assertEqual({(item["team"], item["member"]) for item in parallel}, {(1, 1), (1, 2), (2, 1), (2, 2)})
        self.assertEqual((result["rounds"], result["calls"], result["opening_turns_per_team"]), (2, 6, 1))

    def test_competing_runner_keeps_team_forums_private(self) -> None:
        def respond(request: dict, _key: str, **_kwargs) -> dict:
            prompt = request["messages"][0]["content"][0]["text"]
            message = "alpha plan" if "You are member 1 of 1 in Team 1." in prompt else "beta plan"
            if message not in prompt:
                content = json.dumps({"action": "write_team_forum", "pixels": [], "message": message})
            else:
                content = '{"action":"color","pixels":[],"message":""}'
            return {"model": "m", "provider": "p", "usage": {}, "choices": [{"message": {"content": content}}]}

        member = {"model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        teams = [
            {"subject": "apple", "description": "red fruit", "reference_tag": None, "members": [member]},
            {"subject": "orange", "description": "orange fruit", "reference_tag": None, "members": [member]},
        ]
        with tempfile.TemporaryDirectory() as directory, patch("harness.agent_runner.ROOT", Path(directory)), patch("harness.agent_runner.chat_completion", side_effect=respond):
            result = run_competing_teams(
                teams=teams,
                artwork_size=16,
                step_size=1,
                max_steps=2,
                size=32,
                temperature=0.2,
                max_tokens=512,
                forum_enabled=True,
            )
            run = Path(result["run_path"])
            team_1_prompt = (run / "prompts" / "round_001_team_01_member_01.txt").read_text(encoding="utf-8")
            team_2_prompt = (run / "prompts" / "round_001_team_02_member_01.txt").read_text(encoding="utf-8")
            self.assertIn("alpha plan", team_1_prompt)
            self.assertNotIn("beta plan", team_1_prompt)
            self.assertIn("beta plan", team_2_prompt)
            self.assertNotIn("alpha plan", team_2_prompt)
            self.assertEqual(result["forums"]["team_posts"], {"1": 1, "2": 1})

    def test_resume_reuses_saved_partial_round_responses(self) -> None:
        run_id = "drawing_20260910T205352_891075Z"
        blank = [[None for _ in range(8)] for _ in range(8)]
        member_1 = {"id": 1, "agent_id": 1, "model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        member_2 = {"id": 1, "agent_id": 2, "model": "m", "provider": "p", "reasoning": "disabled", "reasoning_max_tokens": 500}
        config = {
            "canvas_size": 8,
            "artwork_size": 8,
            "competing_teams": 2,
            "personal_history_enabled": False,
            "forum_enabled": False,
            "public_forum_enabled": False,
            "step_size": 1,
            "max_steps": 2,
            "teams": [
                {"id": 1, "subject": "apple", "description": "red", "reference_tag": None, "center": None, "assigned_bounds": [0, 7, 0, 7], "members": [member_1]},
                {"id": 2, "subject": "fish", "description": "blue", "reference_tag": None, "center": None, "assigned_bounds": [0, 7, 0, 7], "members": [member_2]},
            ],
            "parameters": {"temperature": 0.2, "max_tokens": 512},
        }
        history = [
            {
                "round": 1, "team": 1, "member": 1, "agent": 1, "completion_order": 1,
                "changed_actions": [{"x": 0, "y": 0, "color": "red"}], "pixels_changed": 1,
                "snapshot": "snapshots/frame_001.png", "forum_event": None, "usage": {"cost": 0.01},
            },
            {
                "round": 1, "team": 2, "member": 1, "agent": 2, "completion_order": 2,
                "changed_actions": [{"x": 1, "y": 0, "color": "blue"}], "pixels_changed": 1,
                "snapshot": "snapshots/frame_002.png", "forum_event": None, "usage": {"cost": 0.01},
            },
        ]
        saved_response = {
            "model": "m", "provider": "p", "usage": {"cost": 0.10},
            "choices": [{"message": {"content": '{"pixels":[{"x":2,"y":0,"color":"red"}]}'}}],
        }
        missing_response = {
            "model": "m", "provider": "p", "usage": {"cost": 0.20},
            "choices": [{"message": {"content": '{"pixels":[{"x":3,"y":0,"color":"blue"}]}'}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / run_id
            (run / "responses").mkdir(parents=True)
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "initial_canvas.json").write_text(json.dumps({"size": 8, "pixels": blank}), encoding="utf-8")
            (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
            (run / "run_state.json").write_text(json.dumps({"status": "interrupted", "last_completed_round": 1}), encoding="utf-8")
            (run / "responses" / "round_002_team_01_member_01.json").write_text(json.dumps(saved_response), encoding="utf-8")
            with patch("harness.agent_runner.ROOT", root), patch("harness.agent_runner.chat_completion", return_value=missing_response) as completion:
                info = resumable_run(run_id)
                self.assertEqual(info["last_completed_round"], 1)
                result = resume_saved_run(run_id)
            completion.assert_called_once()
            self.assertEqual((result["run_id"], result["rounds"], result["calls"]), (run_id, 2, 4))
            self.assertEqual(result["usage"]["cost"], 0.32)
            restored = json.loads((run / "final_canvas.json").read_text(encoding="utf-8"))["pixels"]
            self.assertEqual(restored[0][:4], ["red", "blue", "red", "blue"])
            final_history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            round_two = [item for item in final_history if item["round"] == 2]
            self.assertEqual([item["response_reused"] for item in round_two], [True, False])
            self.assertEqual(json.loads((run / "run_state.json").read_text(encoding="utf-8"))["status"], "complete")


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.recording = Path(self.directory.name) / "mapping.webm"
        self.server = create_server("127.0.0.1", 0, Path(self.directory.name) / "canvas.json", self.recording)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.directory.cleanup()

    def request(self, path: str, value: dict | None = None) -> tuple[int, dict]:
        data = None if value is None else json.dumps(value).encode()
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"}, method="GET" if value is None else "POST")
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_pixel_endpoint_and_state(self) -> None:
        status, event = self.request("/api/pixel", {"x": 9, "y": 2, "color": "orange", "source": "test"})
        self.assertEqual(status, 200)
        self.assertEqual(event["version"], 1)
        _, state = self.request("/api/state")
        self.assertEqual(state["pixels"][2][9], "orange")

    def test_second_backend_cannot_share_the_active_port(self) -> None:
        with self.assertRaises(OSError):
            create_server(
                "127.0.0.1",
                self.server.server_port,
                Path(self.directory.name) / "duplicate-canvas.json",
                Path(self.directory.name) / "duplicate.webm",
            )

    def test_llm_endpoint_and_invalid_action(self) -> None:
        raw = json.dumps({"actions": [{"action": "place", "x": 1, "y": 1, "color": "blue"}, {"action": "erase", "x": 2, "y": 2}]})
        status, result = self.request("/api/llm-response", {"raw_response": raw, "source": "agent-1"})
        self.assertEqual((status, len(result["events"])), (200, 2))
        status, result = self.request("/api/pixel", {"x": -1, "y": 0, "color": "red"})
        self.assertEqual(status, 400)
        self.assertIn("x must be", result["error"])

    def test_recording_endpoint(self) -> None:
        body = b"\x1aE\xdf\xa3test-webm"
        request = Request(self.url + "/api/recording", data=body, headers={"Content-Type": "video/webm"}, method="POST")
        with urlopen(request, timeout=3) as response:
            result = json.load(response)
        self.assertTrue(result["saved"])
        self.assertEqual(self.recording.read_bytes(), body)

    def test_run_video_is_saved_inside_named_run(self) -> None:
        run_id = "drawing_20260910T000000_000000Z"
        root = Path(self.directory.name) / "harness-root"
        (root / "runs" / run_id).mkdir(parents=True)
        body = b"\x1aE\xdf\xa3one-frame-webm"
        request = Request(self.url + f"/api/run-video?run_id={run_id}", data=body, headers={"Content-Type": "video/webm"}, method="POST")
        with patch("harness.server.ROOT", root), urlopen(request, timeout=3) as response:
            result = json.load(response)
        self.assertEqual(result["video_url"], f"/runs/{run_id}/video.webm")
        self.assertEqual((root / "runs" / run_id / "video.webm").read_bytes(), body)

    def test_completed_run_can_be_stored_as_custom_reference(self) -> None:
        run_id = "drawing_20260910T000000_000001Z"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "harness-root"
            run = root / "runs" / run_id
            run.mkdir(parents=True)
            rows = [[None for _ in range(16)] for _ in range(16)]
            rows[8][8] = "blue"
            (run / "config.json").write_text(json.dumps({"subject": "violet orb", "description": "A small violet orb."}), encoding="utf-8")
            (run / "final_canvas.json").write_text(json.dumps({"size": 16, "pixels": rows}), encoding="utf-8")
            (run / "result.json").write_text("{}", encoding="utf-8")
            references = root / "reference-images"
            with patch("harness.server.ROOT", root), patch("harness.references.REFERENCE_ROOT", references), patch("harness.references.REFERENCE_INDEX", references / "saved_references.json"):
                status, result = self.request("/api/store-reference", {"run_id": run_id})
                self.assertEqual(status, 200)
                self.assertTrue(result["stored"])
                self.assertEqual(result["object"]["name"], "violet orb")
                with urlopen(self.url + f"/api/reference-image?tag={result['tag']}&size=16", timeout=3) as response:
                    self.assertEqual(response.read()[:8], b"\x89PNG\r\n\x1a\n")

    def test_cancel_endpoint_handles_finished_run(self) -> None:
        status, result = self.request("/api/cancel-run", {"run_token": "finished_run_123"})
        self.assertEqual(status, 200)
        self.assertFalse(result["cancel_requested"])

    def test_resumable_run_endpoint_and_resume_dispatch(self) -> None:
        run_id = "drawing_20260910T205352_891075Z"
        info = {"available": True, "run_id": run_id, "mode": "competing", "last_completed_round": 5, "max_steps": 10, "calls": 40, "status": "interrupted"}
        checkpoint_pixels = [[None for _ in range(32)] for _ in range(32)]
        checkpoint_pixels[2][3] = "green"
        expected = {
            "run_id": run_id, "rounds": 10, "calls": 80, "filled": 1,
            "usage": {}, "stop_reason": "max_steps", "snapshot_urls": [], "output_url": "/artifacts/latest_agent_output.png",
        }
        with patch("harness.server.resumable_run", return_value=info):
            status, discovered = self.request("/api/resumable-run")
        self.assertEqual(status, 200)
        self.assertEqual(discovered["last_completed_round"], 5)
        with (
            patch("harness.server.resumable_run", return_value=info),
            patch("harness.server.resume_canvas", return_value={**info, "size": 32, "pixels": checkpoint_pixels}),
            patch("harness.server.resume_saved_run", return_value=expected) as runner,
            patch("harness.server.load_env"),
        ):
            status, result = self.request("/api/resume-run", {"run_id": run_id, "run_token": "resume_test_123"})
        self.assertEqual(status, 200)
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(runner.call_args.args[0], run_id)
        self.assertTrue(callable(runner.call_args.kwargs["should_cancel"]))
        _, state = self.request("/api/state")
        self.assertEqual(state["pixels"][2][3], "green")

    def test_active_http_run_receives_cancellation(self) -> None:
        started = threading.Event()

        def fake_run(**kwargs: object) -> dict:
            started.set()
            should_cancel = kwargs["should_cancel"]
            for _ in range(200):
                if callable(should_cancel) and should_cancel():
                    return {"run_id": "drawing_fake", "rounds": 1, "calls": 1, "filled": 0, "usage": {}, "stop_reason": "cancelled", "snapshot_urls": []}
                time.sleep(0.005)
            raise AssertionError("cancellation did not reach the active runner")

        payload = {
            "run_token": "cancel_test_123",
            "subject": "apple",
            "step_size": 8,
            "max_steps": 10,
            "members": [{"model": "glm", "provider": "relace/fp4", "reasoning": "low", "reasoning_max_tokens": 500}],
            "parameters": {"temperature": 0.2, "max_tokens": 512},
        }
        outcome: dict[str, tuple[int, dict]] = {}

        def start_run() -> None:
            outcome["run"] = self.request("/api/run-agent", payload)

        with patch("harness.server.load_env"), patch("harness.server.run_team_drawing", side_effect=fake_run):
            thread = threading.Thread(target=start_run)
            thread.start()
            self.assertTrue(started.wait(timeout=1))
            status, cancellation = self.request("/api/cancel-run", {"run_token": "cancel_test_123"})
            self.assertEqual(status, 200)
            self.assertTrue(cancellation["cancel_requested"])
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome["run"][1]["stop_reason"], "cancelled")

    def test_http_route_dispatches_multiple_team_configuration(self) -> None:
        member = {"model": "glm", "provider": "relace/fp4", "reasoning": "low", "reasoning_max_tokens": 500}
        payload = {
            "run_token": "multi_team_test_123",
            "artwork_size": 16,
            "step_size": 8,
            "max_steps": 10,
            "forum_enabled": True,
            "public_forum_enabled": True,
            "prompt_template_id": "custom_test",
            "prompt_template": "Draw {{subject}} for Team {{team_index}}.",
            "teams": [
                {"subject": "apple", "description": "red", "reference_tag": None, "center": {"x": 4, "y": 4}, "members": [member]},
                {"subject": "orange", "description": "orange", "reference_tag": None, "center": {"x": 12, "y": 12}, "members": [member]},
            ],
            "parameters": {"temperature": 0.2, "max_tokens": 512},
        }
        expected = {"run_id": "drawing_fake", "rounds": 1, "calls": 2, "filled": 2, "usage": {}, "stop_reason": "single_round", "snapshot_urls": []}
        with patch("harness.server.load_env"), patch("harness.server.run_competing_teams", return_value=expected) as runner:
            status, result = self.request("/api/run-agent", payload)
        self.assertEqual(status, 200)
        self.assertEqual(result["run_id"], "drawing_fake")
        self.assertEqual(len(runner.call_args.kwargs["teams"]), 2)
        self.assertEqual(runner.call_args.kwargs["artwork_size"], 16)
        self.assertTrue(runner.call_args.kwargs["forum_enabled"])
        self.assertTrue(runner.call_args.kwargs["public_forum_enabled"])
        self.assertEqual(runner.call_args.kwargs["prompt_template_id"], "custom_test")
        self.assertEqual(runner.call_args.kwargs["prompt_template_text"], "Draw {{subject}} for Team {{team_index}}.")
        self.assertEqual(runner.call_args.kwargs["teams"][0]["center"], (4, 4))
        self.assertEqual(runner.call_args.kwargs["teams"][1]["center"], (12, 12))

    def test_agent_defaults_do_not_expose_key(self) -> None:
        status, result = self.request("/api/agent-defaults")
        self.assertEqual(status, 200)
        self.assertEqual(result["team_size"], 2)
        self.assertEqual(result["competing_teams"], 1)
        self.assertEqual(result["artwork_size"], 16)
        self.assertEqual(set(result["objects"]), {"apple", "orange", "lemon", "eggplant", "tree", "fish", "arrow", "balloon"})
        self.assertEqual(result["description"], DEFAULT_OBJECTS["apple"]["description"])
        self.assertFalse(result["personal_history_enabled"])
        self.assertFalse(result["forum_enabled"])
        self.assertFalse(result["public_forum_enabled"])
        self.assertEqual(len(result["members"]), 2)
        self.assertTrue(all(member["model"] == "z-ai/glm-5.3-flash" for member in result["members"]))
        self.assertTrue(all(member["provider"] == "relace/fp4" for member in result["members"]))
        self.assertEqual(result["step_size"], 8)
        self.assertEqual(result["max_steps"], 10)
        self.assertEqual(result["parameters"]["max_tokens"], 4096)
        self.assertEqual(result["prompt_template_id"], DEFAULT_PROMPT_TEMPLATE_ID)
        self.assertNotIn("api_key", json.dumps(result).lower())

        status, templates = self.request("/api/prompt-templates")
        self.assertEqual(status, 200)
        self.assertEqual(templates["default_id"], DEFAULT_PROMPT_TEMPLATE_ID)
        self.assertEqual({item["id"] for item in templates["templates"]}, {"forum_optional_v1", "forum_suggestions_v2a", "forum_protocol_v3a", "forum_directed_v2", "forum_conflict_v3"})
        self.assertTrue(all("{{subject}}" in item["template"] for item in templates["templates"]))

    def test_prompt_preview_switches_modes(self) -> None:
        status, iterative = self.request("/api/prompt-preview", {"subject": "apple", "step_size": 8, "team_size": 2})
        self.assertEqual(status, 200)
        self.assertIn("Change at most 8 pixels", iterative["prompt"])
        self.assertIn("There are 2 members in your team", iterative["prompt"])
        status, baseline = self.request("/api/prompt-preview", {"subject": "apple", "step_size": 0, "team_size": 1})
        self.assertEqual(status, 200)
        self.assertNotIn("last saved step", baseline["prompt"])
        self.assertNotIn("team", baseline["prompt"].casefold())
        status, remembered = self.request("/api/prompt-preview", {"subject": "apple", "step_size": 8, "team_size": 2, "personal_history_enabled": True})
        self.assertEqual(status, 200)
        self.assertIn("Personal edit history:", remembered["prompt"])
        status, competing = self.request("/api/prompt-preview", {"subject": "orange", "step_size": 8, "team_size": 2, "competing_teams": 3, "team_index": 2, "team_objectives": ["apple", "orange", "pear"], "artwork_size": 16})
        self.assertEqual(status, 200)
        self.assertIn("Team 2", competing["prompt"])
        self.assertIn("There are 3 teams", competing["prompt"])
        self.assertIn("Team 1: apple; Team 2: orange; Team 3: pear", competing["prompt"])
        self.assertIn("no larger than 16x16", competing["prompt"])
        status, centered = self.request("/api/prompt-preview", {"subject": "orange", "step_size": 8, "team_size": 2, "competing_teams": 2, "team_index": 2, "artwork_size": 16, "center": {"x": 8, "y": 8}})
        self.assertEqual(status, 200)
        self.assertIn("assigned center point is (8, 8)", centered["prompt"])
        self.assertIn("x=0..15, y=0..15", centered["prompt"])
        status, forum = self.request("/api/prompt-preview", {
            "subject": "apple",
            "step_size": 8,
            "team_size": 2,
            "forum_enabled": True,
            "public_forum_enabled": True,
        })
        self.assertEqual(status, 200)
        self.assertIn("private team forum", forum["prompt"])
        self.assertIn("public forum also exists", forum["prompt"])
        self.assertIn("write_public_forum", forum["prompt"])
        status, custom = self.request("/api/prompt-preview", {
            "subject": "orange", "step_size": 8, "team_size": 2, "competing_teams": 3, "team_index": 2,
            "team_objectives": ["apple", "orange", "fish"],
            "prompt_template_id": "custom_test",
            "prompt_template": "Team {{team_index}}/{{competing_teams}} makes {{subject}} in {{progress_label}} {{step_number}}.",
        })
        self.assertEqual(status, 200)
        self.assertEqual(custom["prompt"], "Team 2/3 makes orange in round 1.")

    def test_resize_and_reference_endpoints(self) -> None:
        status, result = self.request("/api/resize", {"size": 24})
        self.assertEqual(status, 200)
        self.assertEqual((result["state"]["width"], result["state"]["height"]), (24, 24))
        for tag in ("orange", "tree", "fish"):
            with urlopen(self.url + f"/api/reference-image?tag={tag}&size=24", timeout=3) as response:
                image = response.read()
            self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        status, preview = self.request("/api/prompt-preview", {
            "subject": "orange",
            "description": DEFAULT_OBJECTS["orange"]["description"],
            "reference_tag": "orange",
            "step_size": 8,
            "team_size": 2,
        })
        self.assertEqual(status, 200)
        self.assertIn("completed orange reference", preview["prompt"])


if __name__ == "__main__":
    unittest.main()
