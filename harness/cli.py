"""Command-line client for exercising the live canvas."""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .store import COLORS, HEIGHT, WIDTH


def mapping_actions(width: int = WIDTH, height: int = HEIGHT):
    """Yield every coordinate explicitly in left-to-right, top-to-bottom order."""
    names = tuple(COLORS)
    band_height = max(1, height // len(names))
    for y in range(height):
        color = names[min(y // band_height, len(names) - 1)]
        for x in range(width):
            yield {"action": "place", "x": x, "y": y, "color": color, "source": "mapping-test"}


def get_state(base_url: str) -> dict[str, Any]:
    try:
        with urlopen(base_url.rstrip("/") + "/api/state", timeout=5) as response:
            return json.load(response)
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Cannot read canvas state at {base_url}: {exc}") from exc


def post(base_url: str, path: str, value: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(value).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Server rejected the command ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"Cannot reach canvas server at {base_url}: {exc.reason}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Send actions to the live collective canvas")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    commands = parser.add_subparsers(dest="command", required=True)

    place = commands.add_parser("place")
    place.add_argument("x", type=int)
    place.add_argument("y", type=int)
    place.add_argument("color", choices=COLORS)

    erase = commands.add_parser("erase")
    erase.add_argument("x", type=int)
    erase.add_argument("y", type=int)

    random_command = commands.add_parser("random", help="place random dots one at a time")
    random_command.add_argument("--count", type=int, default=80)
    random_command.add_argument("--delay", type=float, default=0.04)
    random_command.add_argument("--seed", type=int)

    mapping = commands.add_parser("mapping", help="fill all coordinates row by row to verify x/y mapping")
    mapping.add_argument("--delay", type=float, default=0.015)

    commands.add_parser("clear")
    args = parser.parse_args()

    if args.command == "place":
        result = post(args.url, "/api/pixel", {"action": "place", "x": args.x, "y": args.y, "color": args.color, "source": "cli"})
        print(json.dumps(result))
    elif args.command == "erase":
        result = post(args.url, "/api/pixel", {"action": "erase", "x": args.x, "y": args.y, "source": "cli"})
        print(json.dumps(result))
    elif args.command == "clear":
        print(json.dumps(post(args.url, "/api/clear", {"source": "cli"})))
    elif args.command == "random":
        if args.count < 1:
            raise SystemExit("--count must be positive")
        if args.delay < 0:
            raise SystemExit("--delay cannot be negative")
        generator = random.Random(args.seed)
        names = tuple(COLORS)
        state = get_state(args.url)
        width, height = state["width"], state["height"]
        for index in range(args.count):
            action = {
                "action": "place",
                "x": generator.randrange(width),
                "y": generator.randrange(height),
                "color": generator.choice(names),
                "source": "cli-random",
            }
            post(args.url, "/api/pixel", action)
            print(f"\rPlaced {index + 1}/{args.count}", end="", flush=True)
            if args.delay:
                time.sleep(args.delay)
        print()
    else:
        if args.delay < 0:
            raise SystemExit("--delay cannot be negative")
        state = get_state(args.url)
        width, height = state["width"], state["height"]
        post(args.url, "/api/clear", {"source": "mapping-test"})
        for index, action in enumerate(mapping_actions(width, height), start=1):
            post(args.url, "/api/pixel", action)
            if index % width == 0:
                print(f"\rCompleted row {index // width}/{height}", end="", flush=True)
            if args.delay:
                time.sleep(args.delay)
        print()


if __name__ == "__main__":
    main()
