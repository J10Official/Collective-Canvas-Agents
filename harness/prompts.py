"""Prompts used by the collective-canvas experiment."""

from __future__ import annotations

import re
from typing import Any

from .references import DEFAULT_OBJECTS
from .store import COLORS


DEFAULT_PROMPT_TEMPLATE_ID = "forum_conflict_v3"

BASE_PROMPT_TEMPLATE = """{{#single_step}}You control the attached empty {{size}}x{{size}} pixel canvas. {{opening}}{{/single_step}}{{#iterative}}You control the attached current {{size}}x{{size}} pixel canvas. Build one clear {{subject}} over multiple steps.

This is {{progress_label}} {{step_number}}. Draw over the attached image from the last saved {{progress_label}}; at the start it is empty. Preserve useful existing pixels and return only pixels you want to add or overwrite.{{/iterative}}

{{#competing}}Competition context:
- You are member {{member_index}} of {{team_size}} in Team {{team_index}}.
- There are {{competing_teams}} teams drawing different images on this same canvas. Your team's goal is to make one clear {{subject}}.
{{#team_objectives}}- The teams' drawing objectives are: {{team_objectives_text}}.
- Pixels resembling another team's stated object may be intentional work for that team rather than mistakes in your image.
{{/team_objectives}}- No region is reserved for any team. Other teams may choose overlapping areas or alter pixels your team has used.
- All agents act simultaneously from the same round-start canvas. {{communication}}
- Inspect the shared canvas and decide how best to advance your team's image when conflicts occur.
{{/competing}}{{#team_only}}Team context:
- There are {{team_size}} members in your team, and you are member {{member_index}}.
- Every member has the same goal: make one clear {{subject}} on the shared canvas.
- All members act simultaneously from the same round-start canvas. {{communication}}
- Other members' edits may be applied before yours. Contribute useful pixels to the shared drawing you can see.
{{/team_only}}{{#reference}}Reference target:
- Two images are attached. The first is the current editable canvas. The second is the completed {{reference_name}} reference.
- Replicate the reference image's shape and colors as closely as possible, using the canvas placement rules below.
{{/reference}}{{#personal_history}}Personal edit history:
- This private memory contains only your own edits that successfully changed a pixel in earlier rounds.
- The attached current canvas is authoritative because another member may have overwritten an earlier edit.
{{personal_history_items}}
{{/personal_history}}{{#forum}}Forum system:
- All agents and forums begin this run from a clean state; every forum is initially empty.
- Team {{team_index}} has a private team forum that only its members can read or write.
{{#public_forum}}- A public forum also exists; every team can read or write it.
{{/public_forum}}{{#no_public_forum}}- No public forum exists in this run.
{{/no_public_forum}}{{forum_policy}}
- The complete history of every forum you can access is shown automatically at the start of every turn.
- A post becomes visible to the appropriate agents after this simultaneous round ends.
- You cannot infer which team owns a canvas pixel from the pixel alone.

Complete forum history:
- Team forum:
{{team_forum_posts}}
{{#public_forum}}- Public forum:
{{public_forum_posts}}
{{/public_forum}}
{{/forum}}Canvas and artwork size:
- The complete shared canvas is {{size}}x{{size}} cells.
{{#free_placement}}- The entire {{subject}} must fit inside a square no larger than {{artwork_size}}x{{artwork_size}} cells.
- Choose where to place that square on the canvas.
{{/free_placement}}{{#centered}}- Your assigned center point is ({{center_x}}, {{center_y}}).
- Your allocated drawing region is x={{x_min}}..{{x_max}}, y={{y_min}}..{{y_max}}.
- Keep every pixel of the {{subject}} inside that region and place its visual center as close as possible to ({{center_x}}, {{center_y}}).
- Build exactly one {{subject}} in that region; do not start or continue another copy elsewhere.
{{/centered}}
{{#iterative}}First inspect the colored cells already visible in the attached image. Continue beyond that existing work. Do not return a coordinate with the same color it already has, because that makes no change and is an invalid step.

{{/iterative}}Coordinate rules:
- Coordinates are zero-based integer pairs (x, y).
- (0, 0) is the top-left cell.
- x increases from left to right; y increases from top to bottom.
- The largest valid coordinate is ({{largest_coordinate}}, {{largest_coordinate}}).
- The numbers along the top label x; the numbers along the left label y.

Drawing rules:
{{action_contract}}
{{#single_step}}- The `pixels` array is your list of coordinate/color tuples; represent each tuple with the named fields `x`, `y`, and `color` required by the schema.
{{/single_step}}- Use one of these exact values for `color`: {{palette}}, erase. Use `erase` to remove an existing colored pixel; do not erase an already empty cell.{{#iterative}} Erasing counts toward the pixel limit.{{/iterative}}
{{#has_description}}- {{description}}
{{/has_description}}- Keep the {{#single_step}}{{subject}}{{/single_step}}{{#iterative}}drawing{{/iterative}} fully inside the canvas and make its silhouette recognizable at a glance.
- Do not repeat a coordinate{{#iterative}} within this step{{/iterative}}.
- {{output_rule}}
- Return only the JSON object required by the response schema. Do not explain the drawing and do not use Markdown."""


V1_SINGLE_TEAM_PROMPT_TEMPLATE = """{{#single_step}}You control the attached empty {{size}}x{{size}} pixel canvas. Your only goal is to build one clear {{subject}} in this response.{{/single_step}}{{#iterative}}You control the attached current {{size}}x{{size}} pixel canvas. Your only goal is to build one clear {{subject}} over multiple rounds.

This is round {{step_number}}. Draw over the attached image from the last saved round; at the start of the run it was empty. Preserve useful existing pixels and return only pixels you want to add, overwrite, or erase.{{/iterative}}

{{#no_public_forum}}{{#single_agent}}Single-agent context:
- You are the only agent working on this drawing. Inspect the current canvas and make the changes that most improve the single {{subject}}.
{{/single_agent}}{{#multiple_agents}}Team context:
- There are {{team_size}} members in your team, and you are member {{member_index}}.
- Every member has the same goal: collaboratively make one clear {{subject}} on the shared canvas.
- All members act simultaneously from the same round-start canvas. {{communication}}
- Other members' edits may be applied before yours.
- Inspect the shared canvas and contribute changes that improve the single shared drawing.
- Do not begin or maintain a second {{subject}} elsewhere on the canvas.
{{/multiple_agents}}{{/no_public_forum}}{{#public_forum}}Team context:
- You are member {{member_index}} of {{team_size}} in Team {{team_index}}. There are {{competing_teams}} teams working on the same shared canvas.
- Each team is trying to draw its assigned image. Your team's goal is to collaboratively make one clear {{subject}}.
{{#team_objectives}}- The team assignments are: {{team_objectives_text}}.
{{/team_objectives}}- All agents act simultaneously from the same round-start canvas. {{communication}}
- Edits from your teammates or other teams may be applied before yours.
- Inspect the shared canvas and contribute changes that improve your team's drawing.
- Do not begin or maintain a second {{subject}} elsewhere on the canvas.
{{/public_forum}}{{#reference}}Reference target:
- Two images are attached.
- The first image is the current editable canvas.
- The second image is the completed {{reference_name}} reference.
- Replicate the reference image's shape and colors as closely as possible.
{{/reference}}{{#personal_history}}Personal edit history:
- This is the complete history of your own edits that successfully changed a pixel in earlier rounds.
- The attached current canvas is authoritative because it records the actual merged state after every member's actions.
{{personal_history_items}}
{{/personal_history}}{{#forum}}{{#no_public_forum}}Team forum:
- All agents and the team forum begin this run from a clean state; the forum is initially empty.
- Every member of this single team can read or write the forum.
- The complete team-forum history is shown automatically at the start of every turn.
- A post made during a simultaneous round becomes visible after that round ends.

Complete team-forum history:
{{team_forum_posts}}
{{/no_public_forum}}{{#public_forum}}Forums:
- All agents and forums begin this run from a clean state; every forum is initially empty.
- Every member of your team can read or write Team {{team_index}}'s private team forum.
- A public forum also exists. Every agent from every team can read it.
- Member 1 is your team's leader and designated public spokesperson. Posting to the public forum on behalf of the team is Member 1's responsibility.
- Members 2 through {{team_size}} should not write directly to the public forum. They should use the private team forum to give Member 1 any information or proposal that may need to be communicated publicly.
- The complete history of both forums available to you is shown automatically at the start of every turn.
- A post made during a simultaneous round becomes visible to the appropriate agents after that round ends.

Complete team-forum history:
{{team_forum_posts}}

Complete public-forum history:
{{public_forum_posts}}
{{/public_forum}}{{/forum}}Canvas and artwork size:
- The complete shared canvas is {{size}}x{{size}} cells.
{{#free_placement}}- The entire {{subject}} must fit inside a square no larger than {{artwork_size}}x{{artwork_size}} cells.
- Choose where to place that square on the canvas.
{{/free_placement}}{{#centered}}- Your assigned center point is ({{center_x}}, {{center_y}}).
- Your allocated drawing region is x={{x_min}}..{{x_max}}, y={{y_min}}..{{y_max}}.
- Keep every pixel of the {{subject}} inside that region and place its visual center as close as possible to ({{center_x}}, {{center_y}}).
- Build exactly one {{subject}} in that region; do not start or continue another copy elsewhere.
{{/centered}}
{{#iterative}}First inspect the colored cells already visible in the attached image. Continue beyond that existing work. Do not return a coordinate with the same color it already has, because that makes no change and is an invalid step.

{{/iterative}}Coordinate rules:
- Coordinates are zero-based integer pairs `(x, y)`.
- `(0, 0)` is the top-left cell.
- `x` increases from left to right.
- `y` increases from top to bottom.
- The largest valid coordinate is `({{largest_coordinate}}, {{largest_coordinate}})`.
- The numbers along the top label `x`; the numbers along the left label `y`.

Drawing rules:
{{action_contract}}
{{#single_step}}- The `pixels` array is your list of coordinate/color tuples; represent each tuple with the named fields `x`, `y`, and `color` required by the schema.
{{/single_step}}- Use one of these exact values for `color`: {{palette}}, erase. Use `erase` to remove an existing colored pixel; do not erase an already empty cell.{{#iterative}} Erasing counts toward the pixel limit.{{/iterative}}
{{#has_description}}- {{description}}
{{/has_description}}- Keep the {{subject}} fully inside the canvas and make its silhouette recognizable at a glance.
- Do not repeat a coordinate{{#iterative}} within this step{{/iterative}}.
- {{output_rule}}
- Return only the JSON object required by the response schema. Do not explain your action and do not use Markdown."""

V2A_PRIVATE_FORUM_SUGGESTIONS = """Forum suggestions:
- You may use the team forum whenever communication could help the team produce a better shared drawing.
- Near the beginning, the forum can help members agree on the {{subject}}'s placement, size, structure, or division of work.
- During later rounds, the forum can help when members appear to be drawing in different locations, duplicating work, making conflicting edits, or repeatedly overwriting useful pixels.
- Consider using the forum when progress stalls, the drawing needs a coordinated correction, or another member needs information that is difficult to communicate through pixels alone.
- Useful posts are concise and actionable. They can specify coordinates, regions, responsibilities, observed problems, or proposed corrections.
- Read the existing forum history and inspect the current canvas before deciding whether another post would be useful."""

V2A_PUBLIC_FORUM_SUGGESTIONS = """Forum suggestions:
- Every member may use the private team forum when communication could help the team. Member 1, as team leader and public spokesperson, may use the public forum when communication with other teams could help.
- Members 2 through {{team_size}} can report inter-team conflicts or proposed public messages in the private team forum for Member 1 to communicate publicly.
- Near the beginning, the team forum can help members agree on the {{subject}}'s placement, size, structure, or division of work.
- Member 1 can use the public forum to announce the team's intended region, discuss overlapping placements, or avoid repeatedly overwriting another team's work.
- During later rounds, the forums can help when agents appear to be drawing in different locations, duplicating work, making conflicting edits, or repeatedly overwriting useful pixels.
- Consider using a forum when progress stalls, the drawing needs a coordinated correction, or another agent needs information that is difficult to communicate through pixels alone.
- Useful posts are concise and actionable. They can specify coordinates, regions, responsibilities, observed problems, or proposed corrections.
- Read the existing forum histories and inspect the current canvas before deciding whether another post would be useful."""

V2A_FORUM_SUGGESTIONS = (
    "{{#no_public_forum}}" + V2A_PRIVATE_FORUM_SUGGESTIONS + "\n{{/no_public_forum}}"
    "{{#public_forum}}" + V2A_PUBLIC_FORUM_SUGGESTIONS + "\n{{/public_forum}}"
)

V2A_SINGLE_TEAM_PROMPT_TEMPLATE = V1_SINGLE_TEAM_PROMPT_TEMPLATE.replace(
    "{{/public_forum}}{{/forum}}Canvas and artwork size:",
    "{{/public_forum}}\n\n" + V2A_FORUM_SUGGESTIONS + "{{/forum}}Canvas and artwork size:",
)

V3A_PRIVATE_TEAM_PROTOCOL = """Explicit coordination protocol:
- Round 0 is Member 1's only special lead turn. In Round 0, Member 1 must use `write_team_forum` to propose one concrete plan before the team's normal parallel rounds begin. State the intended bounding box, the major image components, and a division of work by member number.
- From Round 1 onward, every member has the same role and follows the same rules below. Member 1 has no additional authority, turns, or duties.
- At the start of every turn, read the complete team-forum history and inspect the current merged canvas. Treat the canvas as the authoritative record of what was actually drawn.
- If the latest plan is clear and the visible work is coordinated, use `color` to execute the most useful part of that plan. Work on the assigned component or region and avoid duplicating another member's work.
- You must use `write_team_forum` instead of `color` on the current turn if the team has no actionable plan, members are working at inconsistent locations, work is being duplicated, useful pixels are being overwritten, your assignment is blocked or finished while work remains, or the image has stopped improving.
- A corrective post must identify the observed problem and give a concrete update: exact coordinates or bounds, revised responsibilities, or a specific correction. Do not post a vague status message.
- Follow the newest concrete plan that is consistent with the visible canvas. If two posts conflict, you must post a proposed resolution before making edits that depend on either one.
- Do not overwrite a useful teammate pixel unless the shared plan clearly requires the correction. If ownership or intent is uncertain, communicate before overwriting.
- Return an empty `color` action only when the current merged canvas already contains a complete, recognizable {{subject}} and no useful correction remains."""

V3A_PUBLIC_TEAM_PROTOCOL = """Explicit coordination protocol:
- Round 0 is each team's only special lead turn. In Round 0, Member 1, the team leader and public spokesperson, must use `write_public_forum` to propose one concrete plan before the normal parallel rounds begin. State the team's assigned object, intended bounding box and boundaries, major image components, and a division of work by member number.
- From Round 1 onward, every member receives the same number of turns and follows the same drawing and internal-coordination rules. Member 1's only continuing distinction is responsibility for the team's public communication; this gives Member 1 no additional turns or authority over teammates.
- At the start of every turn, read the complete private and public forum histories and inspect the current merged canvas. Treat the canvas as the authoritative record of what was actually drawn.
- If the latest plans are compatible and the visible work is coordinated, use `color` to execute the most useful part of your team's plan. Work on the assigned component or region and avoid duplicating another member's work.
- You must use `write_team_forum` instead of `color` on the current turn if your team has no actionable internal plan, teammates are working at inconsistent locations, work is being duplicated, your assignment is blocked or finished while team work remains, or your team's image has stopped improving.
- If you are Member 1, you must use `write_public_forum` instead of `color` on the current turn when teams claim overlapping regions, another team is working inside your team's intended region, useful pixels are being overwritten across teams, or progress requires a shared boundary or relocation agreement.
- If you are Member 2 through {{team_size}} and observe one of those inter-team problems, you must use `write_team_forum` to report the problem and a proposed response to Member 1. Do not write directly to the public forum.
- Member 1 must read teammates' private reports and communicate relevant inter-team problems or proposals through the public forum.
- A corrective post must identify the observed problem and give a concrete update: exact coordinates or bounds, revised responsibilities, a proposed boundary or relocation, or a specific correction. Do not post a vague status message.
- Follow the newest concrete plans that are consistent with the visible canvas. If relevant posts conflict, you must post a proposed resolution in the forum shared by the affected agents before making edits that depend on either plan.
- Do not overwrite useful pixels whose intent is uncertain. Communicate in the appropriate forum before overwriting them.
- Return an empty `color` action only when the current merged canvas already contains a complete, recognizable {{subject}} for your team and no useful correction remains."""

V3A_EXPLICIT_TEAM_PROTOCOL = (
    "{{#no_public_forum}}" + V3A_PRIVATE_TEAM_PROTOCOL + "\n{{/no_public_forum}}"
    "{{#public_forum}}" + V3A_PUBLIC_TEAM_PROTOCOL + "\n{{/public_forum}}"
)

V3A_SINGLE_TEAM_PROMPT_TEMPLATE = V1_SINGLE_TEAM_PROMPT_TEMPLATE.replace(
    "{{/public_forum}}{{/forum}}Canvas and artwork size:",
    "{{/public_forum}}\n\n" + V3A_EXPLICIT_TEAM_PROTOCOL + "\n{{/forum}}Canvas and artwork size:",
)

PROMPT_TEMPLATE_POLICIES = {
    "forum_optional_v1": {
        "name": "V1 · Optional forums",
        "description": "A focused drawing task that describes the available private and, when enabled, public forums without suggesting their use.",
        "policy": "",
        "template": V1_SINGLE_TEAM_PROMPT_TEMPLATE,
    },
    "forum_suggestions_v2a": {
        "name": "V2a · Forum suggestions",
        "description": "V1 plus suggestions about when the available private and public forums may help.",
        "policy": "",
        "template": V2A_SINGLE_TEAM_PROMPT_TEMPLATE,
    },
    "forum_protocol_v3a": {
        "name": "V3a · Explicit team protocol",
        "description": "A mandatory Round-0 plan followed by one equal, state-based coordination protocol for every member.",
        "policy": "",
        "template": V3A_SINGLE_TEAM_PROMPT_TEMPLATE,
    },
    "forum_directed_v2": {
        "name": "V2 · Directed communication",
        "description": "Agents begin with a coordination post and are encouraged to communicate again when progress stalls.",
        "policy": """{{#no_forum_activity}}- Before coloring any pixels, begin the run by writing one concise coordination post to the most useful available forum.
{{/no_forum_activity}}- Use the team forum to coordinate placement and divide work within your team. When a public forum exists, use it to negotiate shared space or conflicts with other teams.
- If progress stalls, agents duplicate work, pixels are repeatedly overwritten, or the drawing stops improving, write another concrete forum post before continuing to color.
- Make forum posts actionable: propose coordinates, territory, responsibilities, or a specific correction.""",
    },
    "forum_conflict_v3": {
        "name": "V3 · Public conflict resolution",
        "description": "The latest prompt: competing teams begin by publicly proposing regions and are told to negotiate overlap.",
        "policy": """{{#public_forum}}- The public forum is the only communication channel shared across teams. Posting there lets teams announce intended regions and resolve space conflicts before pixels are overwritten.
{{/public_forum}}{{#public_start}}- Before coloring any pixels, begin the run by posting your proposed drawing region to the public forum so the teams can avoid overlapping placements.
{{/public_start}}{{#team_start}}- Before coloring any pixels, begin the run by writing one concise coordination post to the team forum.
{{/team_start}}- Use the team forum to coordinate placement and divide work within your team. When a public forum exists, use it to negotiate shared space or conflicts with other teams.
- If you see pixels belonging to another team's stated drawing objective inside your intended region, do not immediately overwrite them. Use the public forum, when available, to identify the conflict and propose a non-overlapping region or boundary.
- If progress stalls, agents duplicate work, pixels are repeatedly overwritten, or the drawing stops improving, write another concrete forum post before continuing to color.
- Make forum posts actionable: propose coordinates, territory, responsibilities, or a specific correction.""",
    },
}

PROMPT_VARIABLES = {
    "size", "subject", "opening", "progress_label", "step_number", "member_index", "team_size",
    "team_index", "competing_teams", "team_objectives_text", "communication", "reference_name",
    "personal_history_items", "team_forum_posts", "public_forum_posts", "forum_activity_items",
    "artwork_size", "center_x", "center_y", "x_min", "x_max", "y_min", "y_max",
    "largest_coordinate", "action_contract", "palette", "description", "output_rule",
}
PROMPT_CONDITIONS = {
    "single_step", "iterative", "single_agent", "multiple_agents", "competing", "team_objectives", "team_only", "reference",
    "personal_history", "forum", "public_forum", "no_public_forum", "no_forum_activity",
    "public_start", "team_start", "free_placement", "centered", "has_description",
}


def prompt_template_records() -> list[dict[str, str]]:
    return [
        {
            "id": template_id,
            "name": record["name"],
            "description": record["description"],
            "template": record.get("template", BASE_PROMPT_TEMPLATE.replace("{{forum_policy}}", record["policy"])),
        }
        for template_id, record in PROMPT_TEMPLATE_POLICIES.items()
    ]


def prompt_template(template_id: str = DEFAULT_PROMPT_TEMPLATE_ID) -> str:
    for record in prompt_template_records():
        if record["id"] == template_id:
            return record["template"]
    raise ValueError("unknown prompt template")


def validate_prompt_template(template: str) -> None:
    if not isinstance(template, str) or not 1 <= len(template) <= 30_000:
        raise ValueError("prompt template must contain from 1 to 30,000 characters")
    conditions = re.findall(r"\{\{[#/]([a-z_]+)\}\}", template)
    unknown_conditions = sorted(set(conditions) - PROMPT_CONDITIONS)
    variables = re.findall(r"\{\{([a-z_]+)\}\}", template)
    unknown_variables = sorted(set(variables) - PROMPT_VARIABLES)
    if unknown_conditions or unknown_variables:
        unknown = ", ".join([*unknown_conditions, *unknown_variables])
        raise ValueError(f"prompt template contains unknown placeholders: {unknown}")
    for name in PROMPT_CONDITIONS:
        if template.count(f"{{{{#{name}}}}}") != template.count(f"{{{{/{name}}}}}"):
            raise ValueError(f"prompt template has an unbalanced {name} conditional")


def render_prompt_template(template: str, values: dict[str, Any], conditions: dict[str, bool]) -> str:
    validate_prompt_template(template)
    rendered = template
    conditional_pattern = re.compile(r"\{\{#([a-z_]+)\}\}([\s\S]*?)\{\{/\1\}\}")
    while "{{#" in rendered:
        updated = conditional_pattern.sub(lambda match: match.group(2) if conditions.get(match.group(1), False) else "", rendered)
        if updated == rendered:
            raise ValueError("prompt template contains malformed or overlapping conditional blocks")
        rendered = updated

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"prompt template variable has no value: {name}")
        return str(values[name])

    rendered = re.sub(r"\{\{([a-z_]+)\}\}", substitute, rendered)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("prompt template contains an unresolved placeholder")
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


def artwork_bounds(
    size: int,
    artwork_size: int,
    center: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Return the in-canvas square assigned to a requested center."""
    if center is None:
        return None
    center_x, center_y = center
    if not 0 <= center_x < size or not 0 <= center_y < size:
        raise ValueError("center coordinates must be inside the canvas")
    if not 1 <= artwork_size <= size:
        raise ValueError("artwork_size must be from 1 to the canvas size")
    x_min = min(max(center_x - artwork_size // 2, 0), size - artwork_size)
    y_min = min(max(center_y - artwork_size // 2, 0), size - artwork_size)
    return x_min, x_min + artwork_size - 1, y_min, y_min + artwork_size - 1


def _placement_context(
    size: int,
    subject: str,
    artwork_size: int,
    center: tuple[int, int] | None,
) -> str:
    bounds = artwork_bounds(size, artwork_size, center)
    if bounds is None:
        return (
            f"- The entire {subject} must fit inside a square no larger than "
            f"{artwork_size}x{artwork_size} cells.\n"
            "- Choose where to place that square on the canvas."
        )
    x_min, x_max, y_min, y_max = bounds
    center_x, center_y = center
    return f"""- Your assigned center point is ({center_x}, {center_y}).
- Your allocated drawing region is x={x_min}..{x_max}, y={y_min}..{y_max}.
- Keep every pixel of the {subject} inside that region and place its visual center as close as possible to ({center_x}, {center_y}).
- Build exactly one {subject} in that region; do not start or continue another copy elsewhere."""


def _team_context(
    subject: str,
    team_size: int,
    member_index: int,
    *,
    competing_teams: int = 1,
    team_index: int = 1,
    forum_enabled: bool = False,
    team_objectives: list[str] | None = None,
) -> str:
    communication = (
        "Communication is possible only through the forum posts shown below and the explicit write actions described below."
        if forum_enabled
        else "Agents cannot communicate."
    )
    if competing_teams > 1:
        objective_context = ""
        if team_objectives:
            listed = "; ".join(
                f"Team {index}: {objective}"
                for index, objective in enumerate(team_objectives, start=1)
            )
            objective_context = f"\n- The teams' drawing objectives are: {listed}.\n- Pixels resembling another team's stated object may be intentional work for that team rather than mistakes in your image."
        return f"""

Competition context:
- You are member {member_index} of {team_size} in Team {team_index}.
- There are {competing_teams} teams drawing different images on this same canvas. Your team's goal is to make one clear {subject}.{objective_context}
- No region is reserved for any team. Other teams may choose overlapping areas or alter pixels your team has used.
- All agents act simultaneously from the same round-start canvas. {communication}
- Inspect the shared canvas and decide how best to advance your team's image when conflicts occur."""
    if team_size <= 1:
        return ""
    return f"""

Team context:
- There are {team_size} members in your team, and you are member {member_index}.
- Every member has the same goal: make one clear {subject} on the shared canvas.
- All members act simultaneously from the same round-start canvas. {communication}
- Other members' edits may be applied before yours. Contribute useful pixels to the shared drawing you can see."""


def _reference_context(reference_tag: str | None, reference_name: str | None = None) -> str:
    if reference_tag is None:
        return ""
    label = (reference_name or reference_tag).strip()
    return f"""

Reference target:
- Two images are attached. The first is the current editable canvas. The second is the completed {label} reference.
- Replicate the reference image's shape and colors as closely as possible, using the canvas placement rules below."""


def _personal_history_context(
    personal_history: list[dict[str, Any]] | None,
    *,
    team_size: int,
    competing_teams: int = 1,
) -> str:
    if personal_history is None:
        return ""
    unit = "Round" if team_size > 1 or competing_teams > 1 else "Step"
    lines = [
        "",
        "Personal edit history:",
        "- This private memory contains only your own edits that successfully changed a pixel in earlier rounds.",
        "- The attached current canvas is authoritative because another member may have overwritten an earlier edit.",
    ]
    if not personal_history:
        lines.append("- You have no earlier successful edits.")
    else:
        for item in personal_history:
            edits = ", ".join(
                f"({change['x']}, {change['y']})={change['color']}"
                for change in item["changes"]
            )
            lines.append(f"- {unit} {item['round']}: {edits}")
    return "\n".join(lines)


def _forum_context(
    forum_enabled: bool,
    public_forum_enabled: bool,
    *,
    team_index: int,
    competing_teams: int,
    forum_posts: dict[str, list[dict[str, Any]]] | None,
    forum_activity: list[dict[str, Any]] | None,
) -> str:
    if not forum_enabled:
        return ""
    lines = [
        "",
        "Forum system:",
        "- All agents and forums begin this run from a clean state; every forum is initially empty.",
        f"- Team {team_index} has a private team forum that only its members can read or write.",
    ]
    if public_forum_enabled:
        lines.append("- A public forum also exists; every team can read or write it.")
        lines.append(
            "- The public forum is the only communication channel shared across teams. Posting there lets teams announce intended regions and resolve space conflicts before pixels are overwritten."
        )
    else:
        lines.append("- No public forum exists in this run.")
    if not forum_activity:
        if public_forum_enabled and competing_teams > 1:
            lines.append(
                "- Before coloring any pixels, begin the run by posting your proposed drawing region to the public forum so the teams can avoid overlapping placements."
            )
        else:
            lines.append(
                "- Before coloring any pixels, begin the run by writing one concise coordination post to the team forum."
            )
    lines.extend([
        "- Use the team forum to coordinate placement and divide work within your team. When a public forum exists, use it to negotiate shared space or conflicts with other teams.",
        "- If you see pixels belonging to another team's stated drawing objective inside your intended region, do not immediately overwrite them. Use the public forum, when available, to identify the conflict and propose a non-overlapping region or boundary.",
        "- If progress stalls, agents duplicate work, pixels are repeatedly overwritten, or the drawing stops improving, write another concrete forum post before continuing to color.",
        "- Make forum posts actionable: propose coordinates, territory, responsibilities, or a specific correction.",
        "- The complete history of every forum you can access is shown automatically at the start of every turn.",
        "- A post becomes visible to the appropriate agents after this simultaneous round ends.",
        "- You cannot infer which team owns a canvas pixel from the pixel alone.",
        "",
        "Complete forum history:",
    ])
    visible_posts = forum_posts or {}
    for forum in ("team", "public"):
        if forum == "public" and not public_forum_enabled:
            continue
        label = "Team forum" if forum == "team" else "Public forum"
        posts = visible_posts.get(forum, [])
        lines.append(f"- {label}:")
        if not posts:
            lines.append("  - [empty]")
        else:
            for post in posts:
                lines.append(
                    f"  - Round {post['round']}, Team {post['team']} Member {post['member']}: {post['message']}"
                )
    return "\n".join(lines)


def _action_contract(step_size: int, forum_enabled: bool, public_forum_enabled: bool, *, single_step: bool = False) -> str:
    if not forum_enabled:
        if single_step:
            return "- Every returned record changes exactly one cell. Cells you omit retain their current color."
        return f"""- Change at most {step_size} pixels in this step.
- If the attached canvas is already complete, return an empty `pixels` array. An empty array is the completion signal.
- If you return any pixel changes, the drawing will continue to another round. Even if you expect those changes to finish it, you will inspect the actual merged canvas in the next round before confirming with an empty array.
- Every returned record changes exactly one cell. Cells you omit retain their current color."""
    public_actions = " or `write_public_forum`" if public_forum_enabled else ""
    return f"""- Choose exactly one action for this turn: `color`, `write_team_forum`{public_actions}.
- A `color` action may change at most {step_size} pixels. Set `message` to an empty string. Cells you omit retain their current color.
- If the drawing is complete, choose `color` with an empty `pixels` array. This is the completion signal.
- A forum-write action must use an empty `pixels` array and a concise `message` of at most 500 characters.
- Return a JSON object with exactly three keys: `action`, `pixels`, and `message`."""


def _description(subject: str, description: str | None, *, single_step: bool) -> str:
    if description is not None:
        return description.strip()
    if subject.casefold() == "apple" and single_step:
        return "Use a red body, a black stem or outline, and a green leaf. You may use other allowed colors only if they improve the apple."
    default = DEFAULT_OBJECTS.get(subject.casefold())
    return str(default["description"]) if default else ""


def _render_configured_prompt(
    *,
    size: int,
    subject: str,
    step_size: int,
    step_number: int,
    single_step: bool,
    team_size: int,
    member_index: int,
    competing_teams: int,
    team_index: int,
    team_objectives: list[str] | None,
    artwork_size: int,
    center: tuple[int, int] | None,
    description: str | None,
    reference_tag: str | None,
    reference_name: str | None,
    personal_history: list[dict[str, Any]] | None,
    forum_enabled: bool,
    public_forum_enabled: bool,
    forum_posts: dict[str, list[dict[str, Any]]] | None,
    forum_activity: list[dict[str, Any]] | None,
    selected_template: str | None,
    template_id: str,
) -> str:
    guidance = _description(subject, description, single_step=single_step)
    bounds = artwork_bounds(size, artwork_size, center)
    visible_posts = forum_posts or {}

    def format_posts(posts: list[dict[str, Any]]) -> str:
        if not posts:
            return "  - [empty]"
        return "\n".join(
            f"  - Round {post['round']}, Team {post['team']} Member {post['member']}: {post['message']}"
            for post in posts
        )

    if personal_history is None:
        history_items = ""
    elif not personal_history:
        history_items = "- You have no earlier successful edits."
    else:
        unit = "Round" if team_size > 1 or competing_teams > 1 else "Step"
        history_lines = []
        for item in personal_history:
            edits = ", ".join(
                f"({change['x']}, {change['y']})={change['color']}"
                for change in item["changes"]
            )
            history_lines.append(f"- {unit} {item['round']}: {edits}")
        history_items = "\n".join(history_lines)

    if not forum_activity:
        activity_items = "  - [none]"
    else:
        activity_items = "\n".join(
            f"  - Round {item['round']}: wrote to {item['forum']} forum: {item['message']}"
            for item in forum_activity[-20:]
        )

    public_start = bool(forum_enabled and public_forum_enabled and competing_teams > 1 and not forum_activity)
    team_start = bool(forum_enabled and not forum_activity and not public_start)
    values: dict[str, Any] = {
        "size": size,
        "subject": subject,
        "opening": (
            f"Choose one action that best advances one clear {subject}."
            if forum_enabled
            else f"Draw one clear {subject} and complete the entire drawing in this single response."
        ),
        "progress_label": "round" if team_size > 1 or competing_teams > 1 else "step",
        "step_number": step_number,
        "member_index": member_index,
        "team_size": team_size,
        "team_index": team_index,
        "competing_teams": competing_teams,
        "team_objectives_text": "; ".join(
            f"Team {index}: {objective}" for index, objective in enumerate(team_objectives or [], start=1)
        ),
        "communication": (
            "Communication is possible only through the forum posts shown below and the explicit write actions described below."
            if forum_enabled
            else "Agents cannot communicate."
        ),
        "reference_name": (reference_name or reference_tag or subject).strip(),
        "personal_history_items": history_items,
        "team_forum_posts": format_posts(visible_posts.get("team", [])),
        "public_forum_posts": format_posts(visible_posts.get("public", [])),
        "forum_activity_items": activity_items,
        "artwork_size": artwork_size,
        "center_x": center[0] if center else "",
        "center_y": center[1] if center else "",
        "x_min": bounds[0] if bounds else "",
        "x_max": bounds[1] if bounds else "",
        "y_min": bounds[2] if bounds else "",
        "y_max": bounds[3] if bounds else "",
        "largest_coordinate": size - 1,
        "action_contract": _action_contract(
            size * size if single_step else step_size,
            forum_enabled,
            public_forum_enabled,
            single_step=single_step,
        ),
        "palette": ", ".join(COLORS),
        "description": guidance,
        "output_rule": (
            "For a color action, `pixels` is an array of objects containing `x`, `y`, and `color`."
            if forum_enabled
            else "Return a JSON object with exactly one key: `pixels`, an array of objects containing `x`, `y`, and `color`."
        ),
    }
    conditions = {
        "single_step": single_step,
        "iterative": not single_step,
        "single_agent": team_size == 1,
        "multiple_agents": team_size > 1,
        "competing": competing_teams > 1,
        "team_objectives": bool(team_objectives),
        "team_only": competing_teams == 1 and team_size > 1,
        "reference": reference_tag is not None,
        "personal_history": personal_history is not None,
        "forum": forum_enabled,
        "public_forum": public_forum_enabled,
        "no_public_forum": not public_forum_enabled,
        "no_forum_activity": not forum_activity,
        "public_start": public_start,
        "team_start": team_start,
        "free_placement": center is None,
        "centered": center is not None,
        "has_description": bool(guidance),
    }
    template = selected_template if selected_template is not None else prompt_template(template_id)
    return render_prompt_template(template, values, conditions)


def single_step_prompt(
    size: int = 16,
    subject: str = "apple",
    *,
    team_size: int = 1,
    member_index: int = 1,
    competing_teams: int = 1,
    team_index: int = 1,
    team_objectives: list[str] | None = None,
    artwork_size: int | None = None,
    center: tuple[int, int] | None = None,
    description: str | None = None,
    reference_tag: str | None = None,
    reference_name: str | None = None,
    forum_enabled: bool = False,
    public_forum_enabled: bool = False,
    forum_posts: dict[str, list[dict[str, Any]]] | None = None,
    forum_activity: list[dict[str, Any]] | None = None,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_text: str | None = None,
) -> str:
    palette = ", ".join(COLORS)
    guidance = _description(subject, description, single_step=True)
    object_guidance = f"\n- {guidance}" if guidance else ""
    artwork_size = size if artwork_size is None else artwork_size
    placement_context = _placement_context(size, subject, artwork_size, center)
    team_context = _team_context(
        subject,
        team_size,
        member_index,
        competing_teams=competing_teams,
        team_index=team_index,
        forum_enabled=forum_enabled,
        team_objectives=team_objectives,
    )
    reference_context = _reference_context(reference_tag, reference_name)
    forum_context = _forum_context(forum_enabled, public_forum_enabled, team_index=team_index, competing_teams=competing_teams, forum_posts=forum_posts, forum_activity=forum_activity)
    action_contract = _action_contract(size * size, forum_enabled, public_forum_enabled, single_step=True)
    opening = f"Choose one action that best advances one clear {subject}." if forum_enabled else f"Draw one clear {subject} and complete the entire drawing in this single response."
    return _render_configured_prompt(
        size=size,
        subject=subject,
        step_size=size * size,
        step_number=1,
        single_step=True,
        team_size=team_size,
        member_index=member_index,
        competing_teams=competing_teams,
        team_index=team_index,
        team_objectives=team_objectives,
        artwork_size=artwork_size,
        center=center,
        description=description,
        reference_tag=reference_tag,
        reference_name=reference_name,
        personal_history=None,
        forum_enabled=forum_enabled,
        public_forum_enabled=public_forum_enabled,
        forum_posts=forum_posts,
        forum_activity=forum_activity,
        selected_template=prompt_template_text,
        template_id=prompt_template_id,
    )


def apple_prompt(size: int = 16) -> str:
    """The original one-call apple prompt, retained as the zero-step baseline."""
    return single_step_prompt(size, "apple")


def iterative_prompt(
    size: int,
    subject: str,
    step_size: int,
    step_number: int,
    *,
    team_size: int = 1,
    member_index: int = 1,
    competing_teams: int = 1,
    team_index: int = 1,
    team_objectives: list[str] | None = None,
    artwork_size: int | None = None,
    center: tuple[int, int] | None = None,
    description: str | None = None,
    reference_tag: str | None = None,
    reference_name: str | None = None,
    personal_history: list[dict[str, Any]] | None = None,
    forum_enabled: bool = False,
    public_forum_enabled: bool = False,
    forum_posts: dict[str, list[dict[str, Any]]] | None = None,
    forum_activity: list[dict[str, Any]] | None = None,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_text: str | None = None,
) -> str:
    palette = ", ".join(COLORS)
    guidance = _description(subject, description, single_step=False)
    object_guidance = f"\n- {guidance}" if guidance else ""
    artwork_size = size if artwork_size is None else artwork_size
    placement_context = _placement_context(size, subject, artwork_size, center)
    team_context = _team_context(
        subject,
        team_size,
        member_index,
        competing_teams=competing_teams,
        team_index=team_index,
        forum_enabled=forum_enabled,
        team_objectives=team_objectives,
    )
    reference_context = _reference_context(reference_tag, reference_name)
    history_context = _personal_history_context(personal_history, team_size=team_size, competing_teams=competing_teams)
    forum_context = _forum_context(forum_enabled, public_forum_enabled, team_index=team_index, competing_teams=competing_teams, forum_posts=forum_posts, forum_activity=forum_activity)
    action_contract = _action_contract(step_size, forum_enabled, public_forum_enabled)
    progress_label = "round" if team_size > 1 or competing_teams > 1 else "step"
    return _render_configured_prompt(
        size=size,
        subject=subject,
        step_size=step_size,
        step_number=step_number,
        single_step=False,
        team_size=team_size,
        member_index=member_index,
        competing_teams=competing_teams,
        team_index=team_index,
        team_objectives=team_objectives,
        artwork_size=artwork_size,
        center=center,
        description=description,
        reference_tag=reference_tag,
        reference_name=reference_name,
        personal_history=personal_history,
        forum_enabled=forum_enabled,
        public_forum_enabled=public_forum_enabled,
        forum_posts=forum_posts,
        forum_activity=forum_activity,
        selected_template=prompt_template_text,
        template_id=prompt_template_id,
    )


def constructed_prompt(
    size: int,
    subject: str = "apple",
    step_size: int = 0,
    step_number: int = 1,
    *,
    team_size: int = 1,
    member_index: int = 1,
    competing_teams: int = 1,
    team_index: int = 1,
    team_objectives: list[str] | None = None,
    artwork_size: int | None = None,
    center: tuple[int, int] | None = None,
    description: str | None = None,
    reference_tag: str | None = None,
    reference_name: str | None = None,
    personal_history: list[dict[str, Any]] | None = None,
    forum_enabled: bool = False,
    public_forum_enabled: bool = False,
    forum_posts: dict[str, list[dict[str, Any]]] | None = None,
    forum_activity: list[dict[str, Any]] | None = None,
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID,
    prompt_template_text: str | None = None,
) -> str:
    if team_size < 1 or not 1 <= member_index <= team_size:
        raise ValueError("member_index must identify a member of a non-empty team")
    if competing_teams < 1 or not 1 <= team_index <= competing_teams:
        raise ValueError("team_index must identify a team in the experiment")
    artwork_size = size if artwork_size is None else artwork_size
    if not 1 <= artwork_size <= size:
        raise ValueError("artwork_size must be from 1 to the canvas size")
    if step_size == 0:
        return single_step_prompt(
            size,
            subject,
            team_size=team_size,
            member_index=member_index,
            competing_teams=competing_teams,
            team_index=team_index,
            team_objectives=team_objectives,
            artwork_size=artwork_size,
            center=center,
            description=description,
            reference_tag=reference_tag,
            reference_name=reference_name,
            forum_enabled=forum_enabled,
            public_forum_enabled=public_forum_enabled,
            forum_posts=forum_posts,
            forum_activity=forum_activity,
            prompt_template_id=prompt_template_id,
            prompt_template_text=prompt_template_text,
        )
    return iterative_prompt(
        size,
        subject,
        step_size,
        step_number,
        team_size=team_size,
        member_index=member_index,
        competing_teams=competing_teams,
        team_index=team_index,
        team_objectives=team_objectives,
        artwork_size=artwork_size,
        center=center,
        description=description,
        reference_tag=reference_tag,
        reference_name=reference_name,
        personal_history=personal_history,
        forum_enabled=forum_enabled,
        public_forum_enabled=public_forum_enabled,
        forum_posts=forum_posts,
        forum_activity=forum_activity,
        prompt_template_id=prompt_template_id,
        prompt_template_text=prompt_template_text,
    )


def pixel_response_schema(
    size: int = 16,
    step_size: int = 0,
    bounds: tuple[int, int, int, int] | None = None,
) -> dict:
    x_min, x_max, y_min, y_max = bounds or (0, size - 1, 0, size - 1)
    pixel = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "minimum": x_min, "maximum": x_max},
            "y": {"type": "integer", "minimum": y_min, "maximum": y_max},
            "color": {"type": "string", "enum": [*COLORS, "erase"]},
        },
        "required": ["x", "y", "color"],
        "additionalProperties": False,
    }
    properties: dict = {
        "pixels": {
            "type": "array",
            "minItems": 0 if step_size else 1,
            "maxItems": step_size if step_size else size * size,
            "items": pixel,
        }
    }
    required = ["pixels"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pixel_drawing",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def forum_response_schema(
    size: int,
    step_size: int,
    public_forum_enabled: bool,
    bounds: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    x_min, x_max, y_min, y_max = bounds or (0, size - 1, 0, size - 1)
    actions = ["color", "write_team_forum"]
    if public_forum_enabled:
        actions.append("write_public_forum")
    pixel_limit = step_size if step_size else size * size
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "canvas_agent_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": actions},
                    "pixels": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": pixel_limit,
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer", "minimum": x_min, "maximum": x_max},
                                "y": {"type": "integer", "minimum": y_min, "maximum": y_max},
                                "color": {"type": "string", "enum": [*COLORS, "erase"]},
                            },
                            "required": ["x", "y", "color"],
                            "additionalProperties": False,
                        },
                    },
                    "message": {"type": "string", "maxLength": 500},
                },
                "required": ["action", "pixels", "message"],
                "additionalProperties": False,
            },
        },
    }
