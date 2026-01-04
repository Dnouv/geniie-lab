from typing import Optional

from geniie_lab.dataclasses.setting import ExperimentState


def render_instruction(template: str, state: ExperimentState) -> str:
    if "{metrics}" not in template:
        return template
    metrics = state.metrics_context or "No metrics available yet."
    return template.replace("{metrics}", metrics)
