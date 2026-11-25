"""Convert v2 stream parts into protocol events (mirrors ``langgraph.stream._convert``)."""

from __future__ import annotations

import time
from typing import Any, cast

from ._types import ProtocolEvent, _ProtocolEventParams


def convert_to_protocol_event(part: Any) -> ProtocolEvent:
    """Convert a v2 `StreamPart` into a `ProtocolEvent`."""
    part_dict = cast(dict[str, Any], part)
    params: _ProtocolEventParams = {
        "namespace": list(part_dict["ns"]),
        "timestamp": int(time.time() * 1000),
        "data": part_dict["data"],
    }
    if "interrupts" in part_dict:
        params["interrupts"] = part_dict["interrupts"]
    return {
        "type": "event",
        "method": part_dict["type"],
        "params": params,
    }