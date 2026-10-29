"""挖掘 V2 turn 状态和链起点。"""

TURN_STATUSES = ("requested", "evidence_ready", "decided", "completed", "failed")
TURN_TRANSITIONS = {
    "requested": {"evidence_ready", "failed"},
    "evidence_ready": {"decided", "failed"},
    "decided": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}
ZERO_HASH = "0" * 64
