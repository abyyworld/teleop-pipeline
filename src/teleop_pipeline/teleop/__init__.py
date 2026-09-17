"""Live teleoperation: drive an arm, record demonstrations the pipeline accepts.

Everything upstream of this package assumed a raw dump already existed. This is
where one comes from. The output is the same raw layout any other rig produces,
so a recorded session is ingested, validated, scored and hashed by exactly the
code that handles a borrowed dataset.
"""

from .arm import Arm, ArmState, SimulatedArm
from .device import Command, InputDevice, KeyboardDevice, ScriptedDevice
from .recorder import (
    EpisodeRecording,
    record_episode,
    scripted_reach,
    write_session,
)

__all__ = [
    "Arm",
    "ArmState",
    "Command",
    "EpisodeRecording",
    "InputDevice",
    "KeyboardDevice",
    "ScriptedDevice",
    "SimulatedArm",
    "record_episode",
    "scripted_reach",
    "write_session",
]
