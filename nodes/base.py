"""
Shared base classes for Yvann Nodes.
This module provides base classes for all node categories to avoid duplication.
"""
from .. import Yvann


class AudioNodeBase(Yvann):
    """Base class for all audio-related nodes."""
    CATEGORY = "👁️ Yvann Nodes/🔊 Audio"


class UtilsNodeBase(Yvann):
    """Base class for all utility nodes."""
    CATEGORY = "👁️ Yvann Nodes/🛠️ Utils"


class ConvertNodeBase(Yvann):
    """Base class for all conversion nodes."""
    CATEGORY = "👁️ Yvann Nodes/🔄 Convert"
