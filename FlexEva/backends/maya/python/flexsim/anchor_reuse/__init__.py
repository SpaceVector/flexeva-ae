"""Self-contained source-analysis subset used by the AE package.

The author snapshot also contains later anchor-policy experiments that require
private trace corpora.  The AE initializer exports only the paper-relevant,
CPU-only source-analysis path.
"""

from .control_flow import (
    ControlFlowBlock,
    ControlFlowBlockKind,
    ControlFlowEdge,
    ControlFlowGraph,
    build_control_flow_graphs,
)
from .control_regions import ControlRegionKind
from .dryrun_bridge import (
    BlackBoxBoundaryRule,
    LogicScopeSpec,
    LogicSliceGranularityMode,
    LogicSliceGranularityPolicy,
    capture_program_logic_from_instrumented_code,
)
from .resilient_anchor_state import (
    ResilientAnchorState,
    build_resilient_anchor_state,
    resilient_anchor_state_summary,
)

__all__ = [
    "BlackBoxBoundaryRule",
    "ControlFlowBlock",
    "ControlFlowBlockKind",
    "ControlFlowEdge",
    "ControlFlowGraph",
    "ControlRegionKind",
    "LogicScopeSpec",
    "LogicSliceGranularityMode",
    "LogicSliceGranularityPolicy",
    "ResilientAnchorState",
    "build_control_flow_graphs",
    "build_resilient_anchor_state",
    "capture_program_logic_from_instrumented_code",
    "resilient_anchor_state_summary",
]
