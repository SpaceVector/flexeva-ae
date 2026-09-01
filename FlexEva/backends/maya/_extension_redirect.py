"""Load generated CppEvent extension modules from stable proxy modules.

The tracked ``CppEvent/cpp_event_py.py`` and ``CppEvent/cpp_event_tls.py`` files
are import-stable proxies.  The compiled extension artifacts live under the
repository ``python/`` directory and include the Python ABI tag, for example
``cpp_event_py.cpython-312-x86_64-linux-gnu.so``.  This helper finds and loads
that artifact while preserving the public proxy import name.

When no compiled artifact for the current interpreter is available, lightweight
fallbacks keep local schema/serialization tests importable.  Real capture still
requires the compiled extension artifact because wrapper recorder callbacks are
implemented in C++.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from datetime import timedelta
from enum import Enum
from pathlib import Path
from types import ModuleType


def _candidate_extensions(module_name: str, search_dirs: tuple[Path, ...]) -> list[Path]:
    candidates: list[Path] = []
    for directory in search_dirs:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidates.extend(sorted(directory.glob(f"{module_name}*{suffix}")))
    return candidates


def _load_native_extension(module_name: str, path: Path) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create extension spec for {module_name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.__dict__


class _FallbackEventKind(Enum):
    Unknown = 0
    ComputeKernel = 1
    Collective = 2
    PointToPoint = 3
    Barrier = 4
    AllReduce = 5
    Broadcast = 6
    AllGather = 7
    ReduceScatter = 8
    Send = 9
    Recv = 10
    RuntimeCall = 11
    MemcpyHostToDevice = 12
    MemcpyDeviceToHost = 13
    MemcpyDeviceToDevice = 14
    MemoryAllocation = 15
    MemoryFree = 16
    FileRead = 17
    FileWrite = 18
    Debug = 19


class _FallbackEventDomain(Enum):
    Unknown = 0
    Compute = 1
    Communication = 2
    Synchronization = 3
    Runtime = 4
    Memory = 5
    IO = 6


class _FallbackEventScope(Enum):
    Local = 0
    CrossRank = 1


class _FallbackRankGroup:
    def __init__(self) -> None:
        self.id = ""
        self.members: list[int] = []

    def empty(self) -> bool:
        return not self.members

    def size(self) -> int:
        return len(self.members)

    def contains(self, rank: int) -> bool:
        return rank in self.members


class _FallbackPlacement:
    def __init__(self) -> None:
        self.world_size = 0
        self.device = 0
        self.stream = 0
        self.group = _FallbackRankGroup()


class _FallbackEventPayload:
    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}


class _FallbackEventRecord:
    def __init__(self) -> None:
        self.id = 0
        self.domain = _FallbackEventDomain.Unknown
        self.kind = _FallbackEventKind.Unknown
        self.scope = _FallbackEventScope.Local
        self.process_id = 0
        self.thread_id = 0
        self.active_group = _FallbackRankGroup()
        self.api_name = ""
        self.placement = _FallbackPlacement()
        self.timestamp = timedelta(0)
        self.end_timestamp = timedelta(0)
        self.host_duration = timedelta(0)
        self.payload = _FallbackEventPayload()

    def node_count_hint(self) -> int:
        return 1


class _FallbackEventContext:
    def set_active_group(self, *args: object, **kwargs: object) -> None: pass
    def reset_active_group(self) -> None: pass
    def set_placement(self, *args: object, **kwargs: object) -> None: pass
    def reset_placement(self) -> None: pass
    def set_scope(self, *args: object, **kwargs: object) -> None: pass
    def reset_scope(self) -> None: pass
    def snapshot(self) -> dict[str, object]: return {}


class _FallbackEventLog:
    def __init__(self, context: _FallbackEventContext) -> None:
        self._events: list[_FallbackEventRecord] = []

    def append(self, api_name: str, kind: _FallbackEventKind, start_time: timedelta, payload: _FallbackEventPayload | None = None) -> _FallbackEventRecord:
        event = _FallbackEventRecord()
        event.api_name = api_name
        event.kind = kind
        event.timestamp = start_time
        event.end_timestamp = start_time
        if payload is not None:
            event.payload = payload
        self._events.append(event)
        return event

    def append_with_end(self, api_name: str, kind: _FallbackEventKind, start_time: timedelta, end_time: timedelta, payload: _FallbackEventPayload | None = None) -> _FallbackEventRecord:
        event = self.append(api_name, kind, start_time, payload)
        event.end_timestamp = end_time
        event.host_duration = end_time - start_time
        return event

    def snapshot(self) -> list[_FallbackEventRecord]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()


class _FallbackEventLogRecorderAdapter:
    def __init__(self, log: _FallbackEventLog) -> None:
        self._log = log

    def log(self) -> _FallbackEventLog:
        return self._log


_current_recorder: object | None = None


def _fallback_cpp_event_py() -> dict[str, object]:
    def set_recorder(adapter: object) -> None:
        global _current_recorder
        _current_recorder = adapter

    def reset_recorder() -> None:
        global _current_recorder
        _current_recorder = None

    def current_recorder() -> None:
        return None

    def resolve_async_runtime_observations(log: object) -> None:
        return None

    def clear_async_runtime_observations() -> None:
        return None

    module = ModuleType("cpp_event_py")
    module.EventKind = _FallbackEventKind
    module.EventDomain = _FallbackEventDomain
    module.EventScope = _FallbackEventScope
    module.RankGroup = _FallbackRankGroup
    module.Placement = _FallbackPlacement
    module.EventPayload = _FallbackEventPayload
    module.EventRecord = _FallbackEventRecord
    module.EventContext = _FallbackEventContext
    module.EventLog = _FallbackEventLog
    module.EventLogRecorderAdapter = _FallbackEventLogRecorderAdapter
    module.set_recorder = set_recorder
    module.reset_recorder = reset_recorder
    module.current_recorder = current_recorder
    module.resolve_async_runtime_observations = resolve_async_runtime_observations
    module.clear_async_runtime_observations = clear_async_runtime_observations
    module.__fallback__ = True
    return module.__dict__


def _fallback_cpp_event_tls() -> dict[str, object]:
    active_ranks: list[int] = []

    def set_active_ranks_in_tls(ranks: list[int]) -> None:
        active_ranks[:] = list(ranks)

    def get_active_ranks_from_tls() -> list[int]:
        return list(active_ranks)

    def pthread_create_with_context(*args: object, **kwargs: object) -> int:
        return -1

    def fork_with_context() -> int:
        return -1

    module = ModuleType("cpp_event_tls")
    module.set_active_ranks_in_tls = set_active_ranks_in_tls
    module.get_active_ranks_from_tls = get_active_ranks_from_tls
    module.pthread_create_with_context = pthread_create_with_context
    module.fork_with_context = fork_with_context
    module.__fallback__ = True
    return module.__dict__


def load_extension(module_name: str, *search_dirs: Path) -> dict[str, object]:
    candidates = _candidate_extensions(module_name, tuple(Path(path) for path in search_dirs))
    if candidates:
        return _load_native_extension(module_name, candidates[0])
    if module_name == "cpp_event_py":
        return _fallback_cpp_event_py()
    if module_name == "cpp_event_tls":
        return _fallback_cpp_event_tls()
    raise ImportError(
        f"no compiled extension found for {module_name}; searched: "
        + ", ".join(str(path) for path in search_dirs)
    )
