from __future__ import annotations

import json

from flexsim.maya_lite.augment_emulated_helper_threads import (
    augment_trace_file,
    main,
    record_helper_thread_augmentation_status,
)


def test_augment_trace_file_injects_missing_helper_threads(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()

    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"step_windows": {"0": {"start_ts": 1000, "end_ts": 6000}}}),
        encoding="utf-8",
    )
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 1000,
                        "pid": 7,
                        "tid": 11,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 3000,
                        "pid": 7,
                        "tid": 12,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                    "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 3200,
                        "pid": 7,
                        "tid": 13,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (summary_dir / "rank_0.json").write_text(
        json.dumps(
            {
                "helper_thread_templates_by_rank": {
                    "0": [
                        {
                            "source_tid": 99,
                            "start_offset_us": 500,
                            "end_offset_us": 5500,
                            "first_api": "ncclCommGetAsyncError",
                            "dominant_api": "ncclCommGetAsyncError",
                            "event_count": 10,
                            "api_sequence": [
                                "ncclCommGetAsyncError",
                                "cudaSetDevice",
                                "cudaGetLastError",
                                "ncclCommGetAsyncError",
                            ],
                        },
                        {
                            "source_tid": 100,
                            "start_offset_us": 700,
                            "end_offset_us": 700,
                            "first_api": "ncclGetVersion",
                            "dominant_api": "ncclGetVersion",
                            "event_count": 1,
                            "api_sequence": ["ncclGetVersion"],
                        },
                        {
                            "source_tid": 101,
                            "start_offset_us": 900,
                            "end_offset_us": 5900,
                            "first_api": "cudaSetDevice",
                            "dominant_api": "cudaSetDevice",
                            "event_count": 12,
                            "api_sequence": [
                                "cudaSetDevice",
                                "cudaGetDevice",
                                "cudaSetDevice",
                            ],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    injected = augment_trace_file(
        trace_path,
        summary_path=summary_dir / "rank_0.json",
        step_window=(1000, 6000),
    )

    assert injected["injected_events"] == 8
    assert injected["synthetic_min_ts"] == 1500
    assert injected["synthetic_max_ts"] == 6900
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    helper_events = [event for event in events if event.get("synthetic_helper_thread")]
    assert len(helper_events) == 8
    assert helper_events[0]["api"] == "ncclCommGetAsyncError"
    assert helper_events[0]["ts"] == 1500
    assert helper_events[1]["api"] == "ncclGetVersion"
    assert helper_events[1]["ts"] == 1700
    assert helper_events[2]["api"] == "cudaSetDevice"
    assert helper_events[2]["type"] == "context_op"
    assert helper_events[2]["ts"] == 1900
    assert helper_events[3]["api"] == "cudaSetDevice"
    assert helper_events[3]["type"] == "context_op"
    assert helper_events[3]["ts"] == 2833
    assert helper_events[4]["api"] == "cudaGetLastError"
    assert helper_events[4]["ts"] == 4167
    assert helper_events[5]["api"] == "cudaGetDevice"
    assert helper_events[5]["type"] == "context_op"
    assert helper_events[5]["ts"] == 4400
    assert helper_events[6]["api"] == "ncclCommGetAsyncError"
    assert helper_events[6]["ts"] == 5500
    assert helper_events[7]["api"] == "cudaSetDevice"
    assert helper_events[7]["type"] == "context_op"
    assert helper_events[7]["ts"] == 6900


def test_record_helper_thread_augmentation_status_marks_not_required(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(json.dumps({}), encoding="utf-8")

    payload = record_helper_thread_augmentation_status(
        trace_dir,
        expected=False,
        status="not_required",
        summary_dir=None,
    )

    assert payload == {
        "expected": False,
        "status": "not_required",
        "embedded_in_emulator_artifact": True,
        "summary_dir": None,
        "checked_rank_count": 0,
        "injected_by_rank": {},
        "total_injected_events": 0,
        "step_window_extensions_by_rank": {},
    }
    manifest = json.loads((trace_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["helper_thread_augmentation"] == payload


def test_main_extends_capture_manifest_step_window_for_synthetic_helper_tail(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    summary_json = tmp_path / "augment_summary.json"

    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"step_windows": {"0": {"start_ts": 1000, "end_ts": 6000, "source": "trace_markers"}}}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        json.dumps(
            {
                "ts": 1000,
                "pid": 7,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (summary_dir / "rank_0.json").write_text(
        json.dumps(
            {
                "helper_thread_templates_by_rank": {
                    "0": [
                        {
                            "source_tid": 99,
                            "start_offset_us": 500,
                            "end_offset_us": 7000,
                            "first_api": "ncclCommGetAsyncError",
                            "dominant_api": "cudaSetDevice",
                            "event_count": 3,
                            "api_sequence": [
                                "ncclCommGetAsyncError",
                                "cudaSetDevice",
                                "cudaGetLastError",
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert main([str(trace_dir), "--summary-dir", str(summary_dir), "--summary-json", str(summary_json)]) == 0

    manifest = json.loads((trace_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    step_window = manifest["step_windows"]["0"]
    assert step_window["start_ts"] == 1000
    assert step_window["end_ts"] == 6000
    assert step_window["source"] == "trace_markers"
    fidelity_window = manifest["fidelity_windows"]["0"]
    assert fidelity_window["start_ts"] == 1000
    assert fidelity_window["end_ts"] == 8000
    assert fidelity_window["source"] == "trace_markers"
    assert fidelity_window["is_paper_valid_step_window"] is False
    assert fidelity_window["diagnostic_extension"] == "helper_tail"
    assert fidelity_window["diagnostic_only"] is True
    helper_thread_augmentation = manifest["helper_thread_augmentation"]
    assert helper_thread_augmentation["expected"] is True
    assert helper_thread_augmentation["status"] == "completed"
    assert helper_thread_augmentation["embedded_in_emulator_artifact"] is True
    assert helper_thread_augmentation["summary_dir"] == str(summary_dir.resolve())
    assert helper_thread_augmentation["total_injected_events"] == 3
    assert helper_thread_augmentation["injected_by_rank"] == {"0": 3}
    assert helper_thread_augmentation["step_window_extensions_by_rank"] == {
        "0": {"start_ts": 1000, "end_ts": 8000}
    }

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["trace_dir"] == str(trace_dir.resolve())
    assert summary["helper_thread_augmentation"] == helper_thread_augmentation
