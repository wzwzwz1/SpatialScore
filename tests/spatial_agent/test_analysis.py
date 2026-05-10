import json
from pathlib import Path

from spatial_agent.analysis.analyzer import (
    aggregate_runs,
    infer_vsibench_task_id,
    load_lmms_samples,
    load_spatial_traces,
)
from spatial_agent.analysis.report import write_analysis_report


def test_load_lmms_samples_from_json_bundle(tmp_path):
    bundle_path = tmp_path / "samples_vsibench_bundle.json"
    bundle = {
        "logs": [
            {
                "doc_id": 7,
                "doc": {
                    "question": "How many chairs are visible?",
                    "question_type": "object_counting",
                    "ground_truth": "4",
                    "scene_name": "scene0001",
                },
                "target": "4",
                "filtered_resps": ["3"],
                "MRA:.5:.95:.05": 0.0,
            }
        ]
    }
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    samples = load_lmms_samples(bundle_path, task_name="vsibench", split="test")

    assert len(samples) == 1
    assert samples[0]["task_id"] == "vsibench___test___7"
    assert samples[0]["prediction"] == "3"
    assert samples[0]["question_type"] == "object_counting"


def test_aggregate_runs_merges_samples_and_traces(tmp_path):
    sample_path = tmp_path / "samples_vsibench_bundle.json"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()

    sample_path.write_text(
        json.dumps(
            {
                "logs": [
                    {
                        "doc_id": 0,
                        "doc": {
                            "question": "How many chairs are visible?",
                            "question_type": "object_counting",
                            "ground_truth": "4",
                            "scene_name": "scene_a",
                        },
                        "target": "4",
                        "filtered_resps": ["3"],
                        "MRA:.5:.95:.05": 0.0,
                    },
                    {
                        "doc_id": 1,
                        "doc": {
                            "question": "Which option is correct?",
                            "question_type": "object_rel_distance",
                            "ground_truth": "A",
                            "scene_name": "scene_b",
                        },
                        "target": "A",
                        "filtered_resps": ["A"],
                        "accuracy": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    trace = {
        "task_id": "vsibench___test___0",
        "question": "How many chairs are visible?",
        "status": "success",
        "error": None,
        "final_answer": "3",
        "tool_calls": [{"tool_name": "GetObjectMask", "arguments": {"object_name": "chair"}}],
        "tool_observations": [
            {
                "status": "success",
                "tool_name": "GetObjectMask",
                "payload": {"mask_area": 1200},
                "artifacts": [str(tmp_path / "mask.png")],
                "error": None,
            }
        ],
        "reasoning_trace": [
            {"stage": "reason", "decision": {"thought": "Need segmentation."}},
            {"stage": "observe", "observation": {"status": "success"}},
            {"stage": "finalize", "answer": "3"},
        ],
    }
    (trace_dir / "vsibench___test___0.json").write_text(json.dumps(trace), encoding="utf-8")

    samples = load_lmms_samples(sample_path, task_name="vsibench", split="test")
    traces = load_spatial_traces(trace_dir)
    report = aggregate_runs(samples=samples, traces=traces)

    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["trace_coverage"] == 0.5
    assert report["question_types"]["object_counting"]["count"] == 1
    assert report["question_types"]["object_rel_distance"]["count"] == 1
    assert report["tools"]["GetObjectMask"]["calls"] == 1
    assert report["tools"]["GetObjectMask"]["success"] == 1
    assert report["samples"][0]["artifact_count"] == 1
    assert report["samples"][1]["trace_found"] is False


def test_write_analysis_report_emits_outputs(tmp_path):
    artifact = tmp_path / "depth.png"
    artifact.write_bytes(b"fake-image")

    report = {
        "summary": {
            "sample_count": 1,
            "trace_coverage": 1.0,
            "average_reasoning_steps": 3.0,
            "average_tool_calls": 1.0,
            "success_rate": 1.0,
        },
        "question_types": {
            "object_counting": {
                "count": 1,
                "metrics": {"MRA:.5:.95:.05": 0.0},
            }
        },
        "tools": {
            "EstimateObjectDepth": {
                "calls": 1,
                "success": 1,
                "error": 0,
                "unavailable": 0,
            }
        },
        "samples": [
            {
                "task_id": infer_vsibench_task_id(0),
                "question": "How many chairs are visible?",
                "question_type": "object_counting",
                "ground_truth": "4",
                "prediction": "3",
                "trace_found": True,
                "status": "success",
                "tool_names": ["EstimateObjectDepth"],
                "artifact_paths": [str(artifact)],
                "reasoning_steps": 3,
                "tool_call_count": 1,
                "score_fields": {"MRA:.5:.95:.05": 0.0},
                "error": None,
            }
        ],
        "artifacts": [{"sample_task_id": infer_vsibench_task_id(0), "tool_name": "EstimateObjectDepth", "path": str(artifact)}],
    }

    output_dir = tmp_path / "report"
    paths = write_analysis_report(report, output_dir)

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "samples.csv").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "charts" / "question_type_counts.png").exists()
    assert (output_dir / "artifacts" / "depth.png").exists()
    assert "report_html" in paths
