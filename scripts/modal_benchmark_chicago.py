"""Benchmark the weighted engineered-CCH AON kernel on Modal x86 CPUs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


app = modal.App("engineered-cch-chicago-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "scons", "libboost-all-dev", "libgomp1")
    .pip_install("numpy", "pandas", "pyarrow")
    .add_local_dir(
        "/Users/waddell/src/routing-framework",
        remote_path="/root/routing-framework",
        ignore=[".git", "Build", "External/RoutingKit/lib", "External/RoutingKit/bin"],
    )
    .add_local_dir(
        "/Users/waddell/src/manta2/data/chicago_regional/normalized",
        remote_path="/data/chicago_regional",
    )
    .add_local_dir(
        "/Users/waddell/src/RoutingKit",
        remote_path="/root/RoutingKit-modern",
        ignore=[".git", "build", "lib", "bin"],
    )
)


def _prepare_inputs(
    source: Path, output: Path, cost_scale: int = 100_000, demand_limit: int | None = None
) -> None:
    import json
    import numpy as np
    import pandas as pd

    nodes = pd.read_parquet(source / "nodes.parquet").sort_values("node_id")
    links = pd.read_parquet(source / "links.parquet")
    demand = pd.read_parquet(source / "od_demand.parquet")
    with (source / "manifest.json").open() as stream:
        zones = int(json.load(stream)["summary"]["zones"])
    node_ids = nodes.node_id.to_numpy(np.int64)
    if not np.array_equal(node_ids, np.arange(1, len(nodes) + 1)):
        raise ValueError("Chicago node IDs must be dense and one based")

    def normalize(values: np.ndarray, span: float) -> np.ndarray:
        midpoint = (values.min() + values.max()) / 2
        half_range = max((values.max() - values.min()) / 2, 1)
        return (values - midpoint) / half_range * span

    node_frame = pd.DataFrame(
        {
            "node": node_ids - 1,
            "lat": normalize(nodes.y_stateplane_ft.to_numpy(np.float64), 45.0),
            "lon": normalize(nodes.x_stateplane_ft.to_numpy(np.float64), 90.0),
        }
    )
    destination_centroids = node_frame.iloc[:zones].copy()
    destination_centroids["node"] = len(nodes) + np.arange(zones)
    node_frame = pd.concat([node_frame, destination_centroids], ignore_index=True)
    generalized_cost = (
        links.free_flow_time_min_raw.to_numpy(np.float64)
        + 0.25 * links.length_miles_raw.to_numpy(np.float64)
        + 0.1 * links.toll_cents.to_numpy(np.float64)
    )
    equivalent_alpha = (
        links.bpr_alpha.to_numpy(np.float64)
        * links.free_flow_time_min_raw.to_numpy(np.float64)
        / generalized_cost
    )
    arc_frame = pd.DataFrame(
        {
            "tail": links.source_node_id.to_numpy(np.int64) - 1,
            "head": np.where(
                links.target_node_id.to_numpy(np.int64) <= zones,
                len(nodes) + links.target_node_id.to_numpy(np.int64) - 1,
                links.target_node_id.to_numpy(np.int64) - 1,
            ),
            "free_flow": generalized_cost,
            "alpha": equivalent_alpha,
            "beta": links.bpr_beta.to_numpy(np.float64),
            "capacity": links.capacity_veh_per_hour.to_numpy(np.float64),
        }
    )
    demand_frame = pd.DataFrame(
        {
            "origin": demand.origin_zone_id.to_numpy(np.int64) - 1,
            "destination": np.where(
                demand.origin_zone_id.to_numpy(np.int64)
                == demand.destination_zone_id.to_numpy(np.int64),
                demand.origin_zone_id.to_numpy(np.int64) - 1,
                len(nodes) + demand.destination_zone_id.to_numpy(np.int64) - 1,
            ),
            "volume": demand.volume.to_numpy(np.float64),
        }
    )
    if demand_limit is not None:
        demand_frame = demand_frame.iloc[:demand_limit]
    output.mkdir(parents=True, exist_ok=True)
    node_frame.to_csv(output / "nodes.csv", index=False)
    arc_frame.to_csv(output / "arcs.csv", index=False)
    demand_frame.to_csv(output / "demand.csv", index=False)


@app.function(image=image, cpu=16, timeout=3600)
def benchmark(thread_count: int = 16) -> dict:
    root = Path("/root/routing-framework")
    routingkit = Path("/root/RoutingKit-modern")
    subprocess.run(["make", "-j16"], cwd=routingkit, check=True)
    environment = {
        "CPLUS_INCLUDE_PATH": ":".join(
            [
                str(routingkit / "include"),
                str(root / "External/fast-cpp-csv-parser"),
                str(root / "External"),
            ]
        ),
        "LIBRARY_PATH": str(routingkit / "lib"),
        "LD_LIBRARY_PATH": str(routingkit / "lib"),
        "OMP_NUM_THREADS": str(thread_count),
    }
    import os

    build_env = os.environ | environment
    subprocess.run(
        ["scons", "-j16", "variant=Release", "simd=avx", "def=TA_LOG_K=5",
         "def=TA_CCH_PROFILE",
         "BenchmarkWeightedCCHEquilibrium"],
        cwd=root,
        env=build_env,
        check=True,
    )
    inputs = Path("/tmp/chicago-cch")
    _prepare_inputs(Path("/data/chicago_regional"), inputs, demand_limit=None)
    command = [
        str(root / "Build/Release/Launchers/BenchmarkWeightedCCHEquilibrium"),
        str(inputs / "nodes.csv"),
        str(inputs / "arcs.csv"),
        str(inputs / "demand.csv"),
    ]
    runs = []
    for repeat in range(1):
        completed = subprocess.run(
            command, env=build_env, check=True, text=True, capture_output=True
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        result["threads"] = thread_count
        result["simd_lanes"] = 32
        result["cost_scale"] = 100_000
        result["repeat"] = repeat
        runs.append(result)
    return {"runs": runs}


@app.local_entrypoint()
def main() -> None:
    for threads in (16,):
        print(json.dumps(benchmark.remote(threads), indent=2, sort_keys=True))
