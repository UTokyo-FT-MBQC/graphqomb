"""Entanglement Scheduling and TICK Slices
==========================================

This example shows how an explicit scheduler controls preparation,
entanglement, and measurement slices, and how those slices are serialized
with TICK commands.
"""

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.pattern import print_pattern
from graphqomb.qompiler import qompile
from graphqomb.scheduler import ScheduleConfig, Scheduler, Strategy

# Create a simple graph state
print("=== Entanglement Scheduling Demo ===\n")
print("1. Creating graph state...")
node_labels = ["input", "middle1", "middle2", "output"]
edges = [("input", "middle1"), ("middle1", "middle2"), ("middle2", "output")]
inputs = ["input"]
outputs = ["output"]
meas_bases = {
    "input": PlannerMeasBasis(Plane.XY, 0.0),
    "middle1": PlannerMeasBasis(Plane.XY, 0.0),
    "middle2": PlannerMeasBasis(Plane.XY, 0.0),
}

graph, node_map = GraphState.from_graph(
    nodes=node_labels,
    edges=edges,
    inputs=inputs,
    outputs=outputs,
    meas_bases=meas_bases,
)

node0 = node_map["input"]
node1 = node_map["middle1"]
node2 = node_map["middle2"]
node3 = node_map["output"]

print(f"   Nodes: {list(graph.nodes)}")
print(f"   Input: {list(graph.input_node_indices.keys())}")
print(f"   Output: {list(graph.output_node_indices.keys())}")
print(f"   Edges: {list(graph.edges)}")

# Define flow
flow = {node0: {node1}, node1: {node2}, node2: {node3}}

# Create scheduler
print("\n2. Creating scheduler and solving schedule...")
scheduler = Scheduler(graph, flow)
config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
success = scheduler.solve_schedule(config)

if success:
    print("   Scheduling successful!")
    print(f"   Number of time slices: {scheduler.num_slices()}")

    # Show preparation times
    prep_times = {k: v for k, v in scheduler.prepare_time.items() if v is not None}
    print(f"   Preparation times: {prep_times}")

    # Show measurement times
    meas_times = {k: v for k, v in scheduler.measure_time.items() if v is not None}
    print(f"   Measurement times: {meas_times}")

    # Show entanglement times (auto-scheduled by solve_schedule)
    print("\n3. Entanglement times (auto-scheduled)...")
    ent_times = {edge: time for edge, time in scheduler.entangle_time.items() if time is not None}
    print(f"   Entanglement times: {ent_times}")

    # Show detailed timeline
    print("\n4. Detailed timeline (Prep, Entangle, Measure):")
    timeline = scheduler.timeline
    for time_idx, (prep_nodes, ent_edges, meas_nodes) in enumerate(timeline):
        print(f"   Time {time_idx}:")
        if prep_nodes:
            print(f"     Prepare: {sorted(prep_nodes)}")
        if ent_edges:
            edges_str = [f"({min(e)},{max(e)})" for e in ent_edges]
            print(f"     Entangle: {', '.join(edges_str)}")
        if meas_nodes:
            print(f"     Measure: {sorted(meas_nodes)}")

    # Compile pattern (TICK commands are inserted automatically per time slice)
    print("\n5. Compiling pattern with scheduler-driven TICK commands...")
    pattern = qompile(graph, flow, scheduler=scheduler)
    print(f"   Pattern has {len(pattern.commands)} commands")
    print(f"   Maximum space usage: {pattern.max_space} qubits")

    print("\n   Pattern commands:")
    print_pattern(pattern, lim=30)

    # Manual entanglement scheduling example
    print("\n6. Manual entanglement scheduling example...")
    scheduler2 = Scheduler(graph, flow)
    scheduler2.manual_schedule(
        prepare_time={node1: 0, node2: 1, node3: 2},
        measure_time={node0: 1, node1: 2, node2: 3},
        entangle_time={
            (node0, node1): 0,
            (node1, node2): 1,
            (node2, node3): 2,
        },
    )

    scheduler2.validate_schedule()  # Will raise ValueError if invalid
    print("   Manual schedule is valid: True")
    print(f"   Number of time slices: {scheduler2.num_slices()}")

    pattern_manual = qompile(graph, flow, scheduler=scheduler2)
    print(f"   Pattern has {len(pattern_manual.commands)} commands")
    print(f"   Maximum space usage: {pattern_manual.max_space} qubits")

print("\nDemo completed successfully!")
