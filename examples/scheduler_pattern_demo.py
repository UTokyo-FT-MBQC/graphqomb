"""Scheduler IR and Execution Tradeoffs
=======================================

This example compares scheduling strategies, shows how they change depth and
space usage, and illustrates how schedules become TICK-delimited pattern slices.

By default, ``ScheduleConfig`` uses fast greedy heuristics (``use_greedy=True``).
Pass ``use_greedy=False`` to solve for an optimal schedule with the CP-SAT solver,
which also unlocks CP-SAT-only options such as ``max_time``.
"""

# %%
from graphqomb.pattern import Pattern, print_pattern
from graphqomb.qompiler import qompile
from graphqomb.random_objects import generate_random_flow_graph
from graphqomb.scheduler import ScheduleConfig, Scheduler, Strategy

# %%
# Create sample graph state and flow
print("=== Scheduler-based Pattern Generation Demo ===\n")
print("1. Creating sample graph state...")
graph, xflow = generate_random_flow_graph(width=3, depth=3, edge_p=0.7)
print(f"   Graph has {len(graph.nodes)} nodes")
print(f"   Input nodes: {list(graph.input_node_indices.keys())}")
print(f"   Output nodes: {list(graph.output_node_indices.keys())}")
print(f"   Edges: {len(graph.edges)}")

# %%
# Demonstrate space-optimized scheduling (greedy heuristics by default)
print("\n2. Space-optimized Scheduling:")
print("=" * 40)

scheduler_space = Scheduler(graph, xflow)
space_config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
success_space = scheduler_space.solve_schedule(space_config)
pattern_space = None

if success_space:
    scheduler_space.validate_schedule()  # Will raise ValueError if invalid
    print("   Scheduling successful!")
    print(f"   Number of time slices: {scheduler_space.num_slices()}")

    prep_times = {k: v for k, v in scheduler_space.prepare_time.items() if v is not None}
    if prep_times:
        print(f"   Preparation times: {prep_times}")

    meas_times = {k: v for k, v in scheduler_space.measure_time.items() if v is not None}
    if meas_times:
        print(f"   Measurement times: {meas_times}")

    print("\n   Generated Pattern (Space-optimized):")
    pattern_space = qompile(graph, xflow, scheduler=scheduler_space)
    print(f"   Pattern has {len(pattern_space.commands)} commands")
    print(f"   Maximum space usage: {pattern_space.max_space} qubits")
    print(f"   Space usage over time: {pattern_space.space}")

    print("\n   Pattern commands:")
    print_pattern(pattern_space, lim=10)
else:
    print("   Failed to find solution for Space-optimized strategy")

# %%
# Demonstrate time-optimized scheduling (greedy heuristics by default)
print("\n3. Time-optimized Scheduling:")
print("=" * 40)

scheduler_time = Scheduler(graph, xflow)
time_config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
success_time = scheduler_time.solve_schedule(time_config)
pattern_time = None

if success_time:
    scheduler_time.validate_schedule()  # Will raise ValueError if invalid
    print("   Scheduling successful!")
    print(f"   Number of time slices: {scheduler_time.num_slices()}")

    prep_times = {k: v for k, v in scheduler_time.prepare_time.items() if v is not None}
    if prep_times:
        print(f"   Preparation times: {prep_times}")

    meas_times = {k: v for k, v in scheduler_time.measure_time.items() if v is not None}
    if meas_times:
        print(f"   Measurement times: {meas_times}")

    print("\n   Generated Pattern (Time-optimized):")
    pattern_time = qompile(graph, xflow, scheduler=scheduler_time)
    print(f"   Pattern has {len(pattern_time.commands)} commands")
    print(f"   Maximum space usage: {pattern_time.max_space} qubits")
    print(f"   Space usage over time: {pattern_time.space}")

    print("\n   Pattern commands:")
    print_pattern(pattern_time, lim=10)
else:
    print("   Failed to find solution for Time-optimized strategy")

# %%
# Demonstrate space-constrained time optimization using the CP-SAT solver
print("\n4. Space-constrained Time-optimized Scheduling (CP-SAT):")
print("=" * 60)

# Use ScheduleConfig for more detailed control
max_qubits = 5
constrained_config = ScheduleConfig(
    strategy=Strategy.MINIMIZE_TIME,
    max_qubit_count=max_qubits,
    max_time=20,  # Custom time limit (CP-SAT only)
    use_greedy=False,  # Use the CP-SAT solver for an optimal schedule
)

scheduler_constrained = Scheduler(graph, xflow)
success_constrained = scheduler_constrained.solve_schedule(constrained_config, timeout=15)
pattern_constrained = None

if success_constrained:
    scheduler_constrained.validate_schedule()  # Will raise ValueError if invalid
    print("   Scheduling successful!")
    print(f"   Number of time slices: {scheduler_constrained.num_slices()}")
    print(f"   Max qubits constraint: {max_qubits}")

    prep_times = {k: v for k, v in scheduler_constrained.prepare_time.items() if v is not None}
    if prep_times:
        print(f"   Preparation times: {prep_times}")

    meas_times = {k: v for k, v in scheduler_constrained.measure_time.items() if v is not None}
    if meas_times:
        print(f"   Measurement times: {meas_times}")

    print("\n   Generated Pattern (Space-constrained Time-optimized):")
    pattern_constrained = qompile(graph, xflow, scheduler=scheduler_constrained)
    print(f"   Pattern has {len(pattern_constrained.commands)} commands")
    print(f"   Maximum space usage: {pattern_constrained.max_space} qubits")
    print(f"   Space usage over time: {pattern_constrained.space}")

    if pattern_constrained.max_space <= max_qubits:
        print(f"   ✓ Space constraint satisfied: {pattern_constrained.max_space} <= {max_qubits}")
    else:
        print(f"   ⚠ Space constraint violated: {pattern_constrained.max_space} > {max_qubits}")

    print("\n   Pattern commands:")
    print_pattern(pattern_constrained, lim=10)
else:
    print(f"   Failed to find solution with {max_qubits} qubits constraint")

# %%
# Demonstrate custom max_time using the CP-SAT solver
print("\n5. Custom max_time Scheduling (CP-SAT):")
print("=" * 50)

custom_time_config = ScheduleConfig(
    strategy=Strategy.MINIMIZE_SPACE,
    max_time=15,  # Smaller time horizon for faster solving (CP-SAT only)
    use_greedy=False,  # Use the CP-SAT solver for an optimal schedule
)

scheduler_custom = Scheduler(graph, xflow)
success_custom = scheduler_custom.solve_schedule(custom_time_config, timeout=10)

if success_custom:
    scheduler_custom.validate_schedule()  # Will raise ValueError if invalid
    print("   Scheduling with custom max_time successful!")
    print(f"   Number of time slices: {scheduler_custom.num_slices()}")
    print(f"   Custom max_time: {custom_time_config.max_time}")

    pattern_custom = qompile(graph, xflow, scheduler=scheduler_custom)
    print(f"   Maximum space usage: {pattern_custom.max_space} qubits")
else:
    print("   Failed to find solution with custom max_time")

# %%
# Compare all strategies
print("\n6. Strategy Comparison:")
print("=" * 40)

all_results: list[tuple[str, Scheduler, Pattern]] = []
if success_space and pattern_space:
    all_results.append(("Space-optimized", scheduler_space, pattern_space))
if success_time and pattern_time:
    all_results.append(("Time-optimized", scheduler_time, pattern_time))
if success_constrained and pattern_constrained:
    all_results.append(("Space-constrained", scheduler_constrained, pattern_constrained))

min_results_needed = 2
if len(all_results) >= min_results_needed:
    for name, scheduler, pattern in all_results:
        print(
            f"   {name:18}: {scheduler.num_slices()} slices, "
            f"{pattern.max_space} max qubits, {len(pattern.commands)} commands"
        )

    print("\n   Analysis:")
    min_results_for_comparison = 2
    if len(all_results) >= min_results_for_comparison:
        min_slices = min(s.num_slices() for _, s, _ in all_results)
        min_qubits = min(p.max_space for _, _, p in all_results)

        for name, scheduler, pattern in all_results:
            notes: list[str] = []
            if scheduler.num_slices() == min_slices:
                notes.append("fastest execution")
            if pattern.max_space == min_qubits:
                notes.append("lowest qubit usage")
            if notes:
                print(f"   → {name}: {', '.join(notes)}")

print("\nDemo completed successfully!")
