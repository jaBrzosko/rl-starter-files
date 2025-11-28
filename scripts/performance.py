from kg.masker import KGActionMasker, RecommendationMode
import torch
import torch_ac
import timeit
import cProfile
import pstats
import io
from functools import wraps
import time
from memory_profiler import profile as memory_profile
import tracemalloc
import argparse

UNSEEN = [0, 0, 0]
WALL = [2, 5, 0]
EMPTY = [1, 0, 0]
LAVA = [9, 0, 0]
GOAL = [8, 1, 0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_masker(mode, env):
    if mode == RecommendationMode.FUSEKI or mode == RecommendationMode.FUSEKI_OPTIMIZED:
        with open("kg/data/recommendations_fuseki_query.rq", "r") as f:
            query = f.read()
    else:
        if env == "door-key":
            with open("kg/data/recommendation_query_with_neg.rq", "r") as f:
                query = f.read()
        else:
            with open("kg/data/recommendation_query.rq", "r") as f:
                query = f.read()

    if env == "door-key":
        recommendation_file = "kg/data/experiments/minigrid_recommendations_doorkey_complex.ttl"
        problem_type = "door_key"
    else:
        recommendation_file = "kg/data/minigrid_recommendations.ttl"
        problem_type = "crossing"

    masker = KGActionMasker(
        ontology_file="kg/data/minigrid_ontology.ttl",
        recommendation_file=recommendation_file,
        query=query,
        domain_uri="http://example.org/minigrid#",
        env_problem_type=problem_type,
        action_size=7,
        recommendation_mode=mode,
    )
    return masker

def create_sample_env_state():
    map_size = 7
    test_map = [[UNSEEN for _ in range(map_size)] for _ in range(map_size)]
    test_map[2][6] = WALL
    test_map[2][5] = WALL
    test_map[2][4] = WALL
    test_map[2][3] = WALL
    test_map[3][3] = WALL
    test_map[4][3] = WALL
    test_map[5][3] = WALL
    test_map[6][3] = WALL

    test_map[3][6] = EMPTY
    test_map[4][6] = LAVA
    test_map[5][6] = LAVA
    test_map[6][6] = LAVA

    test_map[3][4] = EMPTY
    test_map[4][4] = EMPTY
    test_map[5][4] = EMPTY
    test_map[6][4] = EMPTY
    test_map[3][5] = EMPTY
    test_map[4][5] = GOAL
    test_map[5][5] = EMPTY
    test_map[6][5] = EMPTY
    return torch_ac.DictList({
        "image": [test_map],
    })

def timing_decorator(func):
    """Decorator to measure execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"  {func.__name__}: {elapsed:.6f} seconds")
        return result, elapsed
    return wrapper

def run_performance_test(iterations, mode, env):
    """
    Comprehensive performance evaluation with:
    - Average total time over multiple iterations
    - cProfile breakdown of internal method calls
    - Memory usage tracking
    """
    print("=" * 70)
    print("PERFORMANCE EVALUATION")
    print("=" * 70)
    
    # ==================== SETUP PHASE ====================
    print("\n[1] Setup Phase")
    print("-" * 70)
    
    setup_start = time.perf_counter()
    masker = create_masker(mode, env)
    test_env_state = create_sample_env_state()
    setup_end = time.perf_counter()
    print(f"  Setup time: {setup_end - setup_start:.6f} seconds")
    
    # ==================== TIMING WITH TIMEIT ====================
    print("\n[2] Average Execution Time (timeit)")
    print("-" * 70)
    
    def timed_execution():
        masker.get_action_mask(test_env_state, device=device)
    
    total_time = timeit.timeit(timed_execution, number=iterations)
    avg_time = total_time / iterations
    
    print(f"  Total iterations: {iterations}")
    print(f"  Total time: {total_time:.6f} seconds")
    print(f"  Average time per call: {avg_time:.6f} seconds")
    print(f"  Calls per second: {1/avg_time:.2f}")
    
    # ==================== DETAILED PROFILING WITH CPROFILE ====================
    print("\n[3] Detailed Method Breakdown (cProfile)")
    print("-" * 70)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple times for better statistics
    for _ in range(iterations):
        masker.get_action_mask(test_env_state, device=device)
    
    profiler.disable()
    
    # Create string buffer to capture profile output
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    
    # Print top 20 most time-consuming functions
    print("  Top 20 functions by cumulative time:")
    ps.print_stats(20)
    print(s.getvalue())
    
    # ==================== MEMORY PROFILING ====================
    print("\n[4] Memory Usage Analysis")
    print("-" * 70)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Take snapshot before
    snapshot_before = tracemalloc.take_snapshot()
    
    # Execute function
    for _ in range(10):
        action_mask = masker.get_action_mask(test_env_state, device=device)
    
    # Take snapshot after
    snapshot_after = tracemalloc.take_snapshot()
    
    # Calculate memory difference
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    total_memory = sum(stat.size_diff for stat in top_stats)
    print(f"  Total memory increase: {total_memory / 1024:.2f} KB ({total_memory / (1024*1024):.2f} MB)")
    
    print("\n  Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(f"    {stat}")
    
    tracemalloc.stop()
    
    # ==================== SINGLE EXECUTION BREAKDOWN ====================
    print("\n[5] Single Execution with Manual Timing")
    print("-" * 70)
    
    # Warm-up
    masker.get_action_mask(test_env_state, device=device)
    
    # Timed single execution
    start = time.perf_counter()
    action_mask = masker.get_action_mask(test_env_state, device=device)
    end = time.perf_counter()
    
    print(f"  Single execution time: {end - start:.6f} seconds")
    print(f"  Output shape: {action_mask.shape if hasattr(action_mask, 'shape') else 'N/A'}")
    print(f"  Output type: {type(action_mask)}")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Average execution time: {avg_time*1000:.3f} ms")
    print(f"  Throughput: {1/avg_time:.2f} calls/second")
    print(f"  Memory per 10 calls: {total_memory / (1024*10):.2f} KB")
    print("=" * 70)

def single_run(mode, env):
    masker = create_masker(mode, env)
    test_env_state = create_sample_env_state()

    start = time.perf_counter()
    action_mask = masker.get_action_mask(test_env_state, device=device)
    end = time.perf_counter()

    print(f"Single run execution time: {end - start:.6f} seconds")
    print(f"Output shape: {action_mask.shape if hasattr(action_mask, 'shape') else 'N/A'}")
    print(f"Output value: {action_mask}") 

def run_timing_test(iterations, mode, env):
    masker = create_masker(mode, env)
    test_env_state = create_sample_env_state()

    def timed_execution():
        masker.get_action_mask(test_env_state, device=device)
    
    total_time = timeit.timeit(timed_execution, number=iterations)
    avg_time = total_time / iterations
    
    print(f"  Total iterations: {iterations}")
    print(f"  Total time: {total_time:.6f} seconds")
    print(f"  Average time per call: {avg_time:.6f} seconds")
    print(f"  Calls per second: {1/avg_time:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["detailed", "single-run", "timing"] , default="detailed",)
    parser.add_argument("--iterations", type=int, default=10,
                        help="number of iterations for timing tests (default: 10)")

    parser.add_argument("--graph-mode", choices=["in-memory", "fuseki", "fuseki-optimized", "oxigraph"], default="in-memory",)

    parser.add_argument("--env", choices=["crossing", "door-key"] , default="crossing",
                        help="environment problem type (default: crossing)")

    args = parser.parse_args()

    if args.graph_mode == "in-memory":
        mode = RecommendationMode.IN_MEMORY
    elif args.graph_mode == "fuseki":
        mode = RecommendationMode.FUSEKI
    elif args.graph_mode == "fuseki-optimized":
        mode = RecommendationMode.FUSEKI_OPTIMIZED
    elif args.graph_mode == "oxigraph":
        mode = RecommendationMode.OXIGRAPH
    else:
        raise ValueError(f"Unknown graph mode: {args.graph_mode}")

    if args.mode == "detailed":
        run_performance_test(args.iterations, mode, args.env)
    elif args.mode == "single-run":
        single_run(mode, args.env)
    elif args.mode == "timing":
        run_timing_test(args.iterations, mode, args.env)
