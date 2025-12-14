import os
import json
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device
import minigrid


def evaluate_single_model(env_name, model_dir, episodes=100, seed=0, procs=16):
    utils.seed(seed)
    
    envs = []
    for i in range(procs):
        env = utils.make_env(env_name, seed + 10000 * i, debug=False)
        envs.append(env)
    env = ParallelEnv(envs)
    
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                       argmax=False, num_envs=procs,
                       use_memory=False, use_text=False)
    
    logs = {"num_frames_per_episode": [], "return_per_episode": []}
    
    start_time = time.time()
    
    obss = env.reset()
    
    log_done_counter = 0
    log_episode_return = torch.zeros(procs, device=device)
    log_episode_num_frames = torch.zeros(procs, device=device)
    
    while log_done_counter < episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)
        
        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(procs, device=device)
        
        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
        
        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask
    
    end_time = time.time()
    
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    
    return logs, fps, duration


def evaluate_experiment(experiment_name, env_name, runs, episodes=100, base_seed=0, procs=16):
    experiment_dir = utils.get_experiment_dir(experiment_name)
    
    arg_file = os.path.join(experiment_dir, "args.json")
    if not os.path.exists(arg_file):
        print(f"Warning: {arg_file} does not exist. Skipping experiment {experiment_name}")
        return None
    
    with open(arg_file, "r") as f:
        experiment_args = json.load(f)
    
    model_name = experiment_args.get("model", experiment_name)
    
    all_returns = []
    all_frames = []
    run_results = []
    
    print(f"\n{'='*80}")
    print(f"Evaluating Experiment: {experiment_name}")
    print(f"{'='*80}")
    
    for run in range(runs):
        model_dir = utils.get_model_dir_for_experiment(experiment_name, model_name, run)
        
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist. Skipping run {run}")
            continue
        
        print(f"\nEvaluating run {run}...")
        
        seed = base_seed + run
        logs, fps, duration = evaluate_single_model(env_name, model_dir, episodes, seed, procs)
        
        return_stats = utils.synthesize(logs["return_per_episode"])
        frames_stats = utils.synthesize(logs["num_frames_per_episode"])
        
        run_result = {
            "run": run,
            "return_mean": return_stats["mean"],
            "return_std": return_stats["std"],
            "return_min": return_stats["min"],
            "return_max": return_stats["max"],
            "frames_mean": frames_stats["mean"],
            "frames_std": frames_stats["std"],
            "frames_min": frames_stats["min"],
            "frames_max": frames_stats["max"],
            "fps": fps,
            "duration": duration
        }
        
        run_results.append(run_result)
        
        all_returns.extend(logs["return_per_episode"])
        all_frames.extend(logs["num_frames_per_episode"])
        
        print(f"  Run {run}: Return μ={return_stats['mean']:.2f} σ={return_stats['std']:.2f} "
              f"[{return_stats['min']:.2f}, {return_stats['max']:.2f}]")
    
    if not run_results:
        print(f"No valid runs found for experiment {experiment_name}")
        return None
    
    aggregate_stats = {
        "experiment_name": experiment_name,
        "num_runs": len(run_results),
        "return_stats": utils.synthesize(all_returns),
        "frames_stats": utils.synthesize(all_frames),
        "run_results": run_results
    }
    
    return aggregate_stats


def run_batch_evaluation(experiment_names, env_name, runs=5, episodes=100, base_seed=0, procs=16):
    print(f"Device: {device}\n")
    print(f"Starting batch evaluation of {len(experiment_names)} experiments")
    print(f"Environment: {env_name}")
    print(f"Runs per experiment: {runs}")
    print(f"Episodes per run: {episodes}")
    
    all_results = []
    
    for exp_name in experiment_names:
        result = evaluate_experiment(exp_name, env_name, runs, episodes, base_seed, procs)
        
        if result:
            all_results.append(result)
    
    print(f"\n{'='*80}")
    print("SUMMARY: Experiment Comparison")
    print(f"{'='*80}\n")
    
    print(f"{'Experiment':<30} {'Runs':<6} {'Return (μ±σ)':<20} {'Min/Max':<20}")
    print(f"{'-'*80}")
    
    for result in all_results:
        exp_name = result["experiment_name"]
        num_runs = result["num_runs"]
        ret_mean = result["return_stats"]["mean"]
        ret_std = result["return_stats"]["std"]
        ret_min = result["return_stats"]["min"]
        ret_max = result["return_stats"]["max"]
        
        print(f"{exp_name:<30} {num_runs:<6} {ret_mean:>6.2f}±{ret_std:<5.2f}      "
              f"[{ret_min:>6.2f}, {ret_max:>6.2f}]")
    
    output_file = "batch_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")
    
    return all_results


if __name__ == "__main__":
    EXPERIMENT_NAMES = [
        "lava_gap_base",
        "lava_gap_kg_base",
        "lava_gap_kg_complex",
        "lava_gap_kg_base_episode_cutoff_ec40",
        "lava_gap_kg_base_value_cutoff",
         "lava_gap_kg_base_prob_095",
         "lava_gap_kg_base_prob_098",
         "lava_gap_kg_base_prob_099"
    ]
    
    ENV_NAME = "MiniGrid-LavaGapS7-v0"
    RUNS_PER_EXPERIMENT = 5
    EPISODES_PER_RUN = 500
    BASE_SEED = 0
    NUM_PROCS = 16
    
    USE_ARGMAX = True
    USE_MEMORY = False
    USE_TEXT = False
    
    results = run_batch_evaluation(
        experiment_names=EXPERIMENT_NAMES,
        env_name=ENV_NAME,
        runs=RUNS_PER_EXPERIMENT,
        episodes=EPISODES_PER_RUN,
        base_seed=BASE_SEED,
        procs=NUM_PROCS
    )
