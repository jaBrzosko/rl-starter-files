class Args:
    # General parameters
    env = "default_env"  # name of the environment to train on
    model = None  # name of the model
    experiment_name = None
    seed = 1  # Main seed for randomness
    use_kg = False  # whether to use knowledge graph
    kg_recommendation_file = None  # path to KG recommendation file
    kg_query_file = None  # path to KG SPARQL query file
    kg_problem_type = None

    log_interval = 1  # number of updates between two logs
    save_interval = 10  # number of updates between two saves
    procs = 16  # number of processes
    frames = 10**7  # number of frames of training

    # Parameters for main algorithm
    epochs = 4  # number of epochs for PPO
    batch_size = 256  # batch size for PPO

    frames_per_proc = None  # number of frames per process before update

    discount = 0.99  # discount factor
    lr = 0.001  # learning rate
    gae_lambda = 0.95  # lambda coefficient in GAE formula
    entropy_coef = 0.01  # entropy term coefficient
    value_loss_coef = 0.5  # value loss term coefficient
    max_grad_norm = 0.5  # maximum norm of gradient
    optim_eps = 1e-8  # Adam and RMSprop optimizer epsilon
    optim_alpha = 0.99  # RMSprop optimizer alpha
    clip_eps = 0.2  # clipping epsilon for PPO

    # Masking parameters
    episode_cutoff = None  # number of episodes after which to stop masking
    value_cutoff = None  # value cutoff after which to stop masking
    masking_probability = 1.0  # probability of applying masking

def read_experiment(experiment_name):
    
    empty_env_name = "MiniGrid-Empty-8x8-v0"
    lava_gap_env_name = "MiniGrid-LavaGapS7-v0"
    lava_crossing_env_name = "MiniGrid-LavaCrossingS9N1-v0"
    door_key_env_name = "MiniGrid-DoorKey-6x6-v0"
    query_path = "kg/data/recommendation_query.rq"

    if experiment_name == "empty_base":
        args = Args()
        args.env = empty_env_name
        args.frames = 60_000
    elif experiment_name == "empty_kg_base":
        args = Args()
        args.env = empty_env_name
        args.frames = 50_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_empty_simple.ttl"
        args.kg_problem_type = "empty"
    elif experiment_name == "empty_kg_complex":
        args = Args()
        args.env = empty_env_name
        args.frames = 50_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_empty_complex.ttl"
        args.kg_problem_type = "empty"
    elif experiment_name == "lava_gap_base":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 300_000
    elif experiment_name == "lava_gap_kg_base":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 150_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
    elif experiment_name == "lava_gap_kg_complex":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 150_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_complex.ttl"
        args.kg_problem_type = "crossing"
    # elif experiment_name == "lava_crossing_base":
    #     args = Args()
    #     args.env = lava_crossing_env_name
    #     args.frames = 2_000_000
    elif experiment_name == "door_key_base":
        args = Args()
        args.env = door_key_env_name    
        args.frames = 300_000
    elif experiment_name == "door_key_kg_base":
        args = Args()
        args.env = door_key_env_name
        args.frames = 200_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_doorkey_simple.ttl"
        args.kg_problem_type = "door_key"
    elif experiment_name == "door_key_kg_complex":
        args = Args()
        args.env = door_key_env_name
        args.frames = 100_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_doorkey_complex.ttl"
        args.kg_problem_type = "door_key"
        args.kg_query_file = "kg/data/recommendation_query_with_neg.rq"
    elif experiment_name == "lava_gap_kg_base_episode_cutoff":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 150_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
        args.episode_cutoff = 20
    elif experiment_name == "lava_gap_kg_base_value_cutoff":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 300_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
        args.value_cutoff = 0.4
    elif experiment_name == "lava_gap_kg_base_prob_095":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 200_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
        args.masking_probability = 0.95
    elif experiment_name == "lava_gap_kg_base_prob_098":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 200_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
        args.masking_probability = 0.98
    elif experiment_name == "lava_gap_kg_base_prob_099":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 200_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
        args.masking_probability = 0.99
    elif experiment_name == "lava_gap_kg_base_prob_0995":
        args = Args()
        args.env = lava_gap_env_name
        args.frames = 200_000
        args.use_kg = True
        args.kg_recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        args.kg_problem_type = "crossing"
        args.masking_probability = 0.995
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")

    if args.use_kg and args.kg_query_file is None:
        args.kg_query_file = query_path

    args.model = f"{experiment_name}_model"
    args.experiment_name = experiment_name
    return args
