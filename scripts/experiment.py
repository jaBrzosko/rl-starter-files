import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
import os
import json
from kg.masker import KGActionMasker, RecommendationMode
from scripts.data.experiment_store import read_experiment
from utils import device
from model import ACModel
import minigrid

ONTOLOGY_FILE = "kg/data/minigrid_ontology.ttl"

def evaluate_experiment(args, seed, model_dir):
    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, use_memory=False, use_text=False)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])

    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))


    if args.use_kg:
        with open(args.kg_query_file, "r") as f:
            query = f.read()

        masker = KGActionMasker(
            ontology_file=ONTOLOGY_FILE,
            recommendation_file=args.kg_recommendation_file,
            query=query,
            domain_uri="http://example.org/minigrid#",
            env_problem_type=args.kg_problem_type,
            action_size=envs[0].action_space.n,
            recommendation_mode=RecommendationMode.OXIGRAPH,
        )

        mask_actions = masker.get_action_mask
    else:
        mask_actions = None

    # Load algo
    recurrence = 1

    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size,
                            preprocess_obss=preprocess_obss,
                            mask_actions=mask_actions)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if (args.save_interval > 0 and update % args.save_interval == 0) or num_frames >= args.frames:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

def run_experiment_batch(experiments_args, runs):
    experiment_dir = utils.get_experiment_dir(experiments_args.experiment_name)
    txt_logger = utils.get_txt_logger(experiment_dir)

    # Check if arg files exist and if is the same experiment
    arg_file = os.path.join(experiment_dir, "args.json")
    if not os.path.exists(arg_file):
        txt_logger.info(f"Saving experiment args to {arg_file}")
        utils.create_folders_if_necessary(arg_file)
        content = json.dumps(vars(experiments_args), indent=4)
        with open(arg_file, "w") as f:
            f.write(content)
    else:
        with open(arg_file, "r") as f:
            saved_args = json.load(f)
        current_args = vars(experiments_args)
        if saved_args != current_args:
            txt_logger.error(f"Experiment args in {arg_file} do not match the current args.")
            txt_logger.error(f"Saved args: {saved_args}")
            txt_logger.error(f"Current args: {current_args}")
            return
    
    for run in range(runs):
        model_dir = utils.get_model_dir_for_experiment(experiments_args.experiment_name, experiments_args.model, run)
        seed = experiments_args.seed + run
        txt_logger.info(f"\nStarting run {run} for experiment {experiments_args.experiment_name} in {model_dir} with seed {seed}\n")
        evaluate_experiment(experiments_args, seed, model_dir)
        txt_logger.info(f"\nFinished run {run} for experiment {experiments_args.experiment_name}")


parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str, nargs='+', required=True,
                    help="Names of the experiment scripts (in scripts/experiment).")

parser.add_argument("--runs", type=int, default=1,)

if __name__ == "__main__":
    args_cli = parser.parse_args()

    experiments_args = []
    for exp_name in args_cli.experiment:
        exp_args = read_experiment(exp_name)
        experiments_args.append(exp_args)

    for single_exp_args in experiments_args:
        run_experiment_batch(single_exp_args, args_cli.runs)
