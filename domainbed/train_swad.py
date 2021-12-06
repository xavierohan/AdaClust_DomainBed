import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, FastDataLoader_no_shuffle
from domainbed import swad as swad_module
from domainbed import datasets

from .scripts.helpers import *
from .scripts.clustering import Faiss_Clustering
from .scripts.preprocess import *

def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(dataset, test_envs, args, hparams, n_steps, checkpoint_freq, logger, target_env=None):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = vars(datasets)[args.dataset](args.data_dir,
            test_envs, hparams)

    if "AdaClust" in args.algorithm:
        num_clusters = args.num_clusters or hparams["num_clusters"] * dataset.num_classes # Set number of clusters
        logger.info(f"NUM CLUSTERS: {num_clusters}")

    test_data_sep = []
    train_data_sep = []
    eval_loader_names = []
    in_splits = []
    out_splits = []
    train_domain_labels = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(
            env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i)
        )
        if env_i in test_envs:
            uda, in_ = misc.split_dataset(
                in_, int(len(in_) * 0), misc.seed_hash(args.trial_seed, env_i)
            )
        test_data_sep.append(in_)
        eval_loader_names += ["env{}_in".format(env_i)]
        test_data_sep.append(out)
        eval_loader_names += ["env{}_out".format(env_i)]
        if env_i not in test_envs:
            train_data_sep.append(in_)
            train_domain_labels.extend([env_i]*len(in_))


    train_data = MyDataloader(train_data_sep) # Concat train data
    test_data = MyDataloader(test_data_sep) # Concat test data
    len_train_data = len(train_data)
    len_test_data = len(test_data)

    # Loaders to perform clustering
    if "AdaClust" in args.algorithm:
        train_loader = FastDataLoader_no_shuffle(
            dataset=train_data,
            batch_size=128,
            num_workers=8,
        )
        test_loader = FastDataLoader_no_shuffle(
            dataset=test_data,
            batch_size=128,
            num_workers=8,
        )

    # DomainBed dataloaders
    train_idx_split = get_data_split_idx(train_data_sep)
    train_loaders = [
        InfiniteDataLoader(
            dataset=torch.utils.data.Subset(train_data, idx),
            weights=None,
            batch_size=hparams["batch_size"],
            num_workers=8,
        )
        for idx in train_idx_split
    ]

    train_minibatches_iterator = zip(*train_loaders)

    test_idx_split = get_data_split_idx(test_data_sep)
    eval_loaders = [
        FastDataLoader(
            dataset=torch.utils.data.Subset(test_data, idx),
            batch_size=128,
            num_workers=8,
        )
        for idx in test_idx_split
    ]

    eval_meta = list(zip(eval_loader_names, eval_loaders, [None]*len(test_data_sep)))
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        len(dataset),
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=None,
    )

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    algorithm.to(device)

    # Set flag for AdaClust specific operations
    if "AdaClust" in args.algorithm:
        cluster = True
    else:
        cluster = False

    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = int(len(train_data) / hparams["batch_size"])
    logger.info(f"Number of steps per epoch: {steps_per_epoch}")
    epochs = int(n_steps / steps_per_epoch)

    if args.sched:
        algorithm.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(algorithm.optimizer, T_max=n_steps) # Initialize scheduler

    # Set clustering schedule
    if cluster:
        cluster_step = args.clust_step
        if args.clust_step is None:
            cluster_step = steps_per_epoch * hparams["clust_epoch"] # Cluster every hparams["clust_epoch"] epochs
        cluster_step = [(x*cluster_step) for x in range(n_steps) if (x*cluster_step)<= n_steps]
        if hparams["clust_epoch"]==0: # cluster every 2**n epochs (0, 1, 2, 4, 8, 16, ...)
            cluster_step = [((2 ** x)*steps_per_epoch) for x in range(epochs) if (2**x)<= epochs] # store the steps at which clustering take place
        logger.info(f"Cluster every {cluster_step} steps")
    else:
        cluster_step = [-1] # dummy value

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    
    def return_centroids(algorithm, step):

        with torch.no_grad():
            # Get features and labels
            train_features, train_labels = get_features(train_loader, algorithm, len_train_data, 128, device)
            test_features, test_labels = get_features(test_loader, algorithm, len_test_data, 128, device)
            train_labels = torch.Tensor(train_labels).type(torch.LongTensor)
            test_labels = torch.Tensor(test_labels).type(torch.LongTensor)

        # Clustering on Train Data
        if args.no_pca:
            train_features2 = train_features # if no PCA
        else:
            train_features_pca = np.asarray(train_features)
            pca = PCA(hparams["pca_dim"])
            exp_var = pca.fit(train_features, hparams["offset"])
            train_features2 = pca.apply(torch.from_numpy(train_features_pca)).detach().numpy()
            row_sums = np.linalg.norm(train_features2, axis=1)
            train_features2 = train_features2 / row_sums[:, np.newaxis]

        clustering = Faiss_Clustering(train_features2.copy(order="C"), num_clusters)
        clustering.fit()
        cluster_labels_train = get_cluster_labels(clustering, train_features2)
        images_lists = get_images_list(num_clusters, len_train_data, cluster_labels_train)
        train_centroids = torch.empty((len_train_data, train_features.shape[1]))

        # Get the centroid of the images that share the same cluster in PCA space
        for i, indx in enumerate(images_lists):
            if len(indx) > 0:
                train_centroids[indx] = torch.Tensor(train_features[indx].mean(axis=0))

        # Clustering on Test Data
        if args.no_pca:
            test_features2 = test_features
        else:
            test_features_pca = np.asarray(test_features)
            test_features2 = pca.apply(torch.from_numpy(test_features_pca)).detach().numpy()
            row_sums = np.linalg.norm(test_features2, axis=1)
            test_features2 = test_features2 / row_sums[:, np.newaxis]

        cluster_labels_test = get_cluster_labels(clustering, test_features2)
        images_lists = get_images_list(num_clusters, len_test_data, cluster_labels_test)
        test_centroids = torch.empty((len_test_data, test_features.shape[1]))

        for i, indx in enumerate(images_lists):
            if len(indx) > 0:
                test_centroids[indx] = torch.Tensor(test_features[indx].mean(axis=0))

        return train_centroids, test_centroids, exp_var

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, hparams["swad"])
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"
    start_step=0
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        epoch = step / steps_per_epoch

        if step == 0 or (step in cluster_step):
            if cluster:
                train_centroids, test_centroids, exp_var = return_centroids(algorithm, step)
            else:
                test_centroids = None

        if cluster:
            minibatches_device = [
                (x.to(device), train_centroids[idx].to(device), y.to(device))
                for ((x, y), idx) in next(train_minibatches_iterator)
            ]
        else:
            minibatches_device = [
                (x.to(device), y.to(device)) for ((x, y), idx) in next(train_minibatches_iterator)
            ]

        step_vals = algorithm.update(minibatches_device)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            if cluster:
                results["exp_var"] = str(exp_var)

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)


            # results = (epochs, loss, step, step_time)
            accuracies, summaries = evaluator.evaluate(algorithm, test_centroids)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["loss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset


    # find best
    logger.info("---")
    records = Q(records)
    oracle_best = records.argmax("test_out")["test_in"]
    iid_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    if hparams.indomain_test:
        # if test set exist, use test set for indomain results
        in_key = "train_inTE"
    else:
        in_key = "train_out"

    iid_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    ret = {
        "oracle": oracle_best,
        "iid": iid_best,
        "last": last,
        "last (inD)": last_indomain,
        "iid (inD)": iid_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm.module, test_centroids)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records