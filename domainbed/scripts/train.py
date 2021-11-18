# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, FastDataLoader_no_shuffle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--sched', action='store_false', help='Use learning rate scheduler')
    parser.add_argument('--no_pca', action='store_true', help='Clustering without SVD + Truncation step')
    parser.add_argument('--clust_step', type=int, default=None, help='step to perform clustering')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters')

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))

    from .helpers import get_hparam
    hparams = get_hparam(hparams, args.hparams_seed) # To fix hparams for each hparams_seed, else comment out

    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    if "AdaClust" in args.algorithm:
        num_clusters = args.num_clusters or hparams["num_clusters"] * dataset.num_classes # Set number of clusters
        print("NUM CLUSTERS: ", num_clusters)

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
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(
                in_, int(len(in_) * args.uda_holdout_fraction), misc.seed_hash(args.trial_seed, env_i)
            )
        test_data_sep.append(in_)
        eval_loader_names += ["env{}_in".format(env_i)]
        test_data_sep.append(out)
        eval_loader_names += ["env{}_out".format(env_i)]
        if env_i not in args.test_envs:
            train_data_sep.append(in_)
            train_domain_labels.extend([env_i]*len(in_))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    from .helpers import *
    from .clustering import Faiss_Clustering
    from .preprocess import *

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
            batch_size=64,
            num_workers=8,
        )
        for idx in test_idx_split
    ]


    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # Set flag for AdaClust specific operations
    if "AdaClust" in args.algorithm:
        cluster = True
    else:
        cluster = False

    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = int(len(train_data) / hparams["batch_size"])
    print(f"Number of steps per epoch: {steps_per_epoch}")
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ 
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
        print(f"Cluster every {cluster_step} steps")
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



    last_results_keys = None
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

            evals = zip(eval_loader_names, eval_loaders)
            for i, (name, loader) in enumerate(evals):
                acc = misc.accuracy(algorithm, loader, None, device, test_centroids)
                results[name+'_acc'] = acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
