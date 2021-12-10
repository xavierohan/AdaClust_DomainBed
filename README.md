# Adaptive Methods for Aggregated Domain Generalization (AdaClust)

Official Pytorch Implementation of [Adaptive Methods for Aggregated Domain Generalization](https://arxiv.org/abs/2112.04766)

Xavier Thomas, Dhruv Mahajan, Alex Pentland, Abhimanyu Dubey

## AdaClust related hyperparameters 

* num_clusters: Number of clusters

* pca_dim: Required Feature space dimension after the SVD + Truncation  step

* offset: First Principal Eigenvector in the SVD + Truncation Step

* clust_epoch: Defines the clustering schedule
  * clust_epoch = 0: cluster every 0, 1, 2, 4, 8, 16, ... epochs
  * clust_epoch = k, k>0: cluster every k epochs


## Quick start

Download the datasets:

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

Train a model:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm AdaClust\
       --dataset PACS\
       --test_env 3
```
More details at: https://github.com/facebookresearch/DomainBed

Run SWAD:
```sh
python3 train_all.py exp_name --dataset PACS --algorithm AdaClust --data_dir /my/datasets/path
```
More details at: https://github.com/khanrc/swad


Launch a sweep:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
```

Here, `MyLauncher` is your cluster's command launcher, as implemented in `command_launchers.py`. At the time of writing, the entire sweep trains tens of thousands of models (all algorithms x all datasets x 3 independent trials x 20 random hyper-parameter choices). You can pass arguments to make the sweep smaller:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher\
       --algorithms ERM AdaClust\
       --datasets PACS VLCS\
       --n_hparams 5\
       --n_trials 1
```
## Available model selection criteria

[Model selection criteria](domainbed/model_selection.py) differ in what data is used to choose the best hyper-parameters for a given model:

* `IIDAccuracySelectionMethod`: A random subset from the data of the training domains.
* `LeaveOneOutSelectionMethod`: A random subset from the data of a held-out (not training, not testing) domain.
* `OracleSelectionMethod`: A random subset from the data of the test domain.

After all jobs have either succeeded or failed, you can delete the data from failed jobs with ``python -m domainbed.scripts.sweep delete_incomplete`` and then re-launch them by running ``python -m domainbed.scripts.sweep launch`` again. Specify the same command-line arguments in all calls to `sweep` as you did the first time; this is how the sweep script knows which jobs were launched originally.

To view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path
````

## Running unit tests

DomainBed includes some unit tests and end-to-end tests. While not exhaustive, but they are a good sanity-check. To run the tests:

```sh
python -m unittest discover
```

By default, this only runs tests which don't depend on a dataset directory. To run those tests as well:

```sh
DATA_DIR=/my/datasets/path python -m unittest discover
```

## Citation

```
@misc{thomas2021adaptive,
      title={Adaptive Methods for Aggregated Domain Generalization}, 
      author={Xavier Thomas and Dhruv Mahajan and Alex Pentland and Abhimanyu Dubey},
      year={2021},
      eprint={2112.04766},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This source code is released under the MIT license, included [here](LICENSE).
