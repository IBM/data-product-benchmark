# Data Product Benchmark

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <a href="https://opensource.org/license/mit">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://huggingface.co/datasets/ibm-research/data-product-benchmark">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg" alt="HuggingFace">
  </a>
<a href="https://arxiv.org/abs/2510.21737">
    <img src="https://img.shields.io/badge/Paper-arXiv-red.svg" alt="arxiv">
  </a>
  <a href="https://www.python.org/downloads/release/python-310/">
    <img src="https://img.shields.io/badge/Python-3.10+-teal.svg" alt="Python">
  </a>
</div>

The source code for data product benchmark creation and the baselines associated with the papre titled [`From Factoid Questions to Data Product Requests: Benchmarking Data Product Discovery over Tables and Text`](https://arxiv.org/abs/2510.21737).

## Structure of the repo
```
baselines/
├── data/ # the directory for storing the evaluation results
├── scripts/  # scripts to run the baseline
└── src/  # baseline and evaluation code

benchmark_framework/
├── scripts/  # scripts for benchmark creation
└── src/      # code for benchmark creation
```

## Running the baselines

#### Setting the environment
Create a python environment and install requirements. Reproducibility has been checked for python version 3.12.
```cmd
cd baselines
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

#### Running baseline experiments
Baseline experiments can be directly run using the following script:
You may also need to first make the sh file executable, e.g. `chmod +x scripts/run_baseline.sh`

```
cd baselines
./scripts/run_baseline.sh
```

Running the baseline script will proceed with producing embeddings and running baseline retrieval methods for a single database at a time. The choice of which data will be used to produce results, as well as the choice of embedding model, can be changed within the `run_baselines.sh` script -- see commented lines in the file for specific arg choices.

The script will download the corresponding datasets directly from the HuggingFace repo [ibm-research/data-product-benchmark](https://huggingface.co/datasets/ibm-research/data-product-benchmark). 

After running the baseline script, results will be output to files such as `baselines/data/ConfFinQA/ConfFinQA_test_results_eval_granite.json`

> **Note:** HybridQA is the largest of the datasets, and running this baseline may be slow (running locally with no GPU, producing the entire collection of text embeddings may take about an hour). Embedding speed will be much faster if you are running on a machine with GPU support. Producing baseline results for TATQA and ConfFinQA is expected to finish within a few minutes.

## How to run the benchmark creation 

> **Note:**  Systems that plan to use benchmark to evaluate them, can use the benchmark directly from the data as shown in the running baseline section above, and no need to re-run the benchmark creation process. Benchmark creation is documented here for reproducibility.

#### Downloading the existing QA Benchmarks
The benchmark uses data from the following existing repositories and you will need to download those repositories first. 
- [WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks.git)
- [HybridQA](https://github.com/wenhuchen/HybridQA.git)
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA.git)
- [ConvFinQA](https://github.com/czyssrs/ConvFinQA.git)

To make the process easier, we have added them as git-submodules. Use the following command to clone all of them in a single command.

```commandline
git submodule update --init --recursive
```

The ConvFinQA dataset is in a compressed zip file. Unzip it using the following command. 
```commandline
unzip benchmark_framework/data/raw/ConvFinQA/data.zip -d benchmark_framework/data/raw/ConvFinQA
```

#### Corpus preparation

The following command will run the corpus preparation for HybridQA, TATQA, and ConvFinQA datasets. It reads the raw data from the original Git repos files and create tables and text corpora in a common format that will be used by the next phases of the pipeline.

```commandline
sh benchmark_framework/scripts/0_prepare.sh
```

This step will create the following files.

```
benchmark_framework/data/output
├── ConvFinQA
│   ├── dev
│   │   └── ConvFinQA_dev_corpus.json
│   ├── test
│   │   └── ConvFinQA_test_corpus.json
│   └── train
│       └── ConvFinQA_train_corpus.json
├── HybridQA
│   ├── dev
│   │   └── HybridQA_dev_corpus.json
│   ├── test
│   │   └── HybridQA_test_corpus.json
│   └── train
│       └── HybridQA_train_corpus.json
└── TATQA
    ├── dev
    │   └── TATQA_dev_corpus.json
    ├── test
    │   └── TATQA_test_corpus.json
    └── train
        └── TATQA_train_corpus.json
```

#### Topic clustering of questions (grouped by the tables)

The goal of this step is to cluster the questions (which are already grouped in to tables) by identifying tables that share common analytical themes. Questions from the same table are concatenated to form a document corresponds that table and those documents are clustered by topic using `BERTopic`. Once the clustering is done, cluster quality metrics are calculated on the resulting clusters.


```commandline
sh benchmark_framework/scripts/1_cluster.sh
```

It will generate the following files. 
- `*-clusters.json` files contain the clusters for each dataset/split with the clusters that contain several tables grouped by the topic. 
- `*-clusters_summary.json` files contain cluster quality metrics calculated at each individual cluster level and globably. Global metrics include  silhouette_score, calinski_harabasz_index, davies_bouldin_index and individual cluster metrics include silhouette, intra_cluster_mse, inter_cluster_dist, and db_component. Statistics such as how many tables, questions in the cluster also recorded in tis file. 
```
benchmark_framework/data/output
├── ConvFinQA
│   ├── dev
│   │   ├── ConvFinQA_dev_clusters.json
│   │   └── ConvFinQA_dev_clusters_summary.json
│   ├── test
│   │   ├── ConvFinQA_test_clusters.json
│   │   └── ConvFinQA_test_clusters_summary.json
│   └── train
│       ├── ConvFinQA_train_clusters.json
│       └── ConvFinQA_train_clusters_summary.json
├── HybridQA
│   ├── dev
│   │   ├── HybridQA_dev_clusters.json
│   │   └── HybridQA_dev_clusters_summary.json
│   ├── test
│   │   ├── HybridQA_test_clusters.json
│   │   └── HybridQA_test_clusters_summary.json
│   └── train
│       ├── HybridQA_train_clusters.json
│       └── HybridQA_train_clusters_summary.json
└── TATQA
    ├── dev
    │   ├── TATQA_dev_clusters.json
    │   └── TATQA_dev_clusters_summary.json
    ├── test
    │   ├── TATQA_test_clusters.json
    │   └── TATQA_test_clusters_summary.json
    └── train
        ├── TATQA_train_clusters.json
        └── TATQA_train_clusters_summary.json
```

#### Data Product Filtering

Even though topic clusters can provide some grouping of table that share a common topic, they might not be in the same level of granualarity of a data product Real-world data products need tighter semantic coherence and a manageable scope to be practically useful. This step refines raw clusters by incorporating table schema information and controlling granularity. This step will also ensure that the size of the data products are  manageable.

```commandline
sh benchmark_framework/scripts/2_filtering.sh
```

It will generate the following files with the refined clusters.
```
benchmark_framework/data/output
├── ConvFinQA
│   ├── dev
│   │   └── ConvFinQA_dev_filtered_clusters.json
│   ├── test
│   │   └── ConvFinQA_test_filtered_clusters.json
│   └── train
│       └── ConvFinQA_train_filtered_clusters.json
├── HybridQA
│   ├── dev
│   │   └── HybridQA_dev_filtered_clusters.json
│   ├── test
│   │   └── HybridQA_test_filtered_clusters.json
│   └── train
│       └── HybridQA_train_filtered_clusters.json
└── TATQA
    ├── dev
    │   └── TATQA_dev_filtered_clusters.json
    ├── test
    │   └── TATQA_test_filtered_clusters.json
    └── train
        └── TATQA_train_filtered_clusters.json
```

#### Data Product Request Generation

> **Note:**  From this step forward, the framework will be using LLM calls for various tasks. It uses [DSPy](https://dspy.ai/) for LLM calls which based on [LiteLLM](https://www.litellm.ai/). LiteLLM supports a large number of LLM provides including commercial providers as well as Ollama, vLLM, etc. as decribed [here](https://docs.litellm.ai/docs/providers). Please configure the [llm_provider.py](benchmark_framework/src/llm_provider.py) to based on the LLM provider you have access to.

In this step, we generate Data Product Requests for each of the filtered clusters. 

```commandline
sh benchmark_framework/scripts/3_generation.sh
```

#### Data Product Request Validation

```commandline
sh benchmark_framework/scripts/4_eval.sh
```


#### Data Product Ground Truth Refinement

```commandline
sh benchmark_framework/scripts/5_refinement.sh
```

## Citation

If you use this work in your research, please cite the [following paper](https://arxiv.org/abs/2510.21737):

```bibtex
@misc{zhang2025dpr,
      title={From Factoid Questions to Data Product Requests: Benchmarking Data Product Discovery over Tables and Text}, 
      author={Liangliang Zhang and Nandana Mihindukulasooriya and Niharika S. D'Souza and Sola Shirai and Sarthak Dash and Yao Ma and Horst Samulowitz},
      year={2025},
      url={https://arxiv.org/abs/2510.21737}, 
}
```

## License

This project is released under the MIT license. See [LICENSE](LICENSE) for details.


## IBM Public Repository Disclosure

All content in this repository including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.
