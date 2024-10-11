# EFFECTIVE AND EFFICIENT ADVERSARIAL DETECTION FOR VISION-LANGUAGE MODELS VIA A SINGLE VECTOR

[![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)]()
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature.
To facilitate research on this critical safety problem, we first construct a new la**R**ge-scale **A**dervsarial images dataset with **D**iverse h**A**rmful **R**esponses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses.
With the new RADAR dataset, we further develop a novel and effective  i**N**-time **E**mbedding-based **A**dve**RS**arial **I**mage **DE**tection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call \textit{the attacking direction}, to achieve the detection of adversarial images against benign ones in the input. 
Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency,
and cross-model transferrability of our proposed method. Our code is included in the supplementary file and will be made publicly available.

`Contact tangjingkun@stu.scu.edu.cn anytime. 😊`

### RADAR Construction Pipeline:

![img.png](RADAR_pipeline.png)

### Diversity of RADAR:

![img.png](diversity.png)

Comparison of datasets for adversarially attacking VLMs. "-" means not reported.

### RADAR exemplar sample:

![img.png](RADAR_SAMPLE.png)

In RADAR, each sample consists of an adversarial sample and a benign sample, with each containing
a query, an adversarial/benign image and VLMs’ response. 

### NEARSIDE method:

![img.png](NEARSIDE.png)

Our method learns the attacking direction on a
set of tuples (benign input, adversarial input), and then classifies a test input as benign or adversarial
according to the projection between the input’s embedding and the attacking direction. If the
projection is larger than a threshold, it is classified as an adversarial input, and otherwise as benign.

### Metrics:

![img.png](NEARSIDE and RADAR.png)

Results of JailGuard v.s. NEARSIDE on RADAR test sets (best highlighted in bold).

![img.png](ceoss-model.png)

We utilize the attacking direction extracted from the source VLM (svlm) to detect adversarial input
for the target VLM (tvlm), denoted as svlm → tvlm. We calculate the difference by subtracting the result of (svlm → tvlm) from single model.
where −δ denotes the result is decreased while +δ denotes the opposite.

![img.png](efficiency.png)

Throughput of four different detection methods. The number is the average examples can be detected per second (item/s).

## Table of Repo Content
- [Repo structure](#repo-structure)
- [Installation](#installation)
- [RADAR Dataset Construction](#radar-dataset-construction)
- [NEARSIDE Adversarial Detection](#nearside-adversarial-detection)
- [Cite](#cite)

## Repo structure

```
RADAR-NEARSIDE
├─ JailGuard
│  └─ JailGuard
│     ├─ time_consume.py
│     ├─ jailguard.sh
│     ├─ utils
│     │  ├─ similarity.py
│     │  ├─ config.cfg
│     │  ├─ prompt
│     │  │  ├─ 1
│     │  │  └─ 2
│     │  ├─ utils.py
│     │  ├─ baseline_utils.py
│     │  ├─ augmentations.py
│     │  └─ mask_utils.py
│     └─ jail_guard.py
├─ requirements.txt
├─ Queries_and_corpus
│  ├─ hd
│  │  └─ hd.jsonl
│  ├─ hh_test
│  │  └─ hh_test.jsonl
│  ├─ dc
│  │  └─ dc.csv
│  └─ hh_train
│     └─ hh_train.jsonl
├─ README.md
├─ NEARSIDE
│  ├─ minigpt_PCA_W_train_embedding.py
│  ├─ llava_emb.py
│  ├─ llava_PCA_W_train_embedding.py
│  ├─ PCA_method_transfer.py
│  ├─ test_direction.py
│  ├─ minigpt_emb.py
│  └─ get_direction.py
├─ RADAR
│  ├─ instruction.sh
│  ├─ RADAR_dataset
│  │  ├─ minigpt_hd_test
│  │  │  └─ success_qa.json
│  │  ├─ llava_hh_test
│  │  │  └─ success_qa.json
│  │  ├─ minigpt_hh_test
│  │  │  └─ success_qa.json
│  │  ├─ llava_hd_test
│  │  │  └─ success_qa.json
│  │  ├─ llava_hh_train
│  │  │  └─ success_qa.json
│  │  ├─ llava_dc_test
│  │  │  └─ success_qa.json
│  │  ├─ minigpt_hh_train
│  │  │  └─ success_qa.json
│  │  └─ minigpt_dc_test
│  │     └─ success_qa.json
│  └─ RADAR_constructor
│     ├─ RADAR_constructor_minigpt.py
│     └─ RADAR_constructor_llava.py
└─ dra_src
   └─ attack
      ├─ attack_config.json
      ├─ cache
      │  └─ moderation.json
      ├─ harmfulbench_utils.py
      ├─ attack_batch.py
      ├─ attack.py
      ├─ utils.py
      └─ harmful_test.py

```
`dra_src`: The classifier that used to filter the RADAR dataset. Refer to https://github.com/LLM-DRA/DRA?tab=readme-ov-file.

`images`: COCO val images 2017 and COCO test images 2017. Download at https://cocodataset.org/#download.
 
`JailGuard`: The baseline method after our edition. Refer to https://github.com/shiningrain/JailGuard.

`NEARSIDE`: The proposed detection method.

`Queries_and_corpus`: The source QAs and Corpus that used to train the adversarial images and construct the RADAR dataset.

`RADAR`: The proposed datasets. **`We will release the dataset in the near future, please stay tuned😊`**


## Installation

To run the code in this repository, follow these steps:

1. Clone this repository:

   ```sh
   https://github.com/mob-scu/RADAR-NEARSIDE.git
   cd RADAR-NEARSIDE
   ```

2. Prepare the conda enviroment  (Python 3.10.14 is recommended):

   ```sh
   conda create -n RADAR python==3.10.14
   conda activate RADAR
   ```

3. Install the requirements

   ```sh
   pip install -r requirements.txt
   pip install transformers==4.37.2
   ```

4. Install harmbench model

    Please refer to https://huggingface.co/cais/HarmBench-Llama-2-13b-cls

5. prepare llava, minigpt-4, and adversarial attack image(if you want)
   
    Please refer to https://github.com/unispac/visual-adversarial-examples-jailbreak-large-language-models


## RADAR Dataset Construction
We have already placed our RADAR dataset at `RADAR/RADAR_dataset`

If you want to construct the dataset on your own adversarial images
and QAs, please run the constructor of LLaVA or MiniGPT-4.
- There is an example:
    ```sh
   python RADAR/RADAR_constructor/RADAR_constructor_llava.py \
   --image_fold 'images/RADAR_adversarial_images/llava_hh_train'\
   --origin_fold 'images/val2017'\
   --output_fold 'RADAR/RADAR_dataset/llava_hh_train'\
   --test_data_file 'Queries_and_corpus/hh_train/hh_train.jsonl'\
   --test_data_name 'hh_harmless'
   ```

## NEARSIDE Evaluation
### Standard NEARSIDE
1. Enter the NEARSIDE directory.

   ```sh
   cd NEARSIDE
   ```

2. Collect the embeddings of the samples in the test set.

   ```sh
   python llava_emb.py \
   --list_path [RADAR SET]\
   --raw_image_fold [RAW IMAGE]\
   --output_fold [SAVE FOLD]
   ```
    `RADAR SET`: which set to collect embeddings on
    
    `RAW IMAGE`: the benign images corresponding to the adversarial images in the test set.

    `SAVE FOLD`: the path that you want to save the `TEST SET`'s  embeddings

3. Learn the direction from train set.

   ```sh
   python get_direction.py \
   --embedding_dir [EMBEDDINGS]\
   --save_path [PATH TO SAVE]
   ```

    `EMBEDDINGS`: the embeddings of test set refers to 
  
    `PATH TO SAVE`: the path to save the attack direction
4. Use the direction to detect the adversarial embeddings in test set
   
   ```sh
   python test_direction.py \
   --direction_file [DIRECTION]\
   --test_fold [EBEDDINGS OF TEST SET]
   ```

    `DIRECTION`: the DIRECTION of Train set refers to `PATH TO SAVE` in Step 3  
  
    `EBEDDINGS OF TEST SET`: the path of the embeddings of the test set refers to `SAVE FOLD` in Step 2



### Cross-model NEARSIDE

1. Enter the NEARSIDE directory.

   ```sh
   cd NEARSIDE
   ```

2. Collect the embeddings of PCA training and linear transformation W training

   ```sh
   python llava_PCA_W_train_embedding.py
   python minigpt_PCA_W_train_embedding.py
   ```

3. Use the direction of model A to detect the adversarial embeddings of model B

   ```sh
   python PCA_method_transfer.py\
   --direction_file [DIRECTION A]\
   --test_fold_source [A_transfer_hidden]\
   --test_fold_Target [B_transfer_hidden]\
   --train_fold [EMBEDDING THAT DIRECTION A LEARNED FROM]\
   --test_fold [EMBEDDING OF B THAT WILL BE DETECTED]\
   ```
   `A`, `B` refer to `llava` or `minigpt`.



## Cite

```
@article{
}
```

