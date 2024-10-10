# EFFECTIVE AND EFFICIENT ADVERSARIAL DETEC- TION FOR VISION-LANGUAGE MODELS VIA A SINGLE VECTOR

[![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)]()
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Contact us anytime: tangjingkun@stu.scu.edu.cn.

## Table of Contents

- [Installation](#installation)
- [RADAR Dataset Construction](#radar-dataset-construction)
- [NEARSIDE Adversarial Detection](#nearside-adversarial-detection)
- [Cite](#cite)

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

