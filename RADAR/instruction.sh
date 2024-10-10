# llava
## train data

### hh
CUDA_VISIBLE_DEVICES=0 python RADAR_constructor_llava.py --image_fold visual_constrained_llava_train --origin_fold val2017 --output_fold llava_attack_success_train_hh --test_data_file hh_harmless/train_filtered.jsonl --test_data_name hh_harmless

## test data

### hh
CUDA_VISIBLE_DEVICES=4 python RADAR_constructor_llava.py --image_fold visual_constrained_llava_test --origin_fold test2017 --output_fold llava_attack_success_test_hh --test_data_file hh_harmless/test_filtered.jsonl --test_data_name hh_harmless

### dc_16
  CUDA_VISIBLE_DEVICES=0 python RADAR_constructor_llava.py --image_fold visual_llava_llama_v2_demo --origin_fold test2017 --output_fold llava_attack_success_test_dc_demo_16 --test_data_file harmful_corpus/derogatory_corpus.csv --test_data_name derogatory_corpus
### dc_32
  CUDA_VISIBLE_DEVICES=1 python RADAR_constructor_llava.py --image_fold results_llava_llama_v2_demo_constrained_32 --origin_fold test2017 --output_fold llava_attack_success_test_dc_demo_32 --test_data_file harmful_corpus/derogatory_corpus.csv --test_data_name derogatory_corpus
### dc_64
  CUDA_VISIBLE_DEVICES=1 python RADAR_constructor_llava.py --image_fold results_llava_llama_v2_demo_constrained_64 --origin_fold test2017 --output_fold llava_attack_success_test_dc_demo_64 --test_data_file harmful_corpus/derogatory_corpus.csv --test_data_name derogatory_corpus
### dc_un
  CUDA_VISIBLE_DEVICES=4 python RADAR_constructor_llava.py --image_fold results_llava_llama_v2_demo_unconstrained --origin_fold test2017 --output_fold llava_attack_success_test_dc_demo_un --test_data_file harmful_corpus/derogatory_corpus.csv --test_data_name derogatory_corpus

### hd
CUDA_VISIBLE_DEVICES=1 python RADAR_constructor_llava.py  --image_fold results_llava_llama_constrained_32_harmful_train_filtered --origin_fold test2017 --output_fold llava_attack_success_test_hd --test_data_file harmful_dataset.jsonl --test_data_name harmful_dataset



# minigpt
## train data

### hh
CUDA_VISIBLE_DEVICES=0 python RADAR_constructor_minigpt.py --image_fold /data/tangjingkun/DRA/visual_constrained_eps_32_hh_rlhf --origin_fold val2017 --output_fold /data/tangjingkun/DRA/minigpt_attack_success_train_hh --test_data_file hh_harmless/train_filtered.jsonl --test_data_name hh_harmless --url http://minigpt4.nextcenter.net:58881/v2/models/minigpt/infer

## test data

### hh
CUDA_VISIBLE_DEVICES=1 python RADAR_constructor_minigpt.py --image_fold /data/tangjingkun/DRA/visual_constrained_minpgpt_test --origin_fold test2017 --output_fold minigpt_attack_success_test_hh --test_data_file hh_harmless/test_filtered.jsonl --test_data_name hh_harmless --url http://minigpt4.nextcenter.net:58882/v2/models/minigpt/infer

### derogatory_corpus_32
CUDA_VISIBLE_DEVICES=2 python RADAR_constructor_minigpt.py --image_fold results_minigpt_constrained_32_demo_fixed --origin_fold test2017 --output_fold minigpt_attack_success_test_dc_demo_32 --test_data_file harmful_corpus/derogatory_corpus.csv --test_data_name derogatory_corpus --url http://minigpt4.nextcenter.net:58883/v2/models/minigpt/infer

### harmful dataset
CUDA_VISIBLE_DEVICES=5 python RADAR_constructor_minigpt.py --image_fold results_minigpt_constrained_32_harmful_train_filtered_fixed --origin_fold test2017 --output_fold minigpt_attack_success_test_hd --test_data_file harmful_dataset.jsonl --test_data_name harmful_dataset --url http://minigpt4.nextcenter.net:58885/v2/models/minigpt/infer
CUDA_VISIBLE_DEVICES=4 python RADAR_constructor_minigpt_updown.py --image_fold results_minigpt_constrained_32_harmful_train_filtered_fixed --origin_fold test2017 --output_fold minigpt_attack_success_test_hd --test_data_file harmful_dataset.jsonl --test_data_name harmful_dataset --url http://minigpt4.nextcenter.net:58885/v2/models/minigpt/infer


