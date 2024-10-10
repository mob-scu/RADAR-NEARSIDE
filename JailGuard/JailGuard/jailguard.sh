# llava
### hh_train
#nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/llava_attack_success_train_hh --raw_image_fold /data/huangyoucheng/mm-safety/val2017 --model llava
## hh_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/llava_attack_success_test_hh --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model llava > llava_logs/llava_hh.log 2>&1 &
## dc_32_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/llava_attack_success_test_dc_demo_32 --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model llava > llava_logs/llava_dc_32.log 2>&1 &
## sr_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/llava_attack_success_test_sr --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model llava > llava_logs/llava_sr.log 2>&1 &
## hd_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/llava_attack_success_test_hd --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model llava> llava_logs/llava_hd.log 2>&1 &

# minigpt
### hh_train
#nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_train_hh --raw_image_fold /data/huangyoucheng/mm-safety/val2017 --model minigpt4

# adversarial
## hh_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_hh --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58886/v2/models/minigpt/infer > minigpt_logs/minigpt_hh.log 2>&1 &
## dc_32_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_dc_demo_32 --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58887/v2/models/minigpt/infer > minigpt_logs/minigpt_dc_32.log 2>&1 &
## sr_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_sr --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58888/v2/models/minigpt/infer > minigpt_logs/minigpt_sr.log 2>&1 &
## hd_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_hd --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58889/v2/models/minigpt/infer > minigpt_logs/minigpt_hd.log 2>&1 &

# benign
## hh_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_hh --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58886/v2/models/minigpt/infer --type benign > minigpt_logs/minigpt_hh_benign.log 2>&1 &
## dc_32_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_dc_demo_32 --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58887/v2/models/minigpt/infer --type benign > minigpt_logs/minigpt_dc_32_benign.log 2>&1 &
## sr_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_sr --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58888/v2/models/minigpt/infer --type benign > minigpt_logs/minigpt_sr_benign.log 2>&1 &
## hd_test
nohup python jail_guard.py --list_path /data/huangyoucheng/mm-safety/minigpt_attack_success_test_hd --raw_image_fold /data/huangyoucheng/mm-safety/test2017 --model minigpt4 --url http://minigpt4.nextcenter.net:58889/v2/models/minigpt/infer --type benign > minigpt_logs/minigpt_hd_benign.log 2>&1 &
