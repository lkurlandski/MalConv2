python modify_inc.py \
--config_file=config_files/modify_inc/KernelShap/correspond/most/0.ini \
>>logs/modify_inc/KernelShap/correspond/most/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/KernelShap/correspond/random/0.ini \
>>logs/modify_inc/KernelShap/correspond/random/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/KernelShap/correspond/ordered/0.ini \
>>logs/modify_inc/KernelShap/correspond/ordered/0.log 2>&1;

python modify_inc.py \
--config_file=config_files/modify_inc/KernelShap/least/most/0.ini \
>>logs/modify_inc/KernelShap/correspond/most/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/KernelShap/least/random/0.ini \
>>logs/modify_inc/KernelShap/correspond/random/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/KernelShap/least/ordered/0.ini \
>>logs/modify_inc/KernelShap/correspond/ordered/0.log 2>&1;

python modify_inc.py \
--config_file=config_files/modify_inc/FeatureAblation/correspond/most/0.ini \
>>logs/modify_inc/FeatureAblation/correspond/most/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/FeatureAblation/correspond/random/0.ini \
>>logs/modify_inc/FeatureAblation/correspond/random/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/FeatureAblation/correspond/ordered/0.ini \
>>logs/modify_inc/FeatureAblation/correspond/ordered/0.log 2>&1;

python modify_inc.py \
--config_file=config_files/modify_inc/FeatureAblation/least/most/0.ini \
>>logs/modify_inc/FeatureAblation/correspond/most/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/FeatureAblation/least/random/0.ini \
>>logs/modify_inc/FeatureAblation/correspond/random/0.log 2>&1;
python modify_inc.py \
--config_file=config_files/modify_inc/FeatureAblation/least/ordered/0.ini \
>>logs/modify_inc/FeatureAblation/correspond/ordered/0.log 2>&1;
