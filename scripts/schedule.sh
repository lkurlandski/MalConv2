pid=834762

# An entry in /proc means that the process is still running.
while [ -d "/proc/$pid" ]; do
    sleep 60
done

python modify_inc.py --config_file=/home/lk3591/Documents/MalConv2/config_files/modify_inc/corresponding_ordered.ini
>>/home/lk3591/Documents/MalConv2/logs/modify_inc/corresponding_ordered.log 2>&1 &
wait; \
/home/lk3591/miniconda3/envs/MalConv2/bin/python /home/lk3591/Documents/MalConv2/modify_inc.py \
--config_file=/home/lk3591/Documents/MalConv2/config_files/modify_inc/corresponding_random.ini
\>>/home/lk3591/Documents/MalConv2/logs/modify_inc/corresponding_random.log 2>&1 &\
wait; \
/home/lk3591/miniconda3/envs/MalConv2/bin/python /home/lk3591/Documents/MalConv2/modify_inc.py \
--config_file=/home/lk3591/Documents/MalConv2/config_files/modify_inc/least_most.ini
\>>/home/lk3591/Documents/MalConv2/logs/modify_inc/least_most.log 2>&1 &\
wait; \
/home/lk3591/miniconda3/envs/MalConv2/bin/python /home/lk3591/Documents/MalConv2/modify_inc.py \
--config_file=/home/lk3591/Documents/MalConv2/config_files/modify_inc/least_ordered.ini
\>>/home/lk3591/Documents/MalConv2/logs/modify_inc/least_ordered.log 2>&1 &\
wait; \
/home/lk3591/miniconda3/envs/MalConv2/bin/python /home/lk3591/Documents/MalConv2/modify_inc.py \
--config_file=/home/lk3591/Documents/MalConv2/config_files/modify_inc/least_random.ini
\>>/home/lk3591/Documents/MalConv2/logs/modify_inc/least_random.log 2>&1 &\
