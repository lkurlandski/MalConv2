pid=291658

# An entry in /proc means that the process is still running.
while [ -d "/proc/$pid" ]; do
    sleep 60
done

/home/lk3591/miniconda3/envs/MalConv2/bin/python /home/lk3591/Documents/MalConv2/explain.py --config_file=/home/lk3591/Documents/MalConv2/config_files/explain/KernelShap.ini; /home/lk3591/miniconda3/envs/MalConv2/bin/python /home/lk3591/Documents/MalConv2/explain.py --config_file=/home/lk3591/Documents/MalConv2/config_files/explain/FeatureAblation.ini
