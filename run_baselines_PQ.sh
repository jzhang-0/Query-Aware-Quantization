#! /bin/bash

modeln="PQ"
exp_id="1"
datan="LastFM32D"

exp_path="Exp/${modeln}/$datan/Exp${exp_id}"
mkdir -p ${exp_path}/save

python -u src/script/run_pq.py --M 8 --Ks 16 --datan ${datan}\
    --exp_path ${exp_path}\
    |tee "${exp_path}/exp.log"
