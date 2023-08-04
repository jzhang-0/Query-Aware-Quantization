#! /bin/bash

modeln="ScaNN"
exp_id="1"
datan="LastFM32D"

exp_path="Exp/${modeln}/$datan/Exp${exp_id}"
mkdir -p ${exp_path}/save

python -u src/script/run_scann.py --M 8 --K 16 --datan ${datan}\
    --T 0.01\
    --D 32\
    --nor 0\
    --exp_path ${exp_path}\
    |tee "${exp_path}/exp.log"
