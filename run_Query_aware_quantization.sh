#! /bin/bash
modeln="QaQ"
exp_id="1"
datan=LastFM32D

mkdir -p Exp/${modeln}/$datan/Exp${exp_id}/save

julia --threads 16 src/script/run_SSR_V2.jl --kv 100 --datan ${datan} --M 8 --Ks 16 -s 20 -b 500 --modeln ${modeln} -i ${exp_id} \
    |tee -a Exp/${modeln}/$datan/Exp${exp_id}/${exp_id}.log
