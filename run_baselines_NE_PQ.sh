#! /bin/bash

modeln="NEQ"
exp_id="1"
datan="LastFM32D"

exp_path="Exp/${modeln}/$datan/Exp${exp_id}"
mkdir -p ${exp_path}

python -u src/script/NEQ/s_lastfm32d.py\
    |tee "${exp_path}/exp.log"
