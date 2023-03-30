#! /bin/bash

modeln="SimpleLSH"
exp_id="1"
datan="LastFM32D"

exp_path="Exp/${modeln}/$datan/Exp${exp_id}"
mkdir -p ${exp_path}

python -u src/script/lsh/simplelsh_main_lastfm32D.py\
    |tee "${exp_path}/exp.log"
