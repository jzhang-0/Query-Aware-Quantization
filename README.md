## Introuction
This is the code of Query-Aware Quantization for Maximum Inner Product Search. I'm the first author.

The proposed method is implemented by [Julia](https://julialang.org/) and all basleines are implmented by Python. To use this code, you need to check [requirements.txt](requirements.txt).

We provide the data of lastfm and run script for reproducing results. 
```
bash shell_script/run_Query_aware_quantization.sh
```

If you want to use our method in your dataset, you need to add your dataset name and path in [`config/datapath.json`](config/datapath.json).
  
- **Execution Script**: To run the model, execute the script [`src/script/run_SSR_V2.jl`](src/script/run_SSR_V2.jl).

- **Parameter Details**: For a detailed explanation of the parameters, refer to [`src/script/arg_config.jl`](src/script/arg_config.jl).

- **Method Implementation**: The source code for the proposed method can be found at [`src/model/saq_v2.jl`](src/model/saq_v2.jl).



If this code can help you, please cite this work,
```
@inproceedings{zhang2023query,
  title={Query-Aware Quantization for Maximum Inner Product Search},
  author={Zhang, Jin and Lian, Defu and Zhang, Haodi and Wang, Baoyun and Chen, Enhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={4},
  pages={4875--4883},
  year={2023}
}
```