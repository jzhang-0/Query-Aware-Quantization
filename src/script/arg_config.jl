using ArgParse

function default_config!(s)
    @add_arg_table s begin
        "--topk"
        help = "return topk neighbors"
        arg_type = Int
        default = 512

        "--maxiter"
        help = "max iter in encode and update_codebook"
        arg_type = Int
        default = 2

        "--epoch"
        help = " the number of traverse data (used in sampel_softmax)"
        arg_type = Int
        default = 3

        "--seed"
        help = "random seed"
        arg_type = Int
        default = 1234

        "--train_size"
        help = " the number of datapoints to train model"
        arg_type = Int
        default = Int(1e5)

        "--cuda"
        help = "the id of cuda device "
        arg_type = Int
        default = -1

        "--sampling_num","-s"
        help = "the number of sampling (used in sample softmax with reeplacement, abbr SSR)"
        arg_type = Int
        default = 50
    end
end

function config!(s)
    @add_arg_table s begin
        "--M"
            help = "M"
            arg_type = Int
            default = -1
        "--Ks"
            help = "Ks"
            arg_type = Int
            default = -1
        
        "--datan"
            help = "data name"
            arg_type = String
            default = "-1"
        
        "--batch_size","-b"
            arg_type = Int
            default = -1

        "--exp_id","-i"
            help = "exp id"
            arg_type = String
            default = "0"

        "--metric","-m"
            help = "metric, recall or dot_product"
            arg_type = String
            default = "recall"        
    end
end

function sample_softmax_config!(s)
    default_config!(s)
    config!(s)
    @add_arg_table s begin
        "--modeln"
        help = "model name"
        arg_type = String
        default = "sample_softmax"
    end
end

function SSR_config!(s)
    default_config!(s)
    config!(s)
    @add_arg_table s begin
        "--modeln"
        help = "model name"
        arg_type = String
        default = "SSR"
    end
end

function SSR_v2_config!(s)
    default_config!(s)
    config!(s)
    @add_arg_table s begin
        "--modeln"
        help = "model name"
        arg_type = String
        default = "SSR_V2"

        "--kv"
        help = "the num of centroids"
        arg_type = Int64
        default = 1000
    end
end


function SSR_test_config!(s)
    SSR_config!(s)
    @add_arg_table s begin
        "--pq_codebook_path","-p"
        help = "model name"
        arg_type = String
        default = "-1"

        "--sid"
        arg_type = Int64
        default = -1
    end

end