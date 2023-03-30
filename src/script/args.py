import argparse
import os

def spec_value(arg, newvaule):
    if arg == "-1" or arg == -1:
        return newvaule
    else:
        return arg

def config_args_SpecData(args):
    if args.datan == "LastFM32D":
        args.data = "data/LastFM/LastFM_data32D.npy"  
        args.queries = "data/LastFM/queries_32D.npy"
        args.tr100 = "data/LastFM/true_neighbors_top100_32D.npy"
        args.sample_queries = "data/LastFM/sample_queries32D.npy"   
    
    elif args.datan == "LastFM100D":        
        args.data = "data/LastFM100D/LastFM_data100D.npy"
        args.queries = "data/LastFM100D/queries_100D.npy"
        args.tr100 = "data/LastFM100D/true_neighbors_top100_100D.npy"
        args.sample_queries = "data/LastFM100D/heldout_set.npy"
    

    elif args.datan == "EchoNest":
        args.data = "data/EchoNest/EchoNest_data.npy"  
        args.queries = "data/EchoNest/queries.npy"
        args.tr100 = "data/EchoNest/true_neighbors_top100.npy"
        args.sample_queries = "data/EchoNest/sample_queries.npy"    

    elif args.datan == "glove":
        args.data = "data/glove/glove-100-angular-normalized.npy"  
        args.queries = "data/glove/queries.npy"
        args.tr100 = "data/glove/true_neighbors_top100.npy"
        args.sample_queries = "data/glove/sample_queries.npy"    
    
    elif args.datan == "glove_622":
        args.data = "julia/data/glove/glove-100-angular-normalized.npy"  
        args.queries = "julia/data/glove/queries.npy"
        args.tr100 = "julia/data/glove/top100.npy"
        args.sample_queries = "julia/data/glove/sample_queries.npy"    

    elif args.datan == "glove_mips":
        args.data = "data/glove_mips/glove.npy"  
        args.queries = "data/glove_mips/queries_test.npy"
        args.tr100 = "data/glove_mips/tr100_test.npy"
        args.sample_queries = "data/glove_mips/queries_heldout.npy"    

    elif args.datan == "Amazon":
        args.data = "data/Amazon/Amazon_data.npy"  
        args.queries = "data/Amazon/queries.npy"
        args.tr100 = "data/Amazon/true_neighbors_top100.npy"
        args.sample_queries = "data/Amazon/sample_queries.npy"

    elif args.datan == "music100":
            args.data = "data/music100/database.npy"  
            args.queries = "data/music100/queries.npy"
            args.tr100 = "data/music100/top100.npy"
            args.sample_queries = "data/music100/sample_queries.npy"
 
    elif args.datan == "music100_433":
            args.data = "data/music100/database.npy"  
            args.queries = "data/music100/user_vecs_spilt_433/queries3k.npy"
            args.tr100 = "data/music100/user_vecs_spilt_433/top100_3k.npy"
            args.sample_queries = "data/music100/user_vecs_spilt_433/sample_queries4k.npy"

    elif args.datan == "YahooMusic":
            args.data = "julia/data/YahooMusic/items.npy"  
            args.queries = "julia/data/YahooMusic/queries.npy"
            args.sample_queries = "julia/data/YahooMusic/sample_queries.npy"
            args.tr100= "julia/data/YahooMusic/true_neighbors_top100.npy"

            
    return args

def ex_config(args):
    exp_path = args.exp_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    save_path = f"{exp_path}/save"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.save_path = save_path

    return args



def parse_datapath(parser):
    parser.add_argument('--data', default='-1', type=str, help='path of datafile')
    parser.add_argument('--sample_queries', default='-1', type=str, help='path of sample_queries file')
    parser.add_argument('--queries', default='-1', type=str, help='path of queriesfile')
    parser.add_argument('--tr100', default='-1', type=str, help='path of grountruth file')

    parser.add_argument('--datan', default='-1', type=str, help='the name of dataset')
    return parser

def parse_pq(parser):
    parser.add_argument('--M', default=-1, type=int, help='the number of codebook') 
    parser.add_argument('--Ks', default=-1, type=int, help='the number of codeword in each codebook')
    parser.add_argument('--pq_code_book', default='-1', type=str, help='path of code_book file')
    parser.add_argument('--pq_codes', default='-1', type=str, help='path of codes file')
    parser.add_argument('--exp_path', default='./test', type=str, help='path of exp')
    return parser

def parse_vq(parser):
    parser.add_argument('--kv', default=-1, type=int, help='the number of codeword in each codebook')
    parser.add_argument('--vq_code_book', default='-1', type=str, help='path of code_book file')
    parser.add_argument('--vq_code', default='-1', type=str, help='path of code file')
    return parser

def parse_config(parser):
    parser.add_argument('--model_name', default="-1", type=str, help='model name')
    parser.add_argument('--sample_num', default=-1, type=int, help='the number of  sample_qrueies')
    parser.add_argument('--training_sample_size', default=100000, type=int, help='training_sample_size')
    parser.add_argument('--topk', default='512', type=int, help='top k neighbors return by algorithm')
    parser.add_argument('--nor_q', default='0', type=int, help='whether sampled queries are normalized')
    parser.add_argument('--init', default='PQ', type=str, help='init method, "PQ" ')
     
    return parser

def parse_mode(parser):
    parser.add_argument('--mode', default="plus", type=str, help='plus / recall')
    return parser



def parse_PQ_van(parser):
    parser = parse_datapath(parser)
    parser.add_argument('--M', default=8, type=int, help='anisotropic_quantization_threshold')
    parser.add_argument('--Ks', default=16, type=int, help='the number of codeword in each codebook')
    parser.add_argument('--topk', default=512, type=int, help='sub dim')
    parser.add_argument('--training_sample_size', default=100000, type=int, help='training_sample_size')
    parser.add_argument('--exp_path', default='./test', type=str, help='path of exp')

    return parser

def parse_OPQ_van(parser):
    parser = parse_PQ_van(parser)
    return parser

def parse_QUIP_cov_van(parser):
    parser = parse_datapath(parser)
    parser = parse_pq(parser)
    parser = parse_config(parser)
    return parser


def parse_QUIP_eign_van(parser):
    parser = parse_datapath(parser)
    parser = parse_pq(parser)
    parser = parse_config(parser)
    return parser


# def parse_scann_plus_softmaxw_van(parser):
#     parser = parse_datapath(parser)
#     parser = parse_pq(parser)
#     parser = parse_config(parser)
#     parser = parse_mode(parser)
#     return parser

# def parse_saq_noweight_van(parser):
#     parser = parse_datapath(parser)
#     parser = parse_pq(parser)
#     parser = parse_config(parser)
#     # parser = parse_mode(parser)
#     return parser



# def parse_scann_sample_van(parser):
#     parser = parse_datapath(parser)
#     parser = parse_pq(parser)
#     parser = parse_config(parser)
#     parser = parse_mode(parser)
#     parser.add_argument('--T', default=0.2, type=float, help='w(t)=1(t>T)')
#     return parser

def parse_scann_van(parser):
    parser.add_argument('--mode', default='-1', type=str, help='Scann_PQ or Recall')

    parser.add_argument('--datan', default='-1', type=str, help='the name of dataset')
    parser.add_argument('--data', default='-1', type=str, help='path of datafile')
    parser.add_argument('--D', default=-1, type=int, help='the dim of data')
    parser.add_argument('--T', default=-1, type=float, help='anisotropic_quantization_threshold=T*mean(norm)')
    parser.add_argument('--nor', default=0, type=int, help='Whether the data is normalized')
    parser.add_argument('--output_file', default='-1', type=str, help='Directory path to save the result')
    parser.add_argument('--exp_path', default='./test', type=str, help='path of exp')

    # PQ
    parser.add_argument('--M', default=-1, type=int, help='the number of codebook') 
    parser.add_argument('--K', default=-1, type=int, help='the number of codeword in each codebook')
    parser.add_argument('--train_num', default=100000, type=int, help='the number of item to be trained') 
    parser.add_argument('--max_iter', default=20, type=int, help='the number of max iteration') 


    # Recall
    parser.add_argument('--queries', default='-1', type=str, help='path of queriesfile')
    parser.add_argument('--tr100', default='-1', type=str, help='path of grountruth file')
    parser.add_argument('--pq_code', default='-1', type=str, help='path of data codes file')
    parser.add_argument('--pq_codebooks', default='-1', type=str, help='path of pq codebooks file')

    parser.add_argument('--topk', default='512', type=int, help='top k neighbors return by algorithm')
    return parser
