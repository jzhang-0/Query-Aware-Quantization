using NPZ
using ArgParse
using Parameters
using JSON3

include("../model/saq_v2.jl")
include("../SearchNeighbors/neighbors.jl")
include("../SearchNeighbors/utils.jl")
include("./arg_config.jl")
include("../utils/utils.jl")

s = ArgParseSettings()
SSR_v2_config!(s)
args = parse_args(s)

include("./LoadData_V2.jl")



include("./explog.jl")
JSON3.pretty(args)

@unpack cuda = args
if cuda == -1
    USE_GPU = false
else
    USE_GPU = true
    device!(cuda)
end

@unpack batch_size,sampling_num = args
sampledata = sampling_samples(heldout, batch_size, sampling_num);

DATA = DATA_s(traindata, val)

@unpack M, Ks, epoch, maxiter, metric = args

saq = SAQ_class(M, Ks, DATA)

pq_codebook = pq_codebook_init_points(traindata, M, Ks)
best_pq_codebook = pq_codebook

@unpack kv = args
all_data_cluster_instance = clusterF(data, kv)

ALL_DATA_ID = all_data_cluster_instance.id
CENTROID = all_data_cluster_instance.centroid

NEW_ID = ALL_DATA_ID[idx]
train_cluster_instance = cluster_s(CENTROID, NEW_ID)

best_recall = 0
sid = 0

best_relative_loss = Inf
gt = brute_force_search(traindata, val)
_, D = size(val)

for samples_id in 1:sampling_num
    @printf("\nsamples id : %d\n", samples_id)
    global pq_codebook,sid
    global best_pq_codebook
    global best_recall
    global best_relative_loss

    samples = sampledata[samples_id,:,:]

    pq_codebook,pq_codes = train(
        saq,
        pq_codebook=pq_codebook,
        cluster_instance = train_cluster_instance,
        samples = samples,
        maxiter = maxiter
        )
    
    local pq_codes


    relative_loss = compute_relative_loss(traindata, pq_codebook, pq_codes, val, gt)

    sp = SN_pq.SearchNeighbors_PQ(M, Ks, D, pq_codebook, pq_codes, "dot_product");        
    neighbors_MA = SN_pq.get_neighbors(sp, val, 100)
    recall100 = compute_recall(neighbors_MA, gt)
    @printf("recall in valset: %.2f \n", recall100)
    @printf("relative_loss in valset: %.4f \n", relative_loss)

    bool_recall = recall100 > best_recall
    bool_relloss = relative_loss < best_relative_loss
    
    if bool_recall 
        best_recall = recall100
        @printf("best recall in valset: %.2f \n", best_recall)
    end

    if bool_relloss
        best_relative_loss = relative_loss
        @printf("best relative_loss in valset: %.4f \n", relative_loss)

    end

    if metric == "recall"
        if bool_recall
            best_pq_codebook = pq_codebook
            sid = samples_id
            npzwrite("$savepath/pq_codebook_id$(samples_id).npz",pq_codebook)
        end 
    elseif metric == "dot_product"
        if bool_relloss 
            best_pq_codebook = pq_codebook
            sid = samples_id
            npzwrite("$savepath/pq_codebook_id$(samples_id).npz",pq_codebook)
        end
    else
        error("metric = recall or dot_product")
    end

end



best_samples = sampledata[sid,:,:]
~,D = size(best_samples)

MATRICES = compute_matrices(best_samples, CENTROID)
all_data_semi_pos_matrix_instance = semi_pos_matrix_s(MATRICES, ALL_DATA_ID)
pq_codes, loss = encode(data, best_pq_codebook, all_data_semi_pos_matrix_instance) 

npzwrite("$savepath/best_pq_codebook_M$(M)K$(Ks).npz",best_pq_codebook)
npzwrite("$savepath/pq_codes_M$(M)K$(Ks).npz", pq_codes)

sp = SN_pq.SearchNeighbors_PQ(M, Ks, D, best_pq_codebook, pq_codes, "dot_product");
queries = testdata

@unpack topk = args
neighbors_MA = SN_pq.get_neighbors(sp, queries, topk)

recall_atN(neighbors_MA, top100)

