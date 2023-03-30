if !@isdefined USE_GPU
   global USE_GPU = false
end

using LinearAlgebra
using  Printf
using CUDA
using TimerOutputs
CA = CuArray

include("../utils/utils_dec.jl")
include("../utils/utils.jl")
const _SAQ_CLASS = TimerOutput()

const newaxis = [CartesianIndex()]

abstract type SAQ_CLASS end

struct DATA_s
    train::AbstractArray{<:AbstractFloat,2}
    val::AbstractArray{<:AbstractFloat,2}
    D::Integer
    function DATA_s(train, val)
        D = size(train)[2]
        new(train, val, D)
    end
end

struct cluster_s
    centroid::AbstractArray{<:AbstractFloat, 2} # (k, d)
    id::AbstractArray{<:Integer,1} # (n,)
end

struct semi_pos_matrix_s
    matrices::AbstractArray{<:AbstractFloat, 3} # (k, d, d)
    id::AbstractArray{<:Integer,1} # (n,)
end 

struct weighted_vecs_s
    None
end

struct SAQ_class <: SAQ_CLASS
    M::Integer
    Ks::Integer
    D::Integer
    Ds::Integer
    DATA::DATA_s
    function SAQ_class(M, Ks, DATA)
        D = DATA.D
        Ds = Int(D / M)
        new(M, Ks, D, Ds, DATA)
    end
end

function compute_ma(
    samples::AbstractArray{<:AbstractFloat, 2},
    point::AbstractArray{<:AbstractFloat, 1}, #(D,)
    )

    num,D = size(samples)
    ip = samples * point   
    ip .-= maximum(ip)
    ip = exp.(ip)
    ip_sum = sum(ip)
    ma = zeros(D, D)
    for i in 1:num
        v = @views samples[i,:]
        qi = @views ip[i]/ip_sum * v * transpose(v)
        ma += @views qi
    end
    return ma
end

@dec info function compute_matrices(
    samples::AbstractArray{<:AbstractFloat, 2},
    points::AbstractArray{<:AbstractFloat, 2}, #(n, D)
    )
    n,D = size(points)
    matrices = zeros(n, D, D)
    Threads.@threads for i in 1:n
        point = @views points[i, :]
        matrices[i, :, :] = compute_ma(samples, point)
    end
    
    return matrices
end

function single_loss_2(
    point::AbstractArray{<:AbstractFloat, 1}, #(D,)
    q_data::AbstractArray{<:AbstractFloat, 1},
    ma::AbstractArray{<:AbstractFloat, 2}
    )
    r = point - q_data
    # return r' * ma * r
    return dot(r, ma, r)
end

function single_encode(
    saq::SAQ_CLASS, 
    point::AbstractArray{<:AbstractFloat, 1},
    pq_codebook::AbstractArray{<:AbstractFloat, 3},
    weight_ma::AbstractArray{<:AbstractFloat, 2},
    iter_num::Integer = 3
    )
    """
    pq_codebook (M, Ks, Ds)
    """
    (M, Ks, Ds) = size(pq_codebook)
    D = saq.D

    code = zeros(Int,M)
    code .+= 1

    q_data = zeros(D)
    for i in 1:M
        q_data[ ((i-1) * Ds + 1 ): i * Ds] = pq_codebook[i, code[i], :]
    end

    for _ in 1:iter_num
        for subcb in 1:M
            Q_errorlist = zeros(Ks)
            for i in 1:Ks
                code[subcb] = i
                q_data[ ((subcb-1) * Ds + 1):subcb * Ds] = pq_codebook[subcb, i, :] # pq_codebook[(subcb * Ks + i) * Ds:(subcb * Ks + i + 1) * Ds]
                loss = single_loss_2(point, q_data, weight_ma)
                Q_errorlist[i] = loss
            end

            code[subcb] = argmin(Q_errorlist)
            q_data[ ((subcb-1) * Ds + 1) : subcb * Ds] = pq_codebook[subcb, code[subcb], :]  # self.codebook[(subcb * Ks + s_code[subcb]) * Ds:(subcb * Ks + s_code[subcb] + 1) * Ds]
        end
    end  
    loss = single_loss_2(point, q_data, weight_ma)
    return code,loss

end

@dec info function encode(
    points::AbstractArray{<:AbstractFloat,2}, 
    pq_codebook::AbstractArray{<:AbstractFloat,3},
    semi_pos_matrix_instance::semi_pos_matrix_s
    )
    """
    
    notes: points  semi_pos 
    
    return code,loss
    """
    n, D = size(points)
    M, Ks, Ds = size(pq_codebook)

    code = zeros(Int, n, M)
    loss = zeros(n)
    
    ID = semi_pos_matrix_instance.id
    MATRICS = semi_pos_matrix_instance.matrices

    Threads.@threads for i in 1:n
        point = @view points[i,:]
        
        
        ma_id = ID[i]
        ma = MATRICS[ma_id,:,:]
        
        code[i,:],loss[i] = single_encode(saq, point, pq_codebook, ma)
    end
    return code, sum(loss)
end

@dec info function update_pqcodebook(
    saq::SAQ_CLASS,
    construct_vecs::AbstractArray{<:AbstractFloat,2},
    pq_codes2::AbstractArray{<:Integer,2},
    semi_pos_matrix_instance::semi_pos_matrix_s,
    )

    Ks = saq.Ks
    n,D = size(construct_vecs)
    points = @views saq.DATA.train    

    ID = semi_pos_matrix_instance.id
    MATRICS = semi_pos_matrix_instance.matrices

    Ts = zeros(Ks * D, Ks * D)
    Rs = zeros(Ks * D)
    @views for i in 1:n
        ii = pq_codes2[i,:]
        x = points[i,:]
                
        ma_id = ID[i]
        ma = MATRICS[ma_id,:,:]

        Ts[ii, ii] += ma
        
        Rs[ii] += construct_vecs[i,:]
    end

    C = (Ts + 1e-3 * I(Ks * D)) \ Rs
    
    return C
end

function object_loss(
    saq::SAQ_CLASS,
    code2::AbstractArray{<:Integer,2}, 
    pq_codebook::AbstractArray{<:AbstractFloat,1}, 
    sampledata::AbstractArray{<:AbstractFloat,2}, 
    )
    """    
        Args:
            II:(n,M*Ds)
        return:
            loss
    """
    data = saq.DATA.train

    n = size(data)[1]
    loss = zeros(n)

     
    Threads.@threads for i in 1:n
        x = data[i, :]
        ii = code2[i,:]
        qx = pq_codebook[ii]
        ma = compute_ma(sampledata, x)
        loss[i] = single_loss_2(x, qx, ma)
    end
    return sum(loss)
end

@dec info function compute_construct_ves(
    points::AbstractArray{<:AbstractFloat,2},
    semi_pos_matrix_instance::semi_pos_matrix_s
    )
    """
    points: 
    samples:queries
    """

    n,D = size(points)
    construct_ves = zeros(n,D)

    ID = semi_pos_matrix_instance.id
    MATRICS = semi_pos_matrix_instance.matrices

    Threads.@threads for i in 1:n
        point = @views points[i,:]
        ma = MATRICS[ID[i],:,:]
        construct_ves[i,:] = ma * point
    end
    
    return construct_ves
end

@views function train(
    saq::SAQ_CLASS;
    pq_codebook::AbstractArray{<:AbstractFloat,3},
    cluster_instance::cluster_s,
    samples::AbstractArray{<:AbstractFloat,2},
    maxiter::Integer=10
    )

    Ks = saq.Ks
    D = saq.D
    Ds = saq.Ds

    TRAIN_DATA = saq.DATA.train 
    ID = cluster_instance.id
    CENTROID = cluster_instance.centroid

    MATRICES = compute_matrices(samples, CENTROID)

    spmi = semi_pos_matrix_s(MATRICES, ID) 
    construct_vecs = compute_construct_ves(TRAIN_DATA, spmi)

    for _ in 1:maxiter
        pq_codes, loss = encode(TRAIN_DATA, pq_codebook, spmi)
        global pq_codes
        @printf("loss %.3f \n", loss)
        pq_codes2 = pqcode_To_pqcode2(pq_codes, Ks, D)
        pq_codebook1d = update_pqcodebook(saq, construct_vecs, pq_codes2, spmi)
        pq_codebook = pq_codebook_1dto3d(pq_codebook1d, M, Ks, Ds)
    end

    return pq_codebook, pq_codes
end

using ParallelKMeans
@dec info function clusterF(
    all_points::AbstractArray{<:AbstractFloat, 2},
    kv::Integer # 
    )
    """
    return centroid (d, kv), id
    """
    n, d = size(all_points)
    R = kmeans(all_points', kv, max_iters = 20)
    return cluster_s(R.centers', R.assignments)
end

function compute_relative_loss(
    data::AbstractArray{<:AbstractFloat,2},
    pq_codebook::AbstractArray{<:AbstractFloat,3},
    pq_codes::AbstractArray{<:Integer,2},
    queries::AbstractArray{<:AbstractFloat,2},
    gt::AbstractArray{<:Integer,1},
    )
    _,D = size(data)
    _,M = size(pq_codes)

    relative_loss = 0
    val_n = length(gt)
    for i in 1:val_n
        query_val = queries[i,:]
        max_index = gt[i]
        vec = data[max_index, :]
        code = pq_codes[max_index, :]
        vec_compress = zeros(D)
        Ds = Int(D / M)
        for m in 1:M
            vec_compress[(m-1)*Ds + 1 : m * Ds] = pq_codebook[m, code[m],:]
        end
        appr_ip = query_val'vec_compress

        max_ip = query_val'vec

        relative_loss = relative_loss + abs(max_ip - appr_ip)/(max_ip)
    end
    relative_loss = relative_loss / val_n
    return relative_loss
end

