import Random
import StatsBase
Random.seed!(1234)
using LinearAlgebra
function spilt_samples(samples::AbstractArray{<:AbstractFloat,2}, batch_size::Integer = 100)
    """
        return (batch, batch_size, D)
    """
    n,D = size(samples);
    batch = Int(n / batch_size);
    sampledata = reshape(samples, batch, batch_size, D)   
    return sampledata
end

function sampling_samples(
    samples::AbstractArray{<:AbstractFloat,2}, 
    batch_size::Integer,
    sampling_num::Integer
    )
    n,D = size(samples);
    sampledata = zeros(sampling_num, batch_size, D)
    for i in 1:sampling_num
        idx = StatsBase.sample(1:n, batch_size, replace = false)
        global sampledata[i,:,:] = samples[idx,:]    
    end
    return sampledata
end

function pq_codebook_1dto3d(
    pq_codebook::AbstractArray{<:AbstractFloat,1},
    M::Integer, 
    Ks::Integer, 
    Ds::Integer
    )
    
    pq_codebook_ = zeros(Float32, M, Ks, Ds)
    
    for i in 1:M
        cc = pq_codebook[(i-1)*Ks*Ds + 1 : i*Ks*Ds]
        cc =reshape(cc, Ds, Ks)
        cc = transpose(cc)
        pq_codebook_[i,:,:] = cc
    end
    return pq_codebook_
end

function pq_codebook_3dto1d(    
    pq_codebook::AbstractArray{<:AbstractFloat,3}
    )
    M,Ks,Ds = size(pq_codebook)
    pq_codebook_ = zeros(M*Ks*Ds)
    for i in 1:M
        for i2 in 1:Ks
            pq_codebook_[ (((i-1)*Ks + i2 -1)*Ds + 1) : ((i-1)*Ks + i2)*Ds ] = pq_codebook[i,i2,:]
        end
    end
    return pq_codebook_
end

function pqcode_To_pqcode2(
    pq_codes::AbstractArray{<:Integer,2},
    Ks::Integer,
    D::Integer
    )
    """
    pq_codes:shape=(n, M)
    Ks:the number of codewords in one codebook
    D:the dim of original data
    """
    n,M = size(pq_codes)
    Ds = Int(D / M)
    pq_codes2 = zeros(Integer, n, D)

    @views for k in 1:n
        for i in 1:M
            end_i = (i-1) * Ks * Ds + pq_codes[k, i]*Ds
            pq_codes2[k, ((i-1)*Ds + 1) : i*Ds] = (end_i - Ds + 1) : end_i
        end
    end
    return pq_codes2

end

function pq_codebook_init_points(
    X::AbstractArray{<:AbstractFloat,2},
    M::Integer,
    Ks::Integer
    )
    n, D = size(X)
    Ds = Int(D / M)

    idx = Random.shuffle(1:n)
    pq_codebook = zeros(M,Ks,Ds)
    for i in 1:M
        X_sub = @view X[:, (i-1)*Ds + 1 : i*Ds]
        pq_codebook[i,:,:] = @view X_sub[idx[ Ks*(i-1) + 1 : Ks*i ], :]
    end

    return pq_codebook
end

function normalize_ma(ma::AbstractArray{<:AbstractFloat})
    n,d = size(ma)
    n_ma = zeros(n,d)
    for i in 1:n
        x = ma[i,:]
        n_ma[i,:] = normalize(x)
    end
    return n_ma
end 
