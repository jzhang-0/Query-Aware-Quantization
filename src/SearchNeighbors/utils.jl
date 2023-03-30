using Printf
function brute_force_search(
    target_set::AbstractArray{<:AbstractFloat,2}, 
    test_set::AbstractArray{<:AbstractFloat,2}; 
    metric = "dot_product")
    """
    target_set:(n, d)
    test_set:(nq, d) 
    """
    if metric == "dot_product"
        inner_product = target_set * test_set' # n * nq
        Cidx = argmax(inner_product,dims=1) # 1*n matrix
        return reshape(getindex.(Cidx, 1),:) 
    end
    println("not implment")
end

function compute_recall(neighbors, ground_truth)
    """
    neighbors:(n, topk)
    ground_truth:(n, gt)
    """
    total = 0
    n = size(neighbors)[1]
    for i in 1:n
        gt_row = ground_truth[i,:]
        row = neighbors[i,:]
        total += length(intersect(gt_row, row))
    end

    return total / length(ground_truth)

end 



function  recall_atN_(neighbors_matrix, ground_truth)
    """
    Args:
        neighbors_matrix:(nq,topK=512)
        ground_truth: 1d or 2d
    """
    topk = size(neighbors_matrix)[2]
    if length(size(ground_truth)) == 1
        ng = 1
    end

    if length(size(ground_truth)) == 2
        ng = size(ground_truth)[2]
    end

    N = [1,2,4,8,10,16,20,32,64,100,128,256,512]
    N_t = typeof(N)

    recall_list=[]
    N_list = []
    for tn in N
        if ng <= tn <= topk
            neighbors_matrix_topN = neighbors_matrix[:,1:tn]

            recall = compute_recall(neighbors_matrix_topN, ground_truth)
            recall = round(recall;digits = 4)
            append!(recall_list, recall)
            append!(N_list,tn)

            @printf("recall %d@%d = %.4f\n",ng,tn,recall)
        end
    end
    
    return convert(Vector{Float64}, recall_list), convert(N_t, N_list)
end 

function recall_atN(neighbors_matrix,ground_truth)
    """
    Args:
        neighbors_matrix:(nq,topK=512)
        ground_truth:(nq,) or (nq,>=10)
    """
    if length(size(ground_truth)) == 2 
        ground_truth_1 = ground_truth[:,1]
        ground_truth_10 = ground_truth[:,1:10]
        r1, N1 = recall_atN_(neighbors_matrix, ground_truth_1)
        print("\n")
        r10, N10 = recall_atN_(neighbors_matrix, ground_truth_10)
        print("\n")
        println("N=",N1)
        println("recall1@N:",r1)
        println("N=",N10)
        println("recall10@N:",r10)
    end
    if length(size(ground_truth)) == 1 
        r1, N1 = recall_atN_(neighbors_matrix, ground_truth)
        print("\n")
        println("N=",N1)
        println("recall1@N:",r1)
    end 
end
