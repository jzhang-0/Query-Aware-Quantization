import ExprTools

macro dec(decorator, inner_def)
    inner_def = ExprTools.splitdef(inner_def)
    outer_def = copy(inner_def)
    fname = get(inner_def, :name, nothing)
    if fname !== nothing
        @assert fname isa Symbol
        inner_def[:name] = Symbol(fname, :_inner)
    end
    outer_def[:body] = Expr(:call,
        :($decorator($(ExprTools.combinedef(inner_def)))),
        get(outer_def, :args, [])...,
        get(outer_def, :kwargs, [])...,
    )
    return esc(ExprTools.combinedef(outer_def))
end

info(f) = (x...) -> (
    t0=time(); 
    fun_name = String(Symbol(f));
    println("\nbegin ",fun_name);
    val=f(x...); 
    println("end ",fun_name);
    println(fun_name," took ", time()-t0, " s"); 
    val
    )

macro T(matrix) # transpose(M) = @T(M) = @T M = M' 
    return :(transpose($matrix))
end
