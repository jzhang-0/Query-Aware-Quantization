@unpack modeln,datan,exp_id = args

exppath = "Exp/$modeln/$datan/Exp$exp_id"
savepath = "$exppath/save"
println("savepath = $(savepath)")
try
    mkpath(savepath)
catch
    println("dir has exist")
end

