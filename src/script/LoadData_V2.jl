import Random


json_cfg = read("config/datapath.json", String) 
dpath = JSON3.read(json_cfg)
@unpack datan,seed = args
Random.seed!(seed)

path = dpath[datan][1]

data = NPZ.npzread(path.data);
testdata = NPZ.npzread(path.queries);
val = NPZ.npzread(path.val);
heldout = NPZ.npzread(path.samples);
top100 = NPZ.npzread(path.top100);

# preprocess
train_num = args["train_size"]

begin
    local n,d = size(data)
    local rand_id = Random.randcycle!(Array(1:n))
    global idx = rand_id[1:train_num]
end

test_num=idx[1:5]
println("random number $test_num")

traindata = data[idx,:]
heldout = normalize_ma(heldout)
top100 .+= 1

