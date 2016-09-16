function createModel(mdl, vocsize, Dsize, nout, KKw)
    local featext = nn.Sequential()

    local D     = Dsize --opt.dimension
    local kW    = KKw --opt.kwidth
    local dW    = 1 -- opt.dwidth
    local nhid1 = 200 --opt.nhid1 
    local NumFilter = 100
	    
    if mdl == 'SIGIR' then
	deepQuery=nn.Sequential()
	featext:add(nn.TemporalConvolution(D, NumFilter, kW, 1))
	featext:add(nn.ReLU())
	featext:add(nn.Max(1))
	featext:add(nn.Reshape(1, NumFilter))
	local modelQ= featext:clone('weight','bias','gradWeight','gradBias')
	
	paraQuery=nn.ParallelTable()
	paraQuery:add(modelQ)
       	paraQuery:add(featext)
	deepQuery:add(paraQuery)
	
	local linput, rinput = nn.Identity()(), nn.Identity()()
	local inputs = {linput, rinput}
	local bi_dist = nn.Bilinear(NumFilter, NumFilter, 1, false){linput, rinput}
	local vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), bi_dist}
	local vecs_nn = nn.gModule(inputs, {vec_feats})
	deepQuery:add(vecs_nn)     
	deepQuery:add(nn.Linear(2*NumFilter+1, nhid1))
	deepQuery:add(nn.ReLU())
	deepQuery:add(nn.Dropout(0.5))
	deepQuery:add(nn.Linear(nhid1, nout))
	deepQuery:add(nn.LogSoftMax())
	return deepQuery		
    end			  				
end
