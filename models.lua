function createModel(mdl, vocsize, Dsize, nout, KKw, ext_feat_size)
    local D     = Dsize --opt.dimension
    local kW    = KKw --opt.kwidth
    local dW    = 1 -- opt.dwidth
    local nhid1 = 201 --opt.nhid1 
    local NumFilter = 100
	    
    if mdl == 'SIGIR' then
	deepQuery=nn.Sequential()
        local q_conv = nn.Sequential()
 	q_conv:add(nn.TemporalConvolution(D, NumFilter, kW, 1))
	q_conv:add(nn.Tanh())
	q_conv:add(nn.Max(1))
	q_conv:add(nn.Reshape(1, NumFilter))
	--local modelQ= featext:clone('weight','bias','gradWeight','gradBias')
        local ans_conv = nn.Sequential()
        ans_conv:add(nn.TemporalConvolution(D, NumFilter, kW, 1))
        ans_conv:add(nn.Tanh())
        ans_conv:add(nn.Max(1))
        ans_conv:add(nn.Reshape(1, NumFilter))	
	
	paraQuery=nn.ParallelTable()
	paraQuery:add(q_conv)
       	paraQuery:add(ans_conv)
        if ext_feat_size > 0 then
          paraQuery:add(nn.Identity())
	end
        deepQuery:add(paraQuery)
	
	local linput, rinput, ext_feats = nn.Identity()(), nn.Identity()(), nn.Identity()()
	--local inputs = {linput, rinput}
	local bi_dist = nn.Bilinear(NumFilter, NumFilter, 1, false){linput, rinput}
        --local vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), bi_dist}
 	local vecs_nn = nil
        if ext_feat_size > 0 then
          print('use ext feature in models.lua')
          local inputs = {linput, rinput, ext_feats}
	  local vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), bi_dist, nn.View(-1)(ext_feats)}
	  vecs_nn = nn.gModule(inputs, {vec_feats})
        else
          local inputs = {linput, rinput}
          local vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), bi_dist}
          vecs_nn = nn.gModule(inputs, {vec_feats})    
        end
	--local vecs_nn = nn.gModule(inputs, {vec_feats})
	deepQuery:add(vecs_nn)     
	deepQuery:add(nn.Linear(2*NumFilter+1+ext_feat_size, nhid1))
	deepQuery:add(nn.Tanh())
	deepQuery:add(nn.Dropout(0.5))
	deepQuery:add(nn.Linear(nhid1, nout))
	deepQuery:add(nn.LogSoftMax())
	return deepQuery		
    end			  				
end
