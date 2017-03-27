function createModel(mdl, vocsize, Dsize, nout, KKw, ext_feat_size, dist)
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
        local sim_feat = nil
        if dist == 'bilinear' then
          print('bilinear distance')
          sim_feat = nn.Bilinear(NumFilter, NumFilter, 1, false){linput, rinput}
        elseif dist == 'cos' then
          print('cosine distance')
          sim_feat = nn.CosineDistance(){linput, rinput}
        elseif dist == 'dot' then
          print('dot distance')
          sim_feat = nn.DotProduct(){linput, rinput} 
        else
          print('none distance')
        end
        --local vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), bi_dist}
 	local vec_feats, vecs_nn = nil, nil
        if ext_feat_size > 0 then
          print('use ext feature in models.lua')
          local inputs = {linput, rinput, ext_feats}
          if sim_feat == nil then
	    vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), nn.View(-1)(ext_feats)}
          else
            vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), nn.View(-1)(sim_feat), nn.View(-1)(ext_feats)} 
          end
          vecs_nn = nn.gModule(inputs, {vec_feats})
        else
          local inputs = {linput, rinput}
          if sim_feat == nil then
            vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput)}
          else
            vec_feats = nn.JoinTable(1){nn.View(-1)(linput), nn.View(-1)(rinput), nn.View(-1)(sim_feat)}
          end
          vecs_nn = nn.gModule(inputs, {vec_feats})    
        end
        local nlinear = 0
        if dist == 'bilinear' or dist == 'cos' or dist == 'dot' then
          nlinear = 2*NumFilter+ext_feat_size+1
        elseif dist == 'none' then
          nlinear = 2*NumFilter+ext_feat_size
        end

	deepQuery:add(vecs_nn)     
	deepQuery:add(nn.Linear(nlinear, nhid1))
	deepQuery:add(nn.Tanh())
	deepQuery:add(nn.Dropout(0.5))
	deepQuery:add(nn.Linear(nhid1, nout))
	deepQuery:add(nn.LogSoftMax())
	return deepQuery		
    end			  				
end
