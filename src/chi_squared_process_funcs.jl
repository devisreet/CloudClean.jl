using StatsBase
using LinearAlgebra

export proc_discrete_dt

export proc_discrete_svd  #versions of proc_discrete
export proc_discrete_revised_dt   #versions of proc_discrete
export proc_discrete_ctot_dt
export proc_discrete_ctot_revised_dt #versions of proc_discrete

export ctot_chi_squared_stats  #CHI_2
export chi_squared_stats  #CHI_2
export varyr_chi_squared_stats  #CHI_2
export check_masks  #CHI_2


"""
    proc_discrete(x_locs, y_locs, raw_image, mask_image; Np=33, widx=129, widy=widx, tilex=1, tiley=tilex, seed=2021, ftype::Int=32, ndraw=0) -> out_mean, out_draw

Process an image with a mask, replacing masked pixels with either a mean or draw from a distribution resembling the local pixel-pixel covariance structure in the image.

# Arguments:
- `x_locs`: A 1D array representing the location centers (in the x coordinate) for infilling.
- `y_locs`: A 1D array representing the location centers (in the y coordinate) for infilling.
- `raw_image`: A 2D array representing the input image.
- `mask_image`: A 2D array representing the mask.

# Keywords:
- `Np`: An optional integer specifying the number of pixels in a side (default: 33).
- `widx`: An optional integer specifying the width of the region used for training the local covariance in the x-direction (default: 129).
- `widy`: An optional integer specifying the width of the region used for training the local covariance in the y-direction (default: widx).
- `tilex`: An optional integer specifying the number of tiles in the x-direction for subdividing the image (default: 1).
- `tiley`: An optional integer specifying the number of tiles in the y-direction for subdividing the image (default: tilex).
- `seed`: An optional integer specifying the random number generator seed (default: 2021).
- `ftype`: An optional integer specifying the floating-point precision type (32 or 64) (default: 32).
- `rlim`: Radius limit for the radial mask beyond which pixels are not used for conditioning (units are pixels^2). (default: 625)
- `ndraw`: An optional integer specifying the number of draws of samples from the statistical distribution of possible masked pixel values (default: 0).

# Returns
- If `ndraw` is 0, returns the debiased image as a 2D array.
- If `ndraw` is greater than 0, returns the debiased image as a 2D array and an array of `ndraw` draws.

# Examples
```julia
julia> raw_image = rand(100, 100)
julia> mask_image = kstar_circle_mask(100,rlim=256)
julia> result = proc_continuous([50],[50],raw_image, mask_image, Np=33, widx=129, seed=2021)
```
    
"""
#edits

#for svd version (in cond_cov) of proc_discrete
function proc_discrete_svd(x_locs,y_locs,raw_image,mask_image;Np=33,widx=129,widy=widx,tilex=1,tiley=tilex,seed=2021,ftype::Int=32,rlim=625,ndraw=0, verbose=true, use_svd=false)
    radNp = (Np-1)÷2
    if ftype == 32
        T = Float32
    else
        T = Float64
    end

    # renaming to match conventions
    ref_im = raw_image
    bmaskd = mask_image
    (sx0, sy0) = size(ref_im)

    x_stars = x_locs
    y_stars = y_locs
    (Nstars,) = size(x_stars)

    mod_im = StatsBase.median(ref_im)*ones(T,(sx0, sy0))
 
    testim = mod_im .- ref_im
    bimage = zeros(T,sx0,sy0)
    bimageI = zeros(Int64,sx0,sy0)
    testim2 = zeros(T,sx0,sy0)
    bmaskim2 = zeros(Bool,sx0,sy0)
    goodpix = zeros(Bool,sx0,sy0)

    prelim_infill!(testim,bmaskd,bimage,bimageI,testim2,bmaskim2,goodpix;widx=19,widy=19,ftype=ftype,verbose=verbose)
    testim .= mod_im .- ref_im #fixes current overwrite for 0 infilling

    ## calculate the star farthest outside the edge of the image in x and y
    cx = round.(Int,x_stars)
    cy = round.(Int,y_stars)
    px0 = outest_bounds(cx,sx0)
    py0 = outest_bounds(cy,sy0)

    ## these have to be allocating to get the noise model right
    Δx = (widx-1)÷2
    Δy = (widy-1)÷2
    padx = Np+Δx+px0
    pady = Np+Δy+py0
    in_image = ImageFiltering.padarray(testim2,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_image_raw = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_bmaskd = ImageFiltering.padarray(bmaskd,ImageFiltering.Fill(true,(padx+2,pady+2)));
    out_mean = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    out_mean[in_bmaskd].=NaN
    out_draw = if ndraw!=1
        ImageFiltering.padarray(repeat(testim,outer=[1 1 ndraw]),ImageFiltering.Pad(:reflect,(padx+2,pady+2,0)));
    else
        ImageFiltering.padarray(repeat(testim,outer=[1 1]),ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    end
    for i=1:ndraw
        out_draw[in_bmaskd,i].=NaN
    end

    diffim = view(ref_im,1:sx0-1,:).-view(ref_im,2:sx0,:)
    in_sigiqr = sig_iqr(filter(.!isnan,diffim))
    
    add_sky_noise!(in_image,in_bmaskd,in_sigiqr;seed=seed)

    ## iterate over all star positions and compute errorbars/debiasing corrections
    star_stats = zeros(T,10,Nstars)
    star_k = zeros(Int32,10,Nstars)

    cov = zeros(T,Np*Np,Np*Np)
    μ = zeros(T,Np*Np)

    # compute a radial mask for reduced num cond pixels
    #circmask = kstar_circle_mask(Np,rlim=rlim)

    # some important global sizes for the loop
    cntStar0 = 0
    stepx = (sx0+2) ÷ tilex
    stepy = (sy0+2) ÷ tiley

    # precallocate the image subblocks
    in_subimage = zeros(T,stepx+2*padx,stepy+2*pady)
    ism = zeros(T,stepx+2*padx,stepy+2*pady)
    bimage = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy)
    bism = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy,2*Np-1, Np);
    for jx=1:tilex, jy=1:tiley
        xrng, yrng, star_ind = im_subrng(jx,jy,cx,cy,sx0+2,sy0+2,px0,py0,stepx,stepy,padx,pady,tilex,tiley)
        cntStar = length(star_ind)
        if cntStar > 0
            in_subimage .= in_image[xrng,yrng]
            cov_avg!(bimage, ism, bism, in_subimage, widx=widx, widy=widy,Np=Np)
            offx = padx-Δx-(jx-1)*stepx
            offy = pady-Δy-(jy-1)*stepy
            for i in star_ind
                build_cov!(cov,μ,cx[i]+offx,cy[i]+offy,bimage,bism,Np,widx,widy)
                cov_stamp = cx[i]-radNp:cx[i]+radNp,cy[i]-radNp:cy[i]+radNp
                    
                kmasked2d = in_bmaskd[cov_stamp[1],cov_stamp[2]]
                #kstar, kcond = gen_pix_mask_circ(kmasked2d,circmask;Np=Np)
                kstar, kcond = gen_pix_mask_trivial(kmasked2d; Np=Np) #simpler version
                
                data_in = in_image_raw[cov_stamp[1],cov_stamp[2]]

                # try
                    stat_out = condCovEst_wdiag_svd(cov,μ,kstar,data_in,Np=Np,export_mean=true,n_draw=ndraw,seed=seed, use_svd=use_svd)
                    
                    data_in[kstar].=stat_out[1][kstar]
                    in_image_raw[cov_stamp[1],cov_stamp[2]].=data_in
                    
                    data_in = out_mean[cov_stamp[1],cov_stamp[2]]
                    data_in[kstar].=stat_out[1][kstar]
                    out_mean[cov_stamp[1],cov_stamp[2]].=data_in
                    for i=1:ndraw
                        draw_in = out_draw[cov_stamp[1],cov_stamp[2],i]
                        draw_in[kstar].= stat_out[2][kstar,i]
                        out_draw[cov_stamp[1],cov_stamp[2],i].=draw_in
                    end
                    kmasked2d[kstar].=false
                    in_bmaskd[cov_stamp[1],cov_stamp[2]].=kmasked2d
                    cntStar0 += cntStar
                # catch
                #     println("Positive Definite Error")
                # end
            end
        end
        cntStar0 += cntStar
        if verbose
            println("Finished $cntStar stars in tile ($jx, $jy)")
        end
        flush(stdout)
    end
    if ndraw>0
        return mod_im[1].-out_mean[1:sx0, 1:sy0], mod_im[1].-out_draw[1:sx0, 1:sy0, :]
    else
        return mod_im[1].-out_mean[1:sx0, 1:sy0]
    end
end


function proc_discrete_dt(x_locs,y_locs,raw_image,mask_image;Np=33,widx=129,widy=widx,tilex=1,tiley=tilex,seed=2021,ftype::Int=32,rlim=625,ndraw=0,verbose=true)
    radNp = (Np-1)÷2
    if ftype == 32
        T = Float32
    else
        T = Float64
    end

    # renaming to match conventions
    ref_im = raw_image
    bmaskd = mask_image
    (sx0, sy0) = size(ref_im)

    x_stars = x_locs
    y_stars = y_locs
    (Nstars,) = size(x_stars)

    mod_im = StatsBase.median(ref_im)*ones(T,(sx0, sy0))
 
    testim = mod_im .- ref_im
    bimage = zeros(T,sx0,sy0)
    bimageI = zeros(Int64,sx0,sy0)
    testim2 = zeros(T,sx0,sy0)
    bmaskim2 = zeros(Bool,sx0,sy0)
    goodpix = zeros(Bool,sx0,sy0)

    prelim_infill!(testim,bmaskd,bimage,bimageI,testim2,bmaskim2,goodpix;widx=19,widy=19,ftype=ftype,verbose=verbose)
    testim .= mod_im .- ref_im #fixes current overwrite for 0 infilling

    ## calculate the star farthest outside the edge of the image in x and y
    cx = round.(Int,x_stars)
    cy = round.(Int,y_stars)
    px0 = outest_bounds(cx,sx0)
    py0 = outest_bounds(cy,sy0)

    ## these have to be allocating to get the noise model right
    Δx = (widx-1)÷2
    Δy = (widy-1)÷2
    padx = Np+Δx+px0
    pady = Np+Δy+py0
    in_image = ImageFiltering.padarray(testim2,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_image_raw = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_bmaskd = ImageFiltering.padarray(bmaskd,ImageFiltering.Fill(true,(padx+2,pady+2)));
    out_mean = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    out_mean[in_bmaskd].=NaN
    out_draw = if ndraw!=1
        ImageFiltering.padarray(repeat(testim,outer=[1 1 ndraw]),ImageFiltering.Pad(:reflect,(padx+2,pady+2,0)));
    else
        ImageFiltering.padarray(repeat(testim,outer=[1 1]),ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    end
    for i=1:ndraw
        out_draw[in_bmaskd,i].=NaN
    end

    diffim = view(ref_im,1:sx0-1,:).-view(ref_im,2:sx0,:)
    in_sigiqr = sig_iqr(filter(.!isnan,diffim))
    
    add_sky_noise!(in_image,in_bmaskd,in_sigiqr;seed=seed)

    ## iterate over all star positions and compute errorbars/debiasing corrections
    star_stats = zeros(T,10,Nstars)
    star_k = zeros(Int32,10,Nstars)

    cov = zeros(T,Np*Np,Np*Np)
    μ = zeros(T,Np*Np)

    # compute a radial mask for reduced num cond pixels
    circmask = kstar_circle_mask(Np,rlim=rlim)

    # some important global sizes for the loop
    cntStar0 = 0
    stepx = (sx0+2) ÷ tilex
    stepy = (sy0+2) ÷ tiley

    # precallocate the image subblocks
    in_subimage = zeros(T,stepx+2*padx,stepy+2*pady)
    ism = zeros(T,stepx+2*padx,stepy+2*pady)
    bimage = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy)
    bism = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy,2*Np-1, Np);
    for jx=1:tilex, jy=1:tiley
        xrng, yrng, star_ind = im_subrng(jx,jy,cx,cy,sx0+2,sy0+2,px0,py0,stepx,stepy,padx,pady,tilex,tiley)
        cntStar = length(star_ind)
        if cntStar > 0
            in_subimage .= in_image[xrng,yrng]
            cov_avg!(bimage, ism, bism, in_subimage, widx=widx, widy=widy,Np=Np)
            offx = padx-Δx-(jx-1)*stepx
            offy = pady-Δy-(jy-1)*stepy
            for i in star_ind
                build_cov!(cov,μ,cx[i]+offx,cy[i]+offy,bimage,bism,Np,widx,widy)

                cov_stamp = cx[i]-radNp:cx[i]+radNp,cy[i]-radNp:cy[i]+radNp
                    
                kmasked2d = in_bmaskd[cov_stamp[1],cov_stamp[2]]
                #kstar, kcond = gen_pix_mask_circ(kmasked2d,circmask;Np=Np)
                kstar, kcond = gen_pix_mask_trivial(kmasked2d; Np=Np) #simpler version
                data_in = in_image_raw[cov_stamp[1],cov_stamp[2]]

                # try
                    predcovar = condCovEst_wdiag_dt(cov,μ,kstar,data_in,Np=Np,export_mean=true,n_draw=ndraw,seed=seed);
                    return cov, predcovar, μ
            end
        end
    end
end

#function for chi-2 too
function proc_discrete_revised_dt(x_locs,y_locs,raw_image,mask_image;Np=33,widx=129,widy=widx,tilex=1,tiley=tilex,seed=2021,ftype::Int=32,rlim=625,ndraw=0, verbose=false)
    radNp = (Np-1)÷2
    if ftype == 32
        T = Float32
    else
        T = Float64
    end

    # renaming to match conventions
    ref_im = raw_image
    bmaskd = mask_image
    (sx0, sy0) = size(ref_im)

    x_stars = x_locs
    y_stars = y_locs
    (Nstars,) = size(x_stars)

    mod_im = StatsBase.median(ref_im)*ones(T,(sx0, sy0))
 
    testim = mod_im .- ref_im
    bimage = zeros(T,sx0,sy0)
    bimageI = zeros(Int64,sx0,sy0)
    testim2 = zeros(T,sx0,sy0)
    bmaskim2 = zeros(Bool,sx0,sy0)
    goodpix = zeros(Bool,sx0,sy0)

    prelim_infill!(testim,bmaskd,bimage,bimageI,testim2,bmaskim2,goodpix;widx=19,widy=19,ftype=ftype, verbose=verbose)
    testim .= mod_im .- ref_im #fixes current overwrite for 0 infilling

    ## calculate the star farthest outside the edge of the image in x and y
    cx = round.(Int,x_stars)
    cy = round.(Int,y_stars)
    px0 = outest_bounds(cx,sx0)
    py0 = outest_bounds(cy,sy0)

    ## these have to be allocating to get the noise model right
    Δx = (widx-1)÷2
    Δy = (widy-1)÷2
    padx = Np+Δx+px0
    pady = Np+Δy+py0
    in_image = ImageFiltering.padarray(testim2,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_image_raw = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_bmaskd = ImageFiltering.padarray(bmaskd,ImageFiltering.Fill(true,(padx+2,pady+2)));
    out_mean = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    out_mean[in_bmaskd].=NaN
    out_draw = if ndraw!=1
        ImageFiltering.padarray(repeat(testim,outer=[1 1 ndraw]),ImageFiltering.Pad(:reflect,(padx+2,pady+2,0)));
    else
        ImageFiltering.padarray(repeat(testim,outer=[1 1]),ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    end
    for i=1:ndraw
        out_draw[in_bmaskd,i].=NaN
    end

    diffim = view(ref_im,1:sx0-1,:).-view(ref_im,2:sx0,:)
    in_sigiqr = sig_iqr(filter(.!isnan,diffim))
    
    add_sky_noise!(in_image,in_bmaskd,in_sigiqr;seed=seed)

    ## iterate over all star positions and compute errorbars/debiasing corrections
    star_stats = zeros(T,10,Nstars)
    star_k = zeros(Int32,10,Nstars)

    cov = zeros(T,Np*Np,Np*Np)
    μ = zeros(T,Np*Np)

    # compute a radial mask for reduced num cond pixels
    #circmask = kstar_circle_mask(Np,rlim=rlim)

    # some important global sizes for the loop
    cntStar0 = 0
    stepx = (sx0+2) ÷ tilex
    stepy = (sy0+2) ÷ tiley

    # precallocate the image subblocks
    in_subimage = zeros(T,stepx+2*padx,stepy+2*pady)
    ism = zeros(T,stepx+2*padx,stepy+2*pady)
    bimage = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy)
    bism = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy,2*Np-1, Np);
    for jx=1:tilex, jy=1:tiley
        xrng, yrng, star_ind = im_subrng(jx,jy,cx,cy,sx0+2,sy0+2,px0,py0,stepx,stepy,padx,pady,tilex,tiley)
        cntStar = length(star_ind)
        if cntStar > 0
            in_subimage .= in_image[xrng,yrng]
            cov_avg!(bimage, ism, bism, in_subimage, widx=widx, widy=widy,Np=Np)
            offx = padx-Δx-(jx-1)*stepx
            offy = pady-Δy-(jy-1)*stepy
            for i in star_ind
                
                build_cov!(cov,μ,cx[i]+offx,cy[i]+offy,bimage,bism,Np,widx,widy)
                
                cov_stamp = cx[i]-radNp:cx[i]+radNp,cy[i]-radNp:cy[i]+radNp
                    
                kmasked2d = in_bmaskd[cov_stamp[1],cov_stamp[2]]
                #kstar, kcond = gen_pix_mask_circ(kmasked2d,circmask;Np=Np)
                global kstar, kcond = gen_pix_mask_trivial(kmasked2d; Np=Np) #simpler version
                data_in = in_image_raw[cov_stamp[1],cov_stamp[2]]

                # try
                    global predcovar, stat_out = condCovEst_wdiag_revised_dt(cov,μ,kstar,data_in,Np=Np,export_mean=true,n_draw=ndraw,seed=seed);
                    data_in[kstar].=stat_out[1][kstar]
                    in_image_raw[cov_stamp[1],cov_stamp[2]].=data_in
                    
                    data_in = out_mean[cov_stamp[1],cov_stamp[2]]
                    data_in[kstar].=stat_out[1][kstar]
                    out_mean[cov_stamp[1],cov_stamp[2]].=data_in
                    for i=1:ndraw
                        draw_in = out_draw[cov_stamp[1],cov_stamp[2],i]
                        draw_in[kstar].= stat_out[2][kstar,i]
                        out_draw[cov_stamp[1],cov_stamp[2],i].=draw_in
                    end
                    kmasked2d[kstar].=false
                    in_bmaskd[cov_stamp[1],cov_stamp[2]].=kmasked2d
                    cntStar0 += cntStar
                # catch
                #     println("Positive Definite Error")
                # end
            end
        end
        cntStar0 += cntStar
        if verbose
            println("Finished $cntStar stars in tile ($jx, $jy)")
        end
        flush(stdout)
    end
    if ndraw>0
        # i may have used this function for something else later :( in which i used the mod_im-out_mean output
        return predcovar, cov, kstar, μ, stat_out[1], stat_out[2] #mod_im[1].-out_mean[1:sx0, 1:sy0], mod_im[1].-out_draw[1:sx0, 1:sy0, :] 
    else
        return cov, mod_im[1].-out_mean[1:sx0, 1:sy0] 
    end
end

#grf version where ctot and mu are provided 
#this one is for chi-2 comps 
function proc_discrete_ctot_revised_dt(cov,μ, x_locs,y_locs,raw_image,mask_image;Np=33,widx=129,widy=widx,tilex=1,tiley=tilex,seed=2021,ftype::Int=32,rlim=625,ndraw=0, verbose=true)
    radNp = (Np-1)÷2
    if ftype == 32
        T = Float32
    else
        T = Float64
    end

    # renaming to match conventions
    ref_im = raw_image
    bmaskd = mask_image
    (sx0, sy0) = size(ref_im)

    x_stars = x_locs
    y_stars = y_locs
    (Nstars,) = size(x_stars)

    mod_im = StatsBase.median(ref_im)*ones(T,(sx0, sy0))
 
    testim = mod_im .- ref_im
#     bimage = zeros(T,sx0,sy0)
#     bimageI = zeros(Int64,sx0,sy0)
#     testim2 = zeros(T,sx0,sy0)
#     bmaskim2 = zeros(Bool,sx0,sy0)
#     goodpix = zeros(Bool,sx0,sy0)

#     prelim_infill!(testim,bmaskd,bimage,bimageI,testim2,bmaskim2,goodpix;widx=19,widy=19,ftype=ftype,verbose=verbose)
    testim .= mod_im .- ref_im #fixes current overwrite for 0 infilling

    ## calculate the star farthest outside the edge of the image in x and y
    cx = round.(Int,x_stars)
    cy = round.(Int,y_stars)
    px0 = outest_bounds(cx,sx0)
    py0 = outest_bounds(cy,sy0)

    ## these have to be allocating to get the noise model right
    Δx = (widx-1)÷2
    Δy = (widy-1)÷2
    padx = Np+Δx+px0
    pady = Np+Δy+py0
    # in_image = ImageFiltering.padarray(testim2,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_image_raw = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_bmaskd = ImageFiltering.padarray(bmaskd,ImageFiltering.Fill(true,(padx+2,pady+2)));
    out_mean = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    out_mean[in_bmaskd].=NaN
    out_draw = if ndraw!=1
        ImageFiltering.padarray(repeat(testim,outer=[1 1 ndraw]),ImageFiltering.Pad(:reflect,(padx+2,pady+2,0)));
    else
        ImageFiltering.padarray(repeat(testim,outer=[1 1]),ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    end
    for i=1:ndraw
        out_draw[in_bmaskd,i].=NaN
    end

    diffim = view(ref_im,1:sx0-1,:).-view(ref_im,2:sx0,:)
    in_sigiqr = sig_iqr(filter(.!isnan,diffim))
    
    # add_sky_noise!(in_image,in_bmaskd,in_sigiqr;seed=seed)

    ## iterate over all star positions and compute errorbars/debiasing corrections
    star_stats = zeros(T,10,Nstars)
    star_k = zeros(Int32,10,Nstars)

    #cov = zeros(T,Np*Np,Np*Np)
    #μ = zeros(T,Np*Np)

    # compute a radial mask for reduced num cond pixels
    #circmask = kstar_circle_mask(Np,rlim=rlim)

    # some important global sizes for the loop
    cntStar0 = 0
    stepx = (sx0+2) ÷ tilex
    stepy = (sy0+2) ÷ tiley

    # precallocate the image subblocks
    # in_subimage = zeros(T,stepx+2*padx,stepy+2*pady)
    ism = zeros(T,stepx+2*padx,stepy+2*pady)
    bimage = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy)
    bism = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy,2*Np-1, Np);
    for jx=1:tilex, jy=1:tiley
        xrng, yrng, star_ind = im_subrng(jx,jy,cx,cy,sx0+2,sy0+2,px0,py0,stepx,stepy,padx,pady,tilex,tiley)
        cntStar = length(star_ind)
        if cntStar > 0
            # in_subimage .= in_image[xrng,yrng]
            # cov_avg!(bimage, ism, bism, in_subimage, widx=widx, widy=widy,Np=Np)
            offx = padx-Δx-(jx-1)*stepx
            offy = pady-Δy-(jy-1)*stepy
            for i in star_ind
                # build_cov!(cov,μ,cx[i]+offx,cy[i]+offy,bimage,bism,Np,widx,widy)
                cov_stamp = cx[i]-radNp:cx[i]+radNp,cy[i]-radNp:cy[i]+radNp
                    
                kmasked2d = in_bmaskd[cov_stamp[1],cov_stamp[2]]
                #kstar, kcond = gen_pix_mask_circ(kmasked2d,circmask;Np=Np)
                global kstar, kcond = gen_pix_mask_trivial(kmasked2d; Np=Np) #simpler version
                
                data_in = in_image_raw[cov_stamp[1],cov_stamp[2]]

                # try
                    #stat_out = condCovEst_wdiag(cov,μ,kstar,data_in,Np=Np,export_mean=true,n_draw=ndraw,seed=seed)
                    global predcovar, stat_out = condCovEst_wdiag_revised_dt(cov,μ,kstar,data_in,Np=Np,export_mean=true,n_draw=ndraw,seed=seed);
                    
                    data_in[kstar].=stat_out[1][kstar]
                    in_image_raw[cov_stamp[1],cov_stamp[2]].=data_in
                    
                    data_in = out_mean[cov_stamp[1],cov_stamp[2]]
                    data_in[kstar].=stat_out[1][kstar]
                    out_mean[cov_stamp[1],cov_stamp[2]].=data_in
                    for i=1:ndraw
                        draw_in = out_draw[cov_stamp[1],cov_stamp[2],i]
                        draw_in[kstar].= stat_out[2][kstar,i]
                        out_draw[cov_stamp[1],cov_stamp[2],i].=draw_in
                    end
                    kmasked2d[kstar].=false
                    in_bmaskd[cov_stamp[1],cov_stamp[2]].=kmasked2d
                    cntStar0 += cntStar
                # catch
                #     println("Positive Definite Error")
                # end
            end
        end
        cntStar0 += cntStar
        if verbose
            println("Finished $cntStar stars in tile ($jx, $jy)")
        end
        flush(stdout)
    end
    if ndraw>0
       # return mod_im[1].-out_mean[1:sx0, 1:sy0], mod_im[1].-out_draw[1:sx0, 1:sy0, :]
        return predcovar, cov, kstar, μ, stat_out[1], stat_out[2]
    else
        return mod_im[1].-out_mean[1:sx0, 1:sy0]
    end
end

##USES SVD functionality
function proc_discrete_ctot_dt(cov,μ, x_locs,y_locs,raw_image,mask_image;Np=33,widx=129,widy=widx,tilex=1,tiley=tilex,seed=2021,ftype::Int=32,rlim=625,ndraw=0, verbose=true, use_svd=false, low_rank=false)
    radNp = (Np-1)÷2
    if ftype == 32
        T = Float32
    else
        T = Float64
    end

    # renaming to match conventions
    ref_im = raw_image
    bmaskd = mask_image
    (sx0, sy0) = size(ref_im)

    x_stars = x_locs
    y_stars = y_locs
    (Nstars,) = size(x_stars)

    mod_im = StatsBase.median(ref_im)*ones(T,(sx0, sy0))
 
    testim = mod_im .- ref_im
#     bimage = zeros(T,sx0,sy0)
#     bimageI = zeros(Int64,sx0,sy0)
#     testim2 = zeros(T,sx0,sy0)
#     bmaskim2 = zeros(Bool,sx0,sy0)
#     goodpix = zeros(Bool,sx0,sy0)

#     prelim_infill!(testim,bmaskd,bimage,bimageI,testim2,bmaskim2,goodpix;widx=19,widy=19,ftype=ftype,verbose=verbose)
    testim .= mod_im .- ref_im #fixes current overwrite for 0 infilling

    ## calculate the star farthest outside the edge of the image in x and y
    cx = round.(Int,x_stars)
    cy = round.(Int,y_stars)
    px0 = outest_bounds(cx,sx0)
    py0 = outest_bounds(cy,sy0)

    ## these have to be allocating to get the noise model right
    Δx = (widx-1)÷2
    Δy = (widy-1)÷2
    padx = Np+Δx+px0
    pady = Np+Δy+py0
    # in_image = ImageFiltering.padarray(testim2,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_image_raw = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    in_bmaskd = ImageFiltering.padarray(bmaskd,ImageFiltering.Fill(true,(padx+2,pady+2)));
    out_mean = ImageFiltering.padarray(testim,ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    out_mean[in_bmaskd].=NaN
    out_draw = if ndraw!=1
        ImageFiltering.padarray(repeat(testim,outer=[1 1 ndraw]),ImageFiltering.Pad(:reflect,(padx+2,pady+2,0)));
    else
        ImageFiltering.padarray(repeat(testim,outer=[1 1]),ImageFiltering.Pad(:reflect,(padx+2,pady+2)));
    end
    for i=1:ndraw
        out_draw[in_bmaskd,i].=NaN
    end

    diffim = view(ref_im,1:sx0-1,:).-view(ref_im,2:sx0,:)
    in_sigiqr = sig_iqr(filter(.!isnan,diffim))
    
    # add_sky_noise!(in_image,in_bmaskd,in_sigiqr;seed=seed)

    ## iterate over all star positions and compute errorbars/debiasing corrections
    star_stats = zeros(T,10,Nstars)
    star_k = zeros(Int32,10,Nstars)

    #cov = zeros(T,Np*Np,Np*Np)
    #μ = zeros(T,Np*Np)

    # compute a radial mask for reduced num cond pixels
    #circmask = kstar_circle_mask(Np,rlim=rlim)

    # some important global sizes for the loop
    cntStar0 = 0
    stepx = (sx0+2) ÷ tilex
    stepy = (sy0+2) ÷ tiley

    # precallocate the image subblocks
    # in_subimage = zeros(T,stepx+2*padx,stepy+2*pady)
    ism = zeros(T,stepx+2*padx,stepy+2*pady)
    bimage = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy)
    bism = zeros(T,stepx+2*padx-2*Δx,stepy+2*pady-2*Δy,2*Np-1, Np);
    for jx=1:tilex, jy=1:tiley
        xrng, yrng, star_ind = im_subrng(jx,jy,cx,cy,sx0+2,sy0+2,px0,py0,stepx,stepy,padx,pady,tilex,tiley)
        cntStar = length(star_ind)
        if cntStar > 0
            # in_subimage .= in_image[xrng,yrng]
            # cov_avg!(bimage, ism, bism, in_subimage, widx=widx, widy=widy,Np=Np)
            offx = padx-Δx-(jx-1)*stepx
            offy = pady-Δy-(jy-1)*stepy
            for i in star_ind
                # build_cov!(cov,μ,cx[i]+offx,cy[i]+offy,bimage,bism,Np,widx,widy)
                cov_stamp = cx[i]-radNp:cx[i]+radNp,cy[i]-radNp:cy[i]+radNp
                    
                kmasked2d = in_bmaskd[cov_stamp[1],cov_stamp[2]]
                #kstar, kcond = gen_pix_mask_circ(kmasked2d,circmask;Np=Np)
                kstar, kcond = gen_pix_mask_trivial(kmasked2d; Np=Np) #simpler version
                
                data_in = in_image_raw[cov_stamp[1],cov_stamp[2]]

                # try
                    stat_out = condCovEst_wdiag_svd(cov,μ,kstar,data_in,Np=Np,export_mean=true,n_draw=ndraw,seed=seed, use_svd=use_svd, low_rank=low_rank)
                    
                    data_in[kstar].=stat_out[1][kstar]
                    in_image_raw[cov_stamp[1],cov_stamp[2]].=data_in
                    
                    data_in = out_mean[cov_stamp[1],cov_stamp[2]]
                    data_in[kstar].=stat_out[1][kstar]
                    out_mean[cov_stamp[1],cov_stamp[2]].=data_in
                    for i=1:ndraw
                        draw_in = out_draw[cov_stamp[1],cov_stamp[2],i]
                        draw_in[kstar].= stat_out[2][kstar,i]
                        out_draw[cov_stamp[1],cov_stamp[2],i].=draw_in
                    end
                    kmasked2d[kstar].=false
                    in_bmaskd[cov_stamp[1],cov_stamp[2]].=kmasked2d
                    cntStar0 += cntStar
                # catch
                #     println("Positive Definite Error")
                # end
            end
        end
        cntStar0 += cntStar
        if verbose
            println("Finished $cntStar stars in tile ($jx, $jy)")
        end
        flush(stdout)
    end
    if ndraw>0
        return mod_im[1].-out_mean[1:sx0, 1:sy0], mod_im[1].-out_draw[1:sx0, 1:sy0, :]
    else
        return mod_im[1].-out_mean[1:sx0, 1:sy0]
    end
end


#CHI-SQUARED 
function ctot_chi_squared_stats(cov,μ,x_locs,y_locs,raw_image,mask_image,img;Np=33, widx=129,widy=widx,tilex=1,ftype=64, tiley=tilex,seed=2021,rlim=625,ndraw=1, compute_contr=false, org_mu=false, verbose=true) #see more_correct_look for why these are bad pix

    predcov, cov, kstar, mean_real, mean_infill, star_stats2 = proc_discrete_ctot_revised_dt(cov,μ,x_locs,y_locs,raw_image,mask_image,Np=Np, widx=widx,widy=widy,ftype=ftype, tilex=tilex,tiley=tiley,seed=seed,rlim=rlim,ndraw=ndraw, verbose=verbose);
    
    icov = cholesky(cov);
    ipredcov = cholesky(predcov);
    cenx = x_locs[1];
    ceny = y_locs[1];
    
    median_img = median(raw_image) 
    
    xreal_ctot = chisquared_xreal_ctot(img, raw_image, icov, mean_real, Np, cenx, ceny);
    xreal_cinfill = chisquared_xreal_cinfill(img, raw_image, kstar, ipredcov, mean_infill, median_img, cenx, ceny, Np);
    
    
    xinfill_ctot_manyinfills =Vector{Float64}()
    xinfill_cinfill_manyinfills = Vector{Float64}()

    
    for i in 1:ndraw
        xinfill_ctot = chisquared_xinfill_ctot(star_stats2, icov, mean_real, Np, cenx, ceny, i);
        append!(xinfill_ctot_manyinfills, xinfill_ctot);
            
        xinfill_cinfill = chisquared_xinfill_cinfill(star_stats2, kstar, ipredcov, mean_infill, cenx, ceny, Np, i);
        append!(xinfill_cinfill_manyinfills,xinfill_cinfill);
    end
    
    #all new below for if compute_cont
    if compute_contr
        contributions_xinfill_ctot_manyinfills = []
        
        for i in 1:ndraw
            contributions_xinfill_ctot = split_chisquared_contr_xinfill_ctot(star_stats2, kstar, icov, mean_real, Np, cenx, ceny, i)
            append!(contributions_xinfill_ctot_manyinfills,contributions_xinfill_ctot);
        end  
        
        return [xreal_ctot, xinfill_ctot_manyinfills, xreal_cinfill, xinfill_cinfill_manyinfills, contributions_xinfill_ctot_manyinfills]

    else
        return [xreal_ctot, xinfill_ctot_manyinfills, xreal_cinfill, xinfill_cinfill_manyinfills]
    end
    
   # return [xreal_ctot, xinfill_ctot_manyinfills, xreal_cinfill, xinfill_cinfill_manyinfills]
    
end 
    


### ends ctot things





function chi_squared_stats(x_locs,y_locs,raw_image,mask_image,img;Np=33, widx=129,widy=widx,tilex=1,ftype=64, tiley=tilex,seed=2021,rlim=625,ndraw=1, compute_contr=false, org_mu=false) #see more_correct_look for why these are bad pix

    predcov, cov, kstar, mean_real, mean_infill, star_stats2 = proc_discrete_revised_dt(x_locs,y_locs,raw_image,mask_image,Np=Np, widx=widx,widy=widy,ftype=ftype, tilex=tilex,tiley=tiley,seed=seed,rlim=rlim,ndraw=ndraw, org_mu=org_mu);
    
    icov = cholesky(cov);
    ipredcov = cholesky(predcov);
    cenx = x_locs[1];
    ceny = y_locs[1];
    
    median_img = median(raw_image) 
    
    xreal_ctot = chisquared_xreal_ctot(img, raw_image, icov, mean_real, Np, cenx, ceny);
    xreal_cinfill = chisquared_xreal_cinfill(img, raw_image, kstar, ipredcov, mean_infill, median_img, cenx, ceny, Np);
    
    
    xinfill_ctot_manyinfills =Vector{Float64}()
    xinfill_cinfill_manyinfills = Vector{Float64}()

    
    for i in 1:ndraw
        xinfill_ctot = chisquared_xinfill_ctot(star_stats2, icov, mean_real, Np, cenx, ceny, i);
        append!(xinfill_ctot_manyinfills, xinfill_ctot);
            
        xinfill_cinfill = chisquared_xinfill_cinfill(star_stats2, kstar, ipredcov, mean_infill, cenx, ceny, Np, i);
        append!(xinfill_cinfill_manyinfills,xinfill_cinfill);
    end
    
    #all new below for if compute_cont
    if compute_contr
        contributions_xinfill_ctot_manyinfills = []
        
        for i in 1:ndraw
            contributions_xinfill_ctot = split_chisquared_contr_xinfill_ctot(star_stats2, kstar, icov, mean_real, Np, cenx, ceny, i)
            append!(contributions_xinfill_ctot_manyinfills,contributions_xinfill_ctot);
        end  
        
        return [xreal_ctot, xinfill_ctot_manyinfills, xreal_cinfill, xinfill_cinfill_manyinfills, contributions_xinfill_ctot_manyinfills]

    else
        return [xreal_ctot, xinfill_ctot_manyinfills, xreal_cinfill, xinfill_cinfill_manyinfills]
    end
    
   # return [xreal_ctot, xinfill_ctot_manyinfills, xreal_cinfill, xinfill_cinfill_manyinfills]
    
end 
    
function varyr_chi_squared_stats(x_locs,y_locs,raw_image,img;Np=33, widx=129,widy=widx,tilex=1,ftype=64, tiley=tilex,seed=2021,rlim=625,ndraw=1, badpixels=false, badpixmask=[262:263, 723:723]) #see more_correct_look for why these are bad pix
    
    cenx = x_locs[1]
    ceny = y_locs[1]
    dv = (Np-1)÷2
    
    #loops = isqrt(dv)
    loops = dv
    
    #initialize
    chi_squared_xreal_ctot = []
    chi_squared_xinfill_ctot = []
    chi_squared_xinfill_cinfill= []
    chi_squared_xreal_cinfill = []
    
    #mask bad pixels 
    if badpixels == true
        raw_image[badpixmask[1],badpixmask[2]].=0;
    end
    
    for i in 0:loops
        r=i 
        bimage = zeros(Bool,size(raw_image));
        circmask = .!kstar_circle_mask(Np,rlim=r^2);
        bimage[(cenx-dv):(cenx+dv),(ceny-dv):(ceny+dv)].=circmask;
        raw_image[bimage].=0;
    
        chi_squared_vals = chi_squared_stats(x_locs,y_locs,raw_image,bimage,img,Np=Np,widx=widx,widy=widy,tilex=tilex,ftype=ftype,tiley=tiley,seed=seed,rlim=rlim,ndraw=ndraw)
        
        
        push!(chi_squared_xreal_ctot, chi_squared_vals[1])
        push!(chi_squared_xinfill_ctot, chi_squared_vals[2])
        push!(chi_squared_xreal_cinfill, chi_squared_vals[3])
        push!(chi_squared_xinfill_cinfill, chi_squared_vals[4])
        
    end
    return chi_squared_xreal_ctot,chi_squared_xinfill_ctot, chi_squared_xinfill_cinfill, chi_squared_xreal_cinfill
    
end


#check out our masks 

function check_masks(x_locs,y_locs,raw_image;Np=33, widx=129,widy=widx,tilex=1,ftype=64, tiley=tilex,seed=2021,rlim=625,ndraw=1, badpixels=false, badpixmask=[262:263, 723:723]) #see more_correct_look for why these are bad pix
    cenx = x_locs[1]
    ceny = y_locs[1]
    dv = (Np-1)÷2
    
    #loops = isqrt(dv)
    loops = dv
    r_list= Vector{Float64}()
    kstar_list= []
    k_list = []
    
    #mask bad pixels 
    if badpixels == true
        raw_image[badpixmask[1],badpixmask[2]].=0;
    end
    
    
    for i in 0:loops
        r=i 
        append!(r_list, r)
        
        bimage = zeros(Bool,size(raw_image));
        circmask = .!kstar_circle_mask(Np,rlim=r^2);
        bimage[(cenx-dv):(cenx+dv),(ceny-dv):(ceny+dv)].=circmask;
        raw_image[bimage].=0;
        
        predcov, cov, kstar, mean_real, mean_infill, star_stats1, star_stats2 = proc_discrete_revised_dt(x_locs,y_locs,raw_image,bimage,Np=Np,widx=widx,widy=widy,ftype=ftype, tilex=tilex,tiley=tiley,seed=seed,rlim=rlim,ndraw=ndraw);

        k = .!kstar 
        
        
        push!(k_list, k)
        push!(kstar_list, kstar)
    
    end
    
    return r_list, k_list, kstar_list
    
end 