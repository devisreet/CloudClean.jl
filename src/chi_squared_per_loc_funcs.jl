using LinearAlgebra

export condCovEst_wdiag_dt   ## CHANGED 
export condCovEst_wdiag_revised_dt  ## CHANGED
export condCovEst_wdiag_svd

export chisquared_xreal_ctot  #CHI_2
export chisquared_xinfill_ctot #CHI_2
export chisquared_xinfill_cinfill  #CHI_2
export chisquared_xreal_cinfill  #CHI_2


"""
    condCovEst_wdiag(cov_loc,μ,km,data_in;Np=33,export_mean=false,n_draw=0) -> out

Using a local covariance matrix estimate `cov_loc` and a set of known ("good") pixels `km`, this function computes a prediction for the mean value of masked pixels and the covariance matrix of the masked pixels. The output list can conditionally include the mean reconstruction and draws from the distribution of reconstructions.

# Arguments:
- `cov_loc`: local covariance matrix
- `μ`: vector containing mean value for each pixel in the patch
- `km`: unmasked pixels
- `data_in`: input image

# Keywords:
- `Np`: size of local covariance matrix in pixels (default 33)
- `export_mean`: when true, returns the mean conditional prediction for the "hidden" pixels (default false)
- `n_draw`: when nonzero, returns that number of realizations of the conditional infilling (default 0)

# Outputs:
- `out[1]`: input image returned with masked pixels replaced with mean prediction
- `out[2]`: input image returned with masked pixels replaced with a draw from the predicted distribution
"""

function condCovEst_wdiag_dt(cov_loc,μ,km,data_in;Np=33,export_mean=false,n_draw=0,seed=2022)
    k = .!km
    kstar = km
    cov_kk = Symmetric(cov_loc[k,k])
    cov_kkstar = cov_loc[k,kstar];
    cov_kstarkstar = cov_loc[kstar,kstar];
    icov_kkC = cholesky(cov_kk)
    icovkkCcovkkstar = icov_kkC\cov_kkstar
    predcovar = Symmetric(cov_kstarkstar - (cov_kkstar'*icovkkCcovkkstar)) #kstar by kstar
    ipcovC = cholesky(predcovar)

    @views uncond_input = data_in[:]
    @views cond_input = data_in[:].- μ

    kstarpredn = (cond_input[k]'*icovkkCcovkkstar)'
    kstarpred = kstarpredn .+ μ[kstar]

    out = []
    if export_mean
        mean_out = copy(data_in)
        mean_out[kstar] .= kstarpred
        push!(out,mean_out)
    end
    if n_draw != 0
        sqrt_cov = ipcovC.U
        noise = randn(n_draw,size(sqrt_cov)[1])*sqrt_cov

        draw_out = repeat(copy(data_in)[:],outer=[1 n_draw])
        draw_out[kstar,:] .= repeat(kstarpred,outer=[1 n_draw]) .+ noise'
        push!(out,draw_out)
    end
    return predcovar
end

function condCovEst_wdiag_revised_dt(cov_loc,μ,km,data_in;Np=33,export_mean=false,n_draw=0,seed=2022)
    k = .!km
    #print(count(k))
    kstar = km
    #print(size(kstar))
    #print(count(kstar))
    cov_kk = Symmetric(cov_loc[k,k])
    cov_kkstar = cov_loc[k,kstar];
    cov_kstarkstar = cov_loc[kstar,kstar];
    #print(size(cov_kstarkstar))
    icov_kkC = cholesky(cov_kk)
    icovkkCcovkkstar = icov_kkC\cov_kkstar
    predcovar = Symmetric(cov_kstarkstar - (cov_kkstar'*icovkkCcovkkstar))
    #print(size(predcovar))
    ipcovC = cholesky(predcovar)

    @views uncond_input = data_in[:]
    @views cond_input = data_in[:].- μ

    kstarpredn = (cond_input[k]'*icovkkCcovkkstar)'
    kstarpred = kstarpredn .+ μ[kstar]
    
    #print(size(μ)) --> flattened version of mean

    out = []
    if export_mean
        mean_out = copy(data_in)
        mean_out[kstar] .= kstarpred
        push!(out,mean_out)
        #print(size(mean_out))
    end
    if n_draw != 0
        sqrt_cov = ipcovC.U
        noise = randn(n_draw,size(sqrt_cov)[1])*sqrt_cov

        draw_out = repeat(copy(data_in)[:],outer=[1 n_draw])
        draw_out[kstar,:] .= repeat(kstarpred,outer=[1 n_draw]) .+ noise'
        push!(out,draw_out)
    end
    return predcovar, out
end


function condCovEst_wdiag_svd(cov_loc,μ,km,data_in;Np=33,export_mean=false,n_draw=0,seed=2022, use_svd=false, low_rank=false)
    k = .!km
    kstar = km
    cov_kk = Symmetric(cov_loc[k,k])
    cov_kkstar = cov_loc[k,kstar];
    cov_kstarkstar = cov_loc[kstar,kstar];

    if use_svd
        #u_covkk, s_covkk, v_covkk = svd(Matrix(cov_kk), full=true, alg=LinearAlgebra.QRIteration())
        u_covkk, s_covkk, v_covkk = svd(cov_kk, full=true)
        
        if low_rank
            cutoff = 0.1 
            indxs = s_covkk .> cutoff 
           # println([minimum(s_covkk), length(s_covkk), sum(indxs)])
            icov_kkc = v_covkk[:, indxs]*diagm(1 ./ s_covkk[indxs])*u_covkk[:, indxs]'
        else 
            icov_kkc = v_covkk*diagm(1 ./ s_covkk)*u_covkk'
        end
        
        icovkkCcovkkstar = icov_kkc*cov_kkstar
        
        predcovar = Symmetric(cov_kstarkstar - (cov_kkstar'*icovkkCcovkkstar))
        #u_pcovC, s_pcovC, v_pcovC = svd(Matrix(predcovar), full=true, alg=LinearAlgebra.QRIteration())  #may need low rank here too
        u_pcovC, s_pcovC, v_pcovC = svd(predcovar, full=true)
        ipcovC = v_pcovC*diagm(1 ./ s_pcovC)*u_pcovC'
       # println([minimum(s_pcovC)])
        # print(eigmin(predcovar))
    else
        icov_kkC = cholesky(cov_kk)
        icovkkCcovkkstar = icov_kkC\cov_kkstar
        
        predcovar = Symmetric(cov_kstarkstar - (cov_kkstar'*icovkkCcovkkstar))
        ipcovC = cholesky(predcovar)
    end

    @views uncond_input = data_in[:]
    @views cond_input = data_in[:].- μ

    kstarpredn = (cond_input[k]'*icovkkCcovkkstar)'
    kstarpred = kstarpredn .+ μ[kstar]

    out = []
    if export_mean
        mean_out = copy(data_in)
        mean_out[kstar] .= kstarpred
        push!(out,mean_out)
    end
    if n_draw != 0
        if use_svd
            sqrt_cov = u_pcovC*diagm(sqrt.(s_pcovC))*v_pcovC'
            #noise = (sqrt_cov*randn(n_draw,size(sqrt_cov)[1])')' #flipping sqrt mult dir
            
            noise = randn(n_draw,size(sqrt_cov)[1])*sqrt_cov #first way
        else
            sqrt_cov = ipcovC.U
            noise = randn(n_draw,size(sqrt_cov)[1])*sqrt_cov
        end
        
        # sqrt_cov = ipcovC.U
        #noise = randn(n_draw,size(sqrt_cov)[1])*sqrt_cov

        draw_out = repeat(copy(data_in)[:],outer=[1 n_draw])
        draw_out[kstar,:] .= repeat(kstarpred,outer=[1 n_draw]) .+ noise'
        push!(out,draw_out)
    end

    return out
end

"
Compute Various Chi-Squared Values 
chisquared_xreal_ctot --> 
chisquared_xinfill_ctot --> 
chisquared_xinfill_cinfill -->
chisquared_xreal_cinfill ---> xreal is x0, sub
"

function chisquared_xreal_ctot(img, raw_img, icov, mean_real, Np, cenx, ceny)
    dv = (Np-1)÷2;
    x0_centered = img.-median(raw_img)
    
    x0_centered_flat = vec(x0_centered[(cenx-dv):(cenx+dv),(ceny-dv):(ceny+dv)])+vec(mean_real)
    
    chi_squared = x0_centered_flat'*(icov\x0_centered_flat)/Np^2;
    return chi_squared
end

function chisquared_xinfill_ctot(star_stats2, icov, mean_real, Np, cenx, ceny, infill_num)
    dv = (Np-1)÷2;
    xinfill_Np = star_stats2[:, infill_num]
    xinfill_Np_minus_mean = xinfill_Np-vec(mean_real)
    chi_squared = xinfill_Np_minus_mean'*(icov\xinfill_Np_minus_mean)/Np^2;
    return chi_squared
end

function chisquared_xinfill_cinfill(star_stats2, kstar, ipredcov, mean_infill, cenx, ceny, Np, infill_num)
    dv = (Np-1)÷2;
    infill_pix = count(kstar);
    xinfill = vec(star_stats2[:, infill_num][kstar]); 
    xinfill_minus_mean = xinfill-vec(mean_infill[kstar]); 
    chi_squared = xinfill_minus_mean'*(ipredcov\xinfill_minus_mean)/infill_pix;
    return chi_squared
end

function chisquared_xreal_cinfill(img, raw_img, kstar, ipredcov, mean_infill, median_img, cenx, ceny, Np)
    dv = (Np-1)÷2;
    infill_pix = count(kstar);
    # print(size(img[(cenx-dv):(cenx+dv),(ceny-dv):(ceny+dv)]))
    # print(median(raw_img))
    # print(size(mean_infill))
    
    xi_sub_centered = img[(cenx-dv):(cenx+dv),(ceny-dv):(ceny+dv)].-(median(raw_img).-mean_infill)
    xi_sub_centered_kstar = xi_sub_centered[kstar]
#   xi_sub =vec(img[(cenx-dv):(cenx+dv),(ceny-dv):(ceny+dv)][kstar]);
#   xi_sub_minus_mean =xi_sub-vec(mean_infill[kstar]);
#   xi_sub_minus_mean =xi_sub-(vec(median_img[kstar])-vec(mean_infill[kstar]));
 
    chi_squared = xi_sub_centered_kstar'*(ipredcov\xi_sub_centered_kstar)/infill_pix;
    return chi_squared
end  

function split_chisquared_contr_xinfill_ctot(star_stats2, kstar, icov, mean_real, Np, cenx, ceny, infill_num)
    dv = (Np-1)÷2;
    xinfill_Np = star_stats2[:, infill_num]
    xinfill_Np_minus_mean = xinfill_Np-vec(mean_real)
    
    xinfill_kstar_nonzero = copy(xinfill_Np_minus_mean)
    xinfill_kstar_nonzero[.!kstar].=0
    xinfill_kstar_zero = copy(xinfill_Np_minus_mean)
    xinfill_kstar_zero[kstar].=0
    
    # print(size(xinfill_kstar_nonzero))
    # print(size(xinfill_kstar_zero))
    xreal_contr = xinfill_kstar_zero'*(icov\xinfill_kstar_zero)/count(.!kstar)    #Np^2;
    xinfill_contr= xinfill_kstar_nonzero'*(icov\xinfill_kstar_nonzero)/count(kstar)    #Np^2;
    
    cross_term_n = sqrt(count(.!kstar)*count(kstar))
    xreal_xinfill_contr= xinfill_kstar_zero'*(icov\xinfill_kstar_nonzero)/cross_term_n;
    xinfill_xreal_contr = xinfill_kstar_nonzero'*(icov\xinfill_kstar_zero)/cross_term_n;
    
    full_chi_squared = (xinfill_contr*count(kstar) + xreal_contr*count(.!kstar) + (xinfill_xreal_contr + xreal_xinfill_contr)*cross_term_n)/Np^2

    org_chi_squared = xinfill_Np_minus_mean'*(icov\xinfill_Np_minus_mean)/Np^2;
    return [count(.!kstar),count(kstar),cross_term_n,cross_term_n], [xreal_contr, xinfill_contr, xreal_xinfill_contr, xinfill_xreal_contr], full_chi_squared
end   