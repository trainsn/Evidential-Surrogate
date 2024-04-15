       parameter(npara = 400)

       common/const_real/ h,SA,V,pi,zmid,rmid,residual,tmax
       common/para/ gamma,beta,dpower,delta,var_noise
       common/func/ xJj(npara),test(npara),wnoise(npara)
       common/func/ test1(npara),test2(npara),test3(npara)

       common/biopara0/ rad_r,rad_z
       common/biopara1/ p_rl,p_rlm,p_rd0,p_rs,p_rd1
       common/biopara2/ p_ga,p_g1,p_gd
       common/biopara3/ p_24cm0,p_24cm1,p_24mc,p_24d
       common/biopara4/ p_42a,p_42d
       common/biopara5/ p_b1cm,p_b1mc
       common/biopara10/ p_s20cm,p_s20mc
       common/biopara6/ p_cla4a,p_cla4d
       common/biopara11/ p_0,p_1,p_2,p_3,p_4,pol_ss
       common/biopara12/ pol_del,power_m,pol_eps,power_n 
       common/biopara7/ c24_tot,b1_tot,c42_tot,g_tot,r_tot,s20_tot
       common/biopara8/ dlc,hpower,pho_ini,pho_sloper,pho_slopez
       common/biopara9/ dR,dRL,dG,dGa,dGbg,dGd
       common/biopara10/dC24m,dC42,dC42a,dB1m
       common/biopara11/ GbgnR, Gbgnq
	 

       common/inte/ kfreq_time,kfreq_space,ineg_sig
