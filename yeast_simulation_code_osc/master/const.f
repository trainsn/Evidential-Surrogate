      subroutine input_const(kmaxh,imode,ind_time_ode,ht)
      implicit real*8 (a-h,o-z)
      include "parame.h"

      open(20,file='input_bio_data',status='old')
      open(21,file='input_num_data',status='old')
      open(22,file='data_new',status='old')

 1500 format(80x)

      read(20,1500)
      read(20,*) rad_r,rad_z

      read(20,1500)
      read(20,*) p_rl,p_rlm,p_rd0,p_rs,p_rd1
      read(20,1500)
      read(20,*) p_ga,p_g1,p_gd
      read(20,1500)
      read(20,*) p_24cm0,p_24cm1,p_24mc,p_24d
      read(20,1500)
      read(20,*) p_42a,p_42d
      read(20,1500)
      read(20,*) p_b1cm,p_b1mc
      read(20,1500)
      read(20,*) p_s20cm,p_s20mc
      read(20,1500)
      read(20,*) p_cla4a,p_cla4d
      read(20,1500)
      read(20,*) p_0,p_1,p_2,p_3,p_4,pol_ss
      read(20,1500)
      read(20,*) pol_del,power_m,pol_eps,power_n
      read(20,1500)
      read(20,*) c24_tot,b1_tot,c42_tot,g_tot,r_tot,s20_tot
      read(20,1500)
      read(20,*) dlc,hpower,pho_ini,pho_sloper,pho_slopez
      read(20,1500)
      read(20,*) dR,dRL,dG,dGa,dGbg,dGd
      read(20,1500)
      read(20,*) dC24m,dC42,dC42a,dB1m
      read(20,1500)
      read(20,*) GbgnR, Gbgnq
      read(20,1500)
      read(20,*) kfreq_time,kfreq_space

c      V = 8.d0*pi/3.d0
c      SA = (4.d0*dsqrt(3.d0)*pi+9.d0)*2.d0*pi/9.d0

      V = 4.d0*pi/3.d0 * (rad_z)**3.d0
      SA = 4.d0 * pi * (rad_z)**2.d0
c      print*,SA


cccccccccccc     Read in modified parameters   ccccccccccccc

      read(22,*) p_rl,p_rlm,p_rd0,p_rs,p_rd1,
     *           p_ga,p_g1,p_gd,
     *           p_24cm0,p_24cm1,p_24mc,p_24d,
     *           p_42a,p_42d,
     *           p_b1cm,p_b1mc,
     *           p_cla4a,p_cla4d,
     *           c24_tot,b1_tot,c42_tot,g_tot,r_tot,
     *           Gbgnq,hpower,
     *           dR,dRL,dG,dGa,dGbg,dGd,
     *           dC24m,dC42,dC42a,dB1m

c      write(*,*) p_rl,p_rlm,p_rd0,p_rs,p_rd1,
c     *           p_ga,p_g1,p_gd,
c     *           p_24cm0,p_24cm1,p_24mc,p_24d,
c     *           p_42a,p_42d,
c     *           p_b1cm,p_b1mc,
c     *           p_cla4a,p_cla4d,
c     *           c24_tot,b1_tot,c42_tot,g_tot,r_tot,
c     *           Gbgnq,hpower,
c     *           dR,dRL,dG,dGa,dGbg,dGd,
c     *           dC24m,dC42,dC42a,dB1m

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      p_rs = p_rs/SA
      p_ga = p_ga*SA
      p_g1 = p_g1*SA
      p_24cm0 = p_24cm0*V/SA
      p_24cm1 = p_24cm1*V
      p_42a = p_42a*SA
      p_b1cm = p_b1cm*V
      p_s20cm = p_s20cm*V
      p_24d = p_24d * SA/3000.d0
       
      read(21,1500)
      read(21,*) ind_time_ode
      read(21,1500)
      read(21,*) ht, tmax, ht_pri
      kmaxh = nint(tmax/ht)+1
      imode = nint(ht_pri/ht)


      close (20)
      close (21)
      close (22)


      write(49,*) p_rl,p_rlm,p_rd0,p_rs,p_rd1
      write(49,*) p_ga,p_g1,p_gd
      write(49,*) p_24cm0,p_24cm1,p_24mc,p_24d
      write(49,*) p_42a,p_42d
      write(49,*) p_b1cm,p_b1mc
      write(49,*) p_cla4a,p_cla4d
      write(49,*) c24_tot,b1_tot,c42_tot,g_tot,r_tot
      write(49,*) Gbgnq,hpower
      write(49,*) dR,dRL,dG,dGa,dGbg,dGd
      write(49,*) dC24m,dC42,dC42a,dB1m



      return
      end
