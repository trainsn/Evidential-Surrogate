      subroutine deriv(mm,time,t,f)
      implicit real*8 (a-h,o-z)
      include "parame.h"
      parameter (n=npara,m=14*npara)
      parameter (n1=n  , n2=2*n, n3=3*n, n4=4*n)
      parameter (n5=5*n, n6=6*n, n7=7*n, n8=8*n)
      parameter (n9=9*n, n10=10*n, n11=11*n, n12=12*n)
      parameter (n13=13*n)

      dimension t(mm), f(mm)

      dimension z(n),ze(n),zee(n)
      dimension r(n),re(n),ree(n)
      dimension se(n),see(n)

      dimension w(n),w_sur(n),w1(n),w2(n)
      dimension alpha(n),aa(n)
      dimension rep(n),rl(n),g(n),ga(n),dlcc(n)
      dimension gbn(n),gd(n),gbg(n),gbnstar(n)
      dimension c24m(n),c42(n),c42a(n),b1m(n),b1_star(n)
      dimension cla4a(n),s20m(n),pol(n),b(n)
      dimension pol_s(n)

      data jfrst /1/
      
      save

      if (jfrst.ne.1) go to 20

      write(12,*) p_rl,p_rlm,p_rd0,p_rs,p_rd1
      write(12,*) p_ga,p_g1,p_gd
      write(12,*) p_24cm0,p_24cm1,p_24mc,p_24d
      write(12,*) p_42a,p_42d
      write(12,*) p_b1cm,p_b1mc
      write(12,*) p_s20cm,p_s20mc
      write(12,*) p_cla4a,p_cla4d
      write(12,*) p_0,p_1,p_2,p_3,p_4,pol_ss
      write(12,*) pol_del,power_m,pol_eps,power_n
      write(12,*) c24_tot,b1_tot,c42_tot,g_tot,r_tot,s20_tot
      write(12,*) dlc,hpower,pho_ini, pho_sloper,pho_slopez
      write(12,*) dR,dRL,dG,dGa,dGbg,dGd
      write(12,*) dC24m,dC42,dC42a,dB1m
      write(12,*) GbgnR, Gbgnq
      write(12,*) gamma,beta,dpower,delta
      

      do j = 1, n
         e = ( dfloat(j) - 1.d0 ) * h 
         alpha(j) = e
         r(j) =   rad_r * dsin(e)
         z(j) = - rad_z * dcos(e)

         re(j) =  rad_r * dcos(e)
         ze(j) =  rad_z * dsin(e)

         ree(j) = - rad_r * dsin(e)
         zee(j) =   rad_z * dcos(e)

         se(j) = dsqrt(re(j)*re(j) + ze(j)*ze(j))
         see(j) = (re(j)*ree(j)+ze(j)*zee(j))/se(j)

         aa(j) = pho_ini + pho_slopez * (z(j) - zmid)
     *                   + pho_sloper * (r(j) - rmid)

      enddo

      iout_xyp = 14 
      open(iout_xyp,file = 'da_xyp')
      do i = 1, n
         write(iout_xyp,*) r(i),z(i),aa(i),alpha(i)
      enddo
      close(iout_xyp)

      jfrst = 0

 20   continue

*  add noise to the ligand

c      do j = 1, n
c         aa(j) = ( pho_ini + pho_slopez * (z(j) - zmid) 
c     *                     + pho_sloper * (r(j) - rmid) )
c     &           * (1.d0 + wnoise(j))

c         if (aa(j).lt.0.d0)then
c            aa(j) = 0.d0
c            ineg_sig = 1
c            print*,' negative signal'
c            stop
c         endif

c      enddo

c      do j = 1, n
c        test3(j) = aa(j)
c        print*,j,test(j),wnoise(j)
c      enddo
c      stop


*  read in variables

      do j = 1, n
         rep(j) = t(j) 
          rl(j) = t(n+j)
           g(j) = t(n2+j)
          ga(j) = t(n3+j)
         gbg(j) = t(n4+j)
          gd(j) = t(n5+j)
        c24m(j) = t(n6+j)
         c42(j) = t(n7+j)
        c42a(j) = t(n8+j)
         b1m(j) = t(n9+j)
        s20m(j) = t(n10+j)
       cla4a(j) = t(n11+j)
         pol(j) = t(n12+j)
           b(j) = t(n13+j)
      enddo


*  reaction terms

        c24m_inte = 0.d0
         b1m_inte = 0.d0
        s20m_inte = 0.d0
         gbn_inte = 0.d0
        c42a_inte = 0.d0
         pol_inte = 0.d0

        do j = 1, n
c           gd(j) = g_tot/SA - g(j) - ga(j)
c          gbg(j) = g_tot/SA - g(j)
          gbn(j) = gbg(j) / (g_tot/SA)

          c24m_inte = c24m_inte + xJj(j)*c24m(j)*h
          c42a_inte = c42a_inte + xJj(j)*c42a(j)*h
           b1m_inte = b1m_inte  + xJj(j)*b1m(j)*h
          s20m_inte = s20m_inte + xJj(j)*s20m(j)*h
           gbn_inte = gbn_inte  + xJj(j)*gbn(j)*h
           pol_inte = pol_inte  + xJj(j)*pol(j)*h
           pol_s(j) = c42a_inte
        enddo

        c42a_tilde = c42a_inte / SA
        pol_tilde  = pol_inte / SA

************************
        p_42s   = p_rs
        p_42deg = p_rd0
************************

      call B1starGbnstar(b1m,b1_star,gbn,gbnstar,dlcc)


      do j = 1, n

*  R
         t1 = c42a(j) / (c42a_tilde+1.d-8)
         if(c42a_tilde.le.1.d-8) t1 = 1.d0
c         t1 = pol(j) / (pol_tilde+1.d-8)
c         if(pol_tilde.le.1.d-8) t1 = 1.d0

         a1 = t1

c         t1 = 1.d0

         f(j) = -p_rl*aa(j)*rep(j) + p_rlm*rl(j)
         f(j) = f(j) - p_rd0*rep(j) + p_rs*t1
*  RL
         f(n+j) = p_rl*aa(j)*rep(j) - (p_rlm+p_rd1)*rl(j)

*  G
         f(n2+j) = -p_ga*rl(j)*g(j) + p_g1*gd(j)*gbg(j)

*  Ga
         f(n3+j) = p_ga*rl(j)*g(j) - p_gd*ga(j)

*  Gbg
         f(n4+j) = p_ga*rl(j)*g(j) - p_g1*gd(j)*gbg(j)

*  Gd
         f(n5+j) = p_gd*ga(j) - p_g1*gd(j)*gbg(j)

*  C24m
         c24c = ( c24_tot - c24m_inte ) / V
         f(n6+j) = p_24cm0*gbnstar(j)*c24c + p_24cm1*c24c*b1_star(j)
         f(n6+j) = f(n6+j) - p_24mc*c24m(j) - p_24d*cla4a(j)*c24m(j)

*  C42
         f(n7+j) = -p_42a*c24m(j)*c42(j) + p_42d*c42a(j)
c     &             + a1*p_42s - p_42deg*c42(j)
c     &             + p_42s - p_42deg*c42(j)

*  C42a
         f(n8+j) = p_42a*c24m(j)*c42(j) - p_42d*c42a(j)
c     &             - p_42deg*c42a(j)

*  B1m
         b1c = ( b1_tot - b1m_inte ) / V
         f(n9+j) = p_b1cm*c42a(j)*b1c - p_b1mc*b1m(j)

*  S20m
         s20c = ( s20_tot - s20m_inte ) / V
         f(n10+j) = p_s20cm*c42a(j)*s20c - p_s20mc*s20m(j)

*  Cla4a
         c42astar_tot = c42a_tilde
         f(n11+j) = p_cla4a*c42astar_tot - p_cla4d*cla4a(j)

*  polarisome
*********************
         pol_s(j) = pol_s(j) / (c42a_inte+1.d-8)
*********************
c          r0 = a1 * gbnstar(j)
          r0 = c42a(j)/(c42a_tilde+1.d-8)
          if(c42a_tilde.le.1.d-8) r0 = 1.d0
c          r0 = pol_s(j)
         pp1 = 1.d0 / (1.d0+(pol_del*r0)**(-power_m))

         f(n12+j) =  p_0*pp1  
     & + p_1 * 1.d0/(1.d0+(pol_eps*pp1*pol(j))**(-power_n))
     & - ( p_2 + p_3*b(j) ) * pol(j)

*  b
         f(n13+j) = p_4 * ( pol_tilde - pol_ss ) * b(j)
****************************
         test(j)  = r0
         test1(j) = pp1
         test2(j) = pol(j)
****************************
c         print*,j,f(n13+j),pol_tilde
      enddo

c      stop
c      do j = 1, n
c         print*,f(n5+j)
c         print*, p_gd*ga(j) , p_g1*gd(j)*gbg(j)
c         print*, ga(j) , gbg(j), gd(j), p_gd, p_g1
c       print*,j,f(j)
c       print*,h
c      enddo
c      stop

*  diffusions 


      dfold = 1.d0
*  R
      call sur_dif(n,n,h,rep,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(j) = f(j) + dR * w_sur(j)
      enddo

*  RL
      call sur_dif(n,n,h,rl,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n+j) = f(n+j) +  dRL * w_sur(j)
      enddo

*  G
      call sur_dif(n,n,h,g,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n2+j) = f(n2+j) + dG * w_sur(j)
      enddo

*  Ga
      call sur_dif(n,n,h,ga,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n3+j) = f(n3+j) + dGa * w_sur(j)
      enddo

*  Gbg
      call sur_dif(n,n,h,gbg,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n4+j) = f(n4+j) + dGbg * w_sur(j)
      enddo

*  Gd
      call sur_dif(n,n,h,gd,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n5+j) = f(n5+j) + dGd * w_sur(j)
      enddo

*  C24m
      call sur_dif(n,n,h,c24m,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n6+j) = f(n6+j) + dC24m * w_sur(j)
      enddo

*  C42
      call sur_dif(n,n,h,c42,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n7+j) = f(n7+j) + dC42 * w_sur(j)
      enddo

*  C42a
      call sur_dif(n,n,h,c42a,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n8+j) = f(n8+j) + dC42a * w_sur(j)
      enddo

*  B1m
      call sur_dif(n,n,h,b1m,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n9+j) = f(n9+j) + dB1m * w_sur(j)
      enddo

*  S20m 
      call sur_dif(n,n,h,s20m,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n10+j) = f(n10+j) + dlc * w_sur(j)
      enddo

*  Pol
      call sur_dif(n,n,h,pol,w_sur,r,re,se,see,w1,w2)
      do j = 1, n
         f(n12+j) = f(n12+j) + dlc * w_sur(j)
      enddo

*********  residual  *************
      residual = 0.d0
      do j = 1, mm
       residual = max( residual , abs(f(j)) )
c       print*,j,residual
      enddo
**********************************

c      stop

      return
      end


      subroutine sur_dif(n,m,h,w,w_sur,r,re,se,see,w1,w2)
      implicit real*8 (a-h,o-z)
      dimension w(n),w_sur(n)
      dimension se(m),see(m),r(m),re(m)
      dimension w1(m),w2(m)

      do j = 1, n
         if(j.eq.n) then
            j1=1
         else
            j1=j+1
         endif
         if(j.eq.1) then
            j_1=n
         else
            j_1=j-1
         endif

         w1(j) = (w(j1) - w(j_1))/(2.d0*h)
         w2(j) = (w(j1)- 2.d0*w(j) + w(j_1))/(h*h)
      enddo

      do j = 1, n
         w_sur(j) = (w2(j)*se(j)-see(j)*w1(j))/(se(j)**3.d0)
c         temp = (re(j)/se(j))*(w1(j)/se(j))/r(j)
c         w_sur(j) = w_sur(j) + temp
      enddo

      return
      end

