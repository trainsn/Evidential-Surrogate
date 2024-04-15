      program main
      implicit real*8 (a-h,o-z)
      include "parame.h"
      parameter (n=npara,m=14*n)
      parameter (n1=n  , n2=2*n, n3=3*n, n4=4*n)
      parameter (n5=5*n, n6=6*n, n7=7*n, n8=8*n)
      parameter (n9=9*n, n10=10*n, n11=11*n, n12=12*n)
      parameter (n13=13*n)

      external deriv
      dimension t(m),wk(m,5)
      dimension b1m(n),gbn(n),gbnstar(n),b1_star(n),dlcc(n)

      dimension z(n),y(n),aa2(n)
      dimension w1(n),u1(n),u2(n)
      integer*4 iseed,itime_array(3)


* 1:[R], 2:[RL], 3:[G], 4:[Ga], 5:[Gbg], 6:[Gd]
* 7:[c24m], 8:[c42], 9:[c42a], 10:[B1m], 11:[S20m], 12:[Cla4a]
* 13:[Pol], 14:[b]

c      write(*,*) '  enter the status of the initial condition'
c      write(*,*) '  0 = initial at t=0.'
c      write(*,*) '  1 = flip Ligand, and restart from a restart file'
c      write(*,*) '  2 = no flip, and restart from a restart file'
c     read(*,*) irest
      irest = 0


*  set up parameters 

       pi = 4.d0 * datan(1.d0)
       h = 2.d0 * pi / dfloat(n)
c       h =  pi / dfloat(n)
       zmid = 0.d0
       rmid = 0.d0

      call input_const(kmaxh,imode,ind_time_ode,ht)

      beta = 1.d0
      dpower = 1000.d0
      delta = 1.d0
      gamma = 1.d0


      SA = 0.d0
      do j = 1, n
        e = ( dfloat(j) - 1.d0 ) * h
        xJj(j)=dsqrt(rad_r*rad_r*dcos(e)*dcos(e)+rad_z
     &           *rad_z*dsin(e)*dsin(e))
     &           *50.2655d0/(2.d0*pi*rad_r)
        SA = SA + xJj(j)*h
*************************************
c        e = ( dfloat(j) - 0.5d0 ) * h
c        xJj(j) = 2.d0*pi*rad_r*dsin(e)*dsqrt( (rad_r*dcos(e))**2.d0
c     *            + (rad_z*dsin(e))**2.d0 )
c        SA = SA + xJj(j)*h
*************************************

      enddo
  
c      print*,'SA, V =',SA,V
      
*  initial conditions

      time = 0.d0 
      istart1 = 0
      stre = pho_slopez
      
      if( irest .eq. 0 ) then

c        write(*,*)' '
c        write(*,*)'enter (pho_slopez,pho_sloper) '
c     read(*,*)pho_slopez,pho_sloper
         pho_slopez = 1
         pho_sloper = 0
        t0 = dsqrt(pho_slopez**2.d0+pho_sloper**2.d0)
        pho_slopez = stre * pho_slopez / t0
        pho_sloper = stre * pho_sloper / t0
c        write(*,*)' '
c        write(*,*)pho_ini,pho_slopez,pho_sloper

         do j = 1, n
            t(j)    = r_tot/SA
            t(n+j)  = 0.d0
            t(n2+j) = g_tot/SA
            t(n3+j) = 0.d0
            t(n4+j) = 0.d0
            t(n5+j) = 0.d0
            t(n6+j) = 0.d0
            t(n7+j) = c42_tot/SA
            t(n8+j) = 0.d0
            t(n9+j) = 0.d0
            t(n10+j) = 0.d0
            t(n11+j) = 0.01d0
            t(n12+j) = 1.d0
            t(n13+j) = 1.d0
            b1m(j) = t(n9+j)
            gbn(j) = t(n4+j) / (g_tot/SA)
         enddo
      else

*    restart from a restart file
         open(unit=60,file='restart6',form='unformatted',
     *        status='unknown')
         rewind 60
         read(60)time
         read(60)pho_ini,pho_slopez,pho_sloper

         if( irest .eq. 1 ) then
c           write(*,*)' pho_slopez =',pho_slopez
c           write(*,*)' pho_sloper =',pho_sloper
c           write(*,*)' '
           stre = dsqrt(pho_slopez**2.d0+pho_sloper**2.d0)
c           write(*,*)'enter (pho_slopez,pho_sloper) '
c     read(*,*)pho_slopez,pho_sloper
           pho_slopez = 1
           pho_sloper = 0
           t0 = dsqrt(pho_slopez**2.d0+pho_sloper**2.d0)
           pho_slopez = stre * pho_slopez / t0
           pho_sloper = stre * pho_sloper / t0
         endif

         do j = 1, n
            read(60)t(j)
            read(60)t(n+j)
            read(60)t(n2+j)
            read(60)t(n3+j)
            read(60)t(n4+j)
            read(60)t(n5+j)
            read(60)t(n6+j)
            read(60)t(n7+j)
            read(60)t(n8+j)
            read(60)t(n9+j)
            read(60)t(n10+j)
            read(60)t(n11+j)
            read(60)t(n12+j)
            read(60)t(n13+j)
            b1m(j) = t(n9+j)
            gbn(j) = t(n4+j) / (g_tot/SA)
         enddo
         close(60)
      endif


* start writing da_cont

      call B1starGbnstar(b1m,b1_star,gbn,gbnstar,dlcc)

      iout_lig  = 11
      iout_cont = 12
      iout_ind  = 13
      open(iout_cont,file='da_cont')
      open(iout_ind,file='da_ind')
      open(iout_lig,file='da_ligand')
      write(iout_ind,*) n
      write(iout_ind,*) time
      do j = 1, m
         write(iout_cont,*) t(j)
      enddo
      do j = 1, n
         write(iout_cont,*) gbn(j)
      enddo
      do j = 1, n
         write(iout_cont,*) dlcc(j)
      enddo
      do j = 1, n
         write(iout_cont,*) gbnstar(j)
      enddo
      do j = 1, n
         write(iout_cont,*) b1_star(j)
      enddo
      close(iout_cont)
      close(iout_ind)
      close(iout_lig,status='delete')


*  set up noise, frequency

      do j = 1, n
        e = dfloat(j-1) * h
        zz = - rad_z * dcos(e)
        rr =   rad_r * dsin(e)
        z(j) = e
        aa2(j) = pho_ini + pho_slopez * (zz - zmid)
     *                   + pho_sloper * (rr - rmid)
      enddo

      ddt = 1.d0 / float(kfreq_time)
      ddx = ( z(n) - z(1) ) / float(kfreq_space-1)

      kt_noise = int( ddt / ht ) + 1
      if (kt_noise.lt.5)then
c        print*, 'WARNING: ht too big'
      endif

      kx_noise = int( ddx / h ) + 1
      if (kx_noise.lt.3)then
c        print*, 'WARNING: h too big'
      endif

      y(1) = z(1)
      do j = 1, kfreq_space
        y(j) = y(1) + ddx * float(j-1)
      enddo  


c      print*,'pho_ini =',pho_ini,' pho_slope =',pho_slopez
c     *      ,' pho_sloper =',pho_sloper
c      print*,'T freq =',kfreq_time,', x freq =',kfreq_space

      call itime(itime_array)
      iseed = itime_array(1)+itime_array(2)+itime_array(3)
      iseed = 6909 * iseed + 1
      call srand(iseed)


      time = 0.d0
      var_noise = 3.0d0 
      residual = 0.d0
      ineg_sig = 0

c      print*, 'var_noise =',var_noise
      
************************    time evolution begins     ************************

      do i = 1, kmaxh

         if (tmax-time<ht) then
            ht = tmax-time
            kmaxh = i
         endif
***     generate noise in space every 'kt_noise' step

c        kk0 = mod(i,kt_noise)
c        if ( kk0 .eq. 0 )then
c          do j = 1, n
c            u1(j) = rand()
c          enddo
c          do j = 1, n
c            u2(j) = rand()
c          enddo

c          call fixfreq_noise(z,y,u1,u2,w1)
c          call fixfreq_noise_log(aa2,z,y,u1,u2,w1)

c          call white_noise(u1,u2,w1)
c          call white_noise_log(aa2,u1,u2,w1)
          
c          do j = 1, n
c            wnoise(j) = w1(j) 
c          enddo

c        endif


***   Runge-Kutta in time
         call cn(m,ht,time,t,deriv,wk)

         kkk = mod(i,imode)
         
         if (kkk.eq.0 .or. i.eq.kmaxh) then
c            print*, time, residual,t(n8+1),t(n8+n),t(n12+n),t(n13+n)
            open(iout_cont,file='da_cont', access='append')
            open(iout_ind,file='da_ind', access='append')
            
            write(iout_ind,*) time
            do j = 1, m
               write(iout_cont,*) t(j)
            enddo
            do j = 1, n
              b1m(j) = t(n9+j)
              gbn(j) = t(n4+j) / (g_tot/SA)
            enddo

            call B1starGbnstar(b1m,b1_star,gbn,gbnstar,dlcc)

            do j = 1, n
               write(iout_cont,*) gbn(j)
            enddo
            do j = 1, n
               write(iout_cont,*) test(j)
            enddo
            do j = 1, n
               write(iout_cont,*) gbnstar(j)
            enddo
            do j = 1, n
               write(iout_cont,*) b1_star(j)
            enddo
            close(iout_cont)
            close(iout_ind)

c            open(iout_lig,file='da_ligand', access='append')
c             do j = 1, n
c               write(iout_lig,*) test(j),test1(j),test2(j),test3(j)
c             enddo
c            close(iout_lig)

         endif


         kkk2 = mod(i,10*imode)
         if(kkk2.eq.0)then
            call save(t,time)
         endif

         if( time.ge.tmax) then
c     .or. residual.le.1.d-4)then
c            write(*,*) time, residual
            goto 1001
         endif

      enddo
************************    end of time evolution      ************************

 1001 continue

* save data

      call save(t,time)


      stop
      end



*******************************************************************************
*                                                                             *
*                             subroutines                                     *
*                                                                             *
*******************************************************************************


        subroutine save(t,time)
        implicit real*8 (a-h,o-z)
        include "parame.h"
        parameter (n=npara,m=14*n)
        parameter (n1=n  , n2=2*n, n3=3*n, n4=4*n)
        parameter (n5=5*n, n6=6*n, n7=7*n, n8=8*n)
        parameter (n9=9*n, n10=10*n, n11=11*n, n12=12*n)
        parameter (n13=13*n)

        dimension t(m)

        print*,'save'
        open(unit=60,file='restart6',form='unformatted',
     *    status='unknown')
        rewind 60
        write(60)time
        write(60)pho_ini,pho_slopez,pho_sloper
        do j = 1, n
          write(60)t(j)
          write(60)t(n+j)
          write(60)t(n2+j)
          write(60)t(n3+j)
          write(60)t(n4+j)
          write(60)t(n5+j)
          write(60)t(n6+j)
          write(60)t(n7+j)
          write(60)t(n8+j)
          write(60)t(n9+j)
          write(60)t(n10+j)
          write(60)t(n11+j)
          write(60)t(n12+j)
          write(60)t(n13+j)
        enddo
        close(60)  


        return
        end


      subroutine white_noise(u1,u2,w1)
      implicit real*8 (a-h,o-z)
      include "parame.h"
      parameter (n=npara)
      dimension u1(n),u2(n),w1(n)

      do j = 1, n
c        if(u1(j).le.1.d-12) u1(j)=1.d-12
        w1(j) = sqrt( -2.d0*log(u1(j)) ) * cos(2.d0*pi*u2(j)) 
        w1(j) = var_noise * w1(j)
c        w1(j) = 2.d0*u1(j) - 1.d0
      enddo

c      stop

      return
      end


      subroutine white_noise_log(aa2,u1,u2,w1)
      implicit real*8 (a-h,o-z)
      include "parame.h"
      parameter (n=npara)
      dimension u1(n),u2(n),w1(n)
      dimension aa2(n)
      
      t0 = var_noise**2.d0 + 1.d0
      sig2 = sqrt(log(t0))
      do j = 1, n
        xmu2 = log(aa2(j)/sqrt(t0))
c        if(u1(j).le.1.d-12) u1(j)=1.d-12
        w1(j) = dsqrt( -2.d0*log(u1(j)) ) * cos(2.d0*pi*u2(j)) 
        w1(j) = exp( xmu2 + sig2*w1(j) )
        w1(j) = w1(j)/aa2(j) - 1.d0
      enddo

c      write(*,*)sig2, xmu2
c      stop

      return
      end


      subroutine fixfreq_noise(z,y,u1,u2,w1)
      implicit real*8 (a-h,o-z)
      include "parame.h"
      parameter (n=npara)
      dimension z(n),y(n),w1(n),u1(n),u2(n)

      k = 1
      j = 1
      do 20 while ( k.le.kfreq_space-1 )
         do 30 while (  z(j).le.y(k+1)+1.d-8 .and. j.le.n )
c            w1(j) = 2.d0*u1(k) - 1.d0
c            if(u1(k).le.1.d-12) u1(k)=1.d-12
            w1(j) = sqrt( -2.d0*log(u1(k)) ) * cos(2.d0*pi*u2(k))
            w1(j) = var_noise * w1(j)
            j = j + 1
30       enddo
         k = k + 1
20    enddo


      return
      end



      subroutine fixfreq_noise_log(aa2,z,y,u1,u2,w1)
      implicit real*8 (a-h,o-z)
      include "parame.h"
      parameter (n=npara)
      dimension z(n),y(n),w1(n),u1(n),u2(n)
      dimension aa2(n),w2(n)

      t0 = var_noise**2.d0 + 1.d0
      sig2 = sqrt(log(t0))
      do j = 1, n
        xmu2 = log(aa2(j)/sqrt(t0))
c        if(u1(j).le.1.d-12) u1(j)=1.d-12
        w2(j) = dsqrt( -2.d0*log(u1(j)) ) * cos(2.d0*pi*u2(j)) 
        w2(j) = exp( xmu2 + sig2*w2(j) )
        w2(j) = w2(j)/aa2(j) - 1.d0
      enddo

      k = 1
      j = 1
      do 20 while ( k.le.kfreq_space-1 )
         do 30 while (  z(j).le.y(k+1)+1.d-8 .and. j.le.n )
            w1(j) = w2(k)
            j = j + 1
30       enddo
         k = k + 1
20    enddo


      return
      end

