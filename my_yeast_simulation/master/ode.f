        subroutine ab1(n,h,t,y,deriv,wk,istart)
        implicit real*8(a-h,o-z)
        dimension y(n),wk(n,3)
        external deriv
        save
        if(istart.eq.1) then
           call deriv(n,t,y,wk(1,3))
           call rk1(n,h,t,y,deriv,wk(1,1),istart)

c           do j=1,n
c              y(j)=y(j)+wk(j,3)*h
c           enddo
c           t=t+h

           istart=2
           call deriv(n,t,y,wk(1,2))
        else
           do j=1,n
              y(j)=y(j)+0.5*h*(3.0*wk(j,2)-wk(j,3))
              wk(j,3)=wk(j,2)
           enddo
           call deriv(n,t,y,wk(1,2))
           t=t+h
        endif
        return
        end





        subroutine am1(n,h,t,y,deriv,wk,istart)
        implicit real*8(a-h,o-z)
        dimension y(n),wk(n,5),l(6)
        equivalence (l(1),iyn),(l(2),iypn),(l(3),iypn1),
     1              (l(4),iypn2),(l(5),iypn3)
        external deriv
        save
        if(istart-1) 101,101,103
 101    continue
        a1=55.d0
        a2=-59.d0
        a3=37.d0
        a4=-9.d0
        a5=9.d0
        a6=19.d0
        a7=-5.d0
        a8=1.d0
        local=-3
        h24=h/24.d0
        do 102 i=1,6
        l(i)=i
 102    continue
        call deriv(n,t,y,wk(1,iypn3))
 103    if(local) 104,109,109
 104    call rk1(n,h,t,y,deriv,wk(1,1),istart)
        itmp=1-local
        index=l(itmp)
        do 105 i=1,n
        wk(i,index)=wk(i,1)
 105    continue
        local=local+1
        if(local) 108,106,106
 106    do 107 i=1,n
        wk(i,1)=y(i)
 107    continue
 108    return
 109    do 110 i=1,n
        y(i)=y(i)+h24*(a1*wk(i,iypn)+a2*wk(i,iypn1)+a3*
     1  wk(i,iypn2)+a4*wk(i,iypn3))
 110    continue
        t=t+h
        call deriv(n,t,y,wk(1,iypn3))
        do 111 i=1,n
        wk(i,iyn)=wk(i,iyn)+h24*(a5*wk(i,iypn3)+a6*wk(i,iypn)
     1  +a7*wk(i,iypn1)+a8*wk(i,iypn2))
        y(i)=wk(i,iyn)
 111    continue
        call deriv(n,t,y,wk(1,iypn3))
        do 112 i=1,4
        itmp=7-i
        l(itmp)=l(itmp-1)
 112    continue
        l(2)=l(6)
        return
        end

      
        subroutine rk1(n,h,t,y,deriv,wk,istart)
        implicit real*8(a-h,o-z)
        dimension a(4),b(4),c(4),y(n),wk(n,2)

        external deriv

        save
        if(istart-1) 101,101,103
 101    continue
        rt2=sqrt(2.d0)
        a(1)=0.5d0
        c(1)=a(1)
        c(4)=a(1)
        a(2)=0.5d0*(2.d0-rt2)
        c(2)=a(2)
        a(3)=0.5d0*(2.d0+rt2)
        c(3)=a(3)
        b(1)=2.d0
        b(4)=b(1)
        b(2)=1.d0
        b(3)=b(2)
        a(4)=1.d0/6.d0
        do 102 i=1,n
        wk(i,2)=0.d0
 102    continue
        qt=0.d0
        istart=2
        call deriv(n,t,y,wk)
 103    continue
        do 105 j=1,4
        do 104 i=1,n
        temp=a(j)*(wk(i,1)-b(j)*wk(i,2))
        w=y(i)
        y(i)=y(i)+h*temp
        temp=(y(i)-w)/h
        wk(i,2)=wk(i,2)+3.d0*temp-c(j)*wk(i,1)
 104    continue
        temp=a(j)*(1.d0-b(j)*qt)
        w=t
        t=t+h*temp
        temp=(t-w)/h
        qt=qt+3.d0*temp-c(j)
        call deriv(n,t,y,wk)
 105    continue
        return
        end



        subroutine cn(n,h,t,y,deriv,wk)
        implicit real*8(a-h,o-z)
        dimension y(n), wk(n), fvec(n), ynew(n), yold(n), rhs(n)
        real*8 err,stoptime,hnew
        integer k
        external deriv
         save

         stoptime = t+h
         hnew = h
         do while (t < stoptime-1.d-10)
 201        call deriv(n,t,y,wk)
            rhs = y + 0.5d0*hnew*wk
            err = 1.d0
            k = 0
            yold = y
            do while (err > 1.d-12)
               call deriv(n,t,yold,wk)
               fvec = yold - 0.5d0*hnew*wk - rhs
               ynew =  yold - fvec
               err = maxval(abs(ynew-yold))
               k = k+1
               if (k>20 .or. err>1.d0) then
                  hnew = hnew/2.d0
                  goto 201
               endif
               yold = ynew
            end do
            y = yold
            t = t+hnew
            
         end do
        end


