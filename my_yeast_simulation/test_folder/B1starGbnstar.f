        subroutine B1starGbnstar(b1m,b1_star,gbn,gbnstar,dm)
        implicit real*8 (a-h,o-z)
        include "parame.h"
        parameter (n=npara, m=14*n)
        dimension b1m(n),b1_star(n),gbn(n),gbnstar(n),dm(n)

        gbn_inte = 0.d0
        b1m_inte = 0.d0

        do j = 1, n
          gbn_inte = gbn_inte + xJj(j)*gbn(j)*h
          b1m_inte = b1m_inte + xJj(j)*b1m(j)*h
        enddo

*   Gbg_n*

        t0 = delta*SA / (gbn_inte+1.d-12)
        if ( gbn_inte .le. 1.d-12 ) t0 = 0.d0
        do j = 1, n
          temp = ( t0*gbn(j) )**(-Gbgnq)
          gbnstar(j) = GbgnR / ( 1.d0 + temp )
c          write(*,*)'gbnstar',j,gbnstar(j)
        enddo

***************
c        test1 = gbnstar(10)
***************

*   regulated Dm

        t0 = beta*SA / (gbn_inte+1.d-12)
        if ( gbn_inte .le. 1.d-12 ) t0 = 0.d0
        do j = 1, n
c          temp = ( t0*gbn(j) )**dpower
c          dm(j) = dlc  / ( 1.d0 + temp )
c          dm(j) = dlc
          dm(j) = 1.d0
        enddo

*   B1*

        b1star_tot = b1m_inte / SA
        t0 = gamma * SA / (2.d0*b1m_inte+1.d-12)
        if ( b1m_inte .le. 1.d-12 ) t0 = 0.d0
        do j = 1, n
          temp = (t0*gbnstar(j)*b1m(j))**hpower
          b1_star(j) = b1star_tot*temp / ( 1.d0 + temp )
        enddo

***************
c        test2 = b1_star(10)
***************


        return
        end
