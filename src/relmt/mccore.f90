! relMT - Program to compute relative earthquake moment tensors
! Copyright (C) 2024 Wasja Bloch, Doriane Drolet, Michael Bostock
!
! This program is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free Software
! Foundation, either version 3 of the License, or (at your option) any later
! version.
!
! This program is distributed in the hope that it will be useful, but WITHOUT
! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
! FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along with
! this program. If not, see <http://www.gnu.org/licenses/>.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

! FILE: core.f90
subroutine mccc_ssf2(scomp0,dt,mxlag,ndec,rowi,coli,valu,dd,cc,dd2,cc2,ns,nt)
implicit none
real(8), intent(in) :: mxlag,dt
integer, intent(in) :: ns,nt
integer, intent(in) :: ndec
real(8), intent(in), dimension(nt,ns) :: scomp0
real(8), intent(out), dimension(ns*(ns-1)*(ns-2)/3+1) :: dd
real(8), intent(out), dimension(ns*(ns-1)/2) :: cc,cc2,dd2
real(8), intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: valu
integer, intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: rowi,coli
real(8), dimension(nt,ns) :: scomp
real(8), dimension(2*nt,ns*(ns-1)/2) :: ccij
real(8), dimension(nt,3) :: gmat
real(8), dimension(3) :: cct
real(8), allocatable :: eobj(:),sij(:),sik(:),sjk(:),rlag(:),sij2(:)
integer, allocatable :: il(:),ijn(:),ikn(:),jkn(:),ij2(:)
integer, dimension(2) :: kk
integer :: ix,jx,kx,ll,mm,nl,nl2,ii,jj,k

! Initialize arrays.
    cc=0
    dd=0
    cc2=0
    dd2=0
    coli=0
    rowi=0
    valu=0

! Allocate grid search dimensions.
    nl=2*int(mxlag/(ndec*dt))+1
    nl2=nl*nl
    allocate (eobj(nl2),sij(nl2),sik(nl2),sjk(nl2),rlag(nl),sij2(nl))
    allocate (ijn(nl2),ikn(nl2),jkn(nl2),il(nl),ij2(nl))

! Create index vectors for accessing correlation functions.
    il=(/ (ix, ix=1,nl) /)
    ijn=ndec*(reshape(spread(il,1,nl),(/ nl2 /))-int(nl/2))+nt
    ikn=ndec*(reshape(spread(il,2,nl),(/ nl2 /))-int(nl/2))+nt
    ij2=ndec*(il-int(nl/2))+nt
    do ll=1,nl
        jkn((ll-1)*nl+1:ll*nl)=-ll+il(:)
    enddo
    jkn=ndec*jkn+nt+1
    rlag=ndec*dt*real(il-int(nl/2)-1)
!    print*,'MXLAG: +/-',dt*(ndec*int(nl/2))
!    print*,'DRLAG:',ndec*dt

! Remove mean and normalize seismograms.
    do ix=1,ns
        scomp(:,ix)=scomp0(:,ix)-sum(scomp0(:,ix))/nt
        scomp(:,ix)=scomp(:,ix)/norm2(scomp(:,ix))
    enddo

! First precompute correlation coefficients and find location of maximum
!    print*,'Precomputing correlations'
    do ix=1,ns-1
        ii=ns*(ix-1)-(ix*(ix-1))/2
        do jx=ix+1,ns
            call correlate(scomp(:,ix),scomp(:,jx),ccij(:,ii+jx-ix),nt)
            sij2=ccij(ij2,ii+jx-ix)
            k=maxloc(abs(sij2),dim=1)
            cc2(ii+jx-ix)=sij2(k)
            dd2(ii+jx-ix)=rlag(k)
        enddo
    enddo

! Loop over all stations
    ll=1
    mm=1

! 1st loop.
    do ix=1,ns-2
!        print*,'Processing waveform: ',ix
        ii=ns*(ix-1)-(ix*(ix-1))/2

! 2nd loop.
        do jx=ix+1,ns-1
            sij=ccij(ijn,ii+jx-ix)
            jj=ns*(jx-1)-(jx*(jx-1))/2

! 3rd loop.
            do kx=jx+1,ns
                sik=ccij(ikn,ii+kx-ix)
                sjk=ccij(jkn,jj+kx-jx)


! Define EOBJ function and find minimum location.
                eobj=(1+2*sij*sjk*sik-sij*sij-sjk*sjk-sik*sik)
                kk=minloc(transpose(reshape(eobj, (/ nl,nl /))))

! Compute correlation coefficients between time series and their rank 2 approximations
                gmat(:,1)=scomp(:,ix)
                gmat(:,2)=cshift(scomp(:,jx),-int(rlag(kk(1))/dt))
                gmat(:,3)=cshift(scomp(:,kx),-int(rlag(kk(2))/dt))
                call ccorf2(gmat,nt,cct)

! Augment summed coefficients. Traces that are not well represented by linear combination of other traces will have low sum.
                cc(ii+jx-ix)=cc(ii+jx-ix)+cct(1)
                cc(ii+kx-ix)=cc(ii+kx-ix)+cct(2)
                cc(jj+kx-jx)=cc(jj+kx-jx)+cct(3)
!                cc(ix,jx)=cc(ix,jx)+cct(1)
!                cc(ix,kx)=cc(ix,kx)+cct(2)
!                cc(jx,kx)=cc(jx,kx)+cct(3)

! Fill rows,columns and values for COO spars format.
                rowi(mm)=ll
                coli(mm)=ix
                valu(mm)=1.0
                rowi(mm+1)=ll
                coli(mm+1)=jx
                valu(mm+1)=-1.0
                rowi(mm+2)=ll+1
                coli(mm+2)=ix
                valu(mm+2)=1.0
                rowi(mm+3)=ll+1
                coli(mm+3)=kx
                valu(mm+3)=-1.0
                dd(ll)=rlag(kk(1))
                dd(ll+1)=rlag(kk(2))
                ll=ll+2
                mm=mm+4
            enddo
        enddo
    enddo

! Final row for 0 sum constraint to full rank.
    rowi(mm:)=ll
    valu(mm:)=1.0
    do jx=0,ns-1
       coli(jx+mm)=jx+1
    enddo
    dd(ll)=0.0

! Correct all array indices by -1 for python indexing.
    rowi=rowi-1
    coli=coli-1

! Compute average CC by normalizing sum and inverting Fisher transform.
! Note each triple produces 3*n*(n-1)*(n-2)/6 cij estimates
! that are distributed equally over n*(n-1)/2 cij possibilities leading to
! a multiplicity of n-2 for each cij
    cc=cc/(ns-2)
    cc=(exp(2*cc)-1)/(exp(2*cc)+1)

! Deallocate memory.
    deallocate (eobj,sij,sik,sjk,rlag,sij2,ijn,ikn,jkn,il,ij2)

end subroutine
!-----------------
subroutine mccc_ssf3(scomp0,dt,mxlag,ndec,rowi,coli,valu,dd,cc3,dd2,cc2,ns,nt)
implicit none
real(8), intent(in) :: mxlag,dt
integer, intent(in) :: ns,nt
integer, intent(in) :: ndec
real(8), intent(in), dimension(nt,ns) :: scomp0
real(8), intent(out), dimension(ns*(ns-1)*(ns-2)/2) :: cc3
real(8), intent(out), dimension(ns*(ns-1)*(ns-2)/3+1) :: dd
real(8), intent(out), dimension(ns*(ns-1)/2) :: cc2,dd2
real(8), intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: valu
integer, intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: rowi,coli
real(8), dimension(nt,ns) :: scomp
real(8), dimension(2*nt,ns*(ns-1)/2) :: ccij
real(8), dimension(nt,3) :: gmat
real(8), allocatable :: eobj(:),sij(:),sik(:),sjk(:),rlag(:),sij2(:)
integer, allocatable :: il(:),ijn(:),ikn(:),jkn(:),ij2(:)
integer, dimension(2) :: kk
integer :: ix,jx,kx,ll,mm,nn,nl,nl2,ii,jj,k

! Initialize arrays.
    cc3=0
    dd=0
    cc2=0
    dd2=0
    coli=0
    rowi=0
    valu=0

! Allocate grid search dimensions.
    nl=2*int(mxlag/(ndec*dt))+1
    nl2=nl*nl
    allocate (eobj(nl2),sij(nl2),sik(nl2),sjk(nl2),rlag(nl),sij2(nl))
    allocate (ijn(nl2),ikn(nl2),jkn(nl2),il(nl),ij2(nl))

! Create index vectors for accessing correlation functions.
    il=(/ (ix, ix=1,nl) /)
    ijn=ndec*(reshape(spread(il,1,nl),(/ nl2 /))-int(nl/2))+nt
    ikn=ndec*(reshape(spread(il,2,nl),(/ nl2 /))-int(nl/2))+nt
    ij2=ndec*(il-int(nl/2))+nt
    do ll=1,nl
        jkn((ll-1)*nl+1:ll*nl)=-ll+il(:)
    enddo
    jkn=ndec*jkn+nt+1
    rlag=ndec*dt*real(il-int(nl/2)-1)
!    print*,'MXLAG: +/-',dt*(ndec*int(nl/2))
!    print*,'DRLAG:',ndec*dt

! Remove mean and normalize seismograms.
    do ix=1,ns
        scomp(:,ix)=scomp0(:,ix)-sum(scomp0(:,ix))/nt
        scomp(:,ix)=scomp(:,ix)/norm2(scomp(:,ix))
    enddo

! First precompute correlation coefficients and find location of maximum
!    print*,'Precomputing correlations'
    do ix=1,ns-1
        ii=ns*(ix-1)-(ix*(ix-1))/2
        do jx=ix+1,ns
            call correlate(scomp(:,ix),scomp(:,jx),ccij(:,ii+jx-ix),nt)
            sij2=ccij(ij2,ii+jx-ix)
            k=maxloc(abs(sij2),dim=1)
            cc2(ii+jx-ix)=sij2(k)
            dd2(ii+jx-ix)=rlag(k)
        enddo
    enddo

! Loop over all stations
    ll=1
    mm=1
    nn=1
! 1st loop.
    do ix=1,ns-2
!        print*,'Processing waveform: ',ix
        ii=ns*(ix-1)-(ix*(ix-1))/2

! 2nd loop.
        do jx=ix+1,ns-1
            sij=ccij(ijn,ii+jx-ix)
            jj=ns*(jx-1)-(jx*(jx-1))/2

! 3rd loop.
            do kx=jx+1,ns
                sik=ccij(ikn,ii+kx-ix)
                sjk=ccij(jkn,jj+kx-jx)


! Define EOBJ function and find minimum location.
                eobj=(1+2*sij*sjk*sik-sij*sij-sjk*sjk-sik*sik)
                kk=minloc(transpose(reshape(eobj, (/ nl,nl /))))

! Compute correlation coefficients between time series and their rank 2 approximations
                gmat(:,1)=scomp(:,ix)
                gmat(:,2)=cshift(scomp(:,jx),-int(rlag(kk(1))/dt))
                gmat(:,3)=cshift(scomp(:,kx),-int(rlag(kk(2))/dt))
                call ccorf3(gmat,nt,cc3(nn:nn+2))

! Fill rows,columns and values for COO spars format.
                rowi(mm)=ll
                coli(mm)=ix
                valu(mm)=1.0
                rowi(mm+1)=ll
                coli(mm+1)=jx
                valu(mm+1)=-1.0
                rowi(mm+2)=ll+1
                coli(mm+2)=ix
                valu(mm+2)=1.0
                rowi(mm+3)=ll+1
                coli(mm+3)=kx
                valu(mm+3)=-1.0
                dd(ll)=rlag(kk(1))
                dd(ll+1)=rlag(kk(2))
                ll=ll+2
                nn=nn+3
                mm=mm+4
            enddo
        enddo
    enddo

! Final row for 0 sum constraint to full rank.
    rowi(mm:)=ll
    valu(mm:)=1.0
    do jx=0,ns-1
       coli(jx+mm)=jx+1
    enddo
    dd(ll)=0.0

! Correct all array indices by -1 for python indexing.
    rowi=rowi-1
    coli=coli-1

! Deallocate memory.
    deallocate (eobj,sij,sik,sjk,rlag,sij2,ijn,ikn,jkn,il,ij2)

end subroutine
!-----------------
subroutine mccc_ssf0(scomp0,dt,mxlag,ndec,verb,rowi,coli,valu,dd,cc3,ns,nt)
! Input:
! scomp0: Input seismogram matrix of NS seismograms x NT samples
! dt: Sampling interval (seconds)
! mxlag: Maximum time lag to search for correlation (seconds)
! ndec: Number of samples to decimate time lag search (samples)
! verb: Verbose (logical)
!
! Output:
! rowi: sparse matrix row indices
! coli: sparse matrix column indices
! valu: Values of G matrix (Eq. 3 in Bostock et al., 2021, BSSA)
! dd: Relative lag time triplets from S wave combinations i j k (p. 135, 'S waves')
! cc3: Corresponding cross correlation coefficients
! ns: Number of seismograms
! nt: Number of samples

implicit none
real(8), intent(in) :: mxlag,dt
integer, intent(in) :: ns,nt
integer, intent(in) :: ndec
logical, intent(in) :: verb
real(8), intent(in), dimension(nt,ns) :: scomp0
real(8), intent(out), dimension(ns*(ns-1)*(ns-2)/2) :: cc3
real(8), intent(out), dimension(ns*(ns-1)*(ns-2)/3+1) :: dd
real(8), intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: valu
integer, intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: rowi,coli
real(8), dimension(nt,ns) :: scomp
real(8), dimension(2*nt,ns*(ns-1)/2) :: ccij
real(8), dimension(nt,3) :: gmat
real(8), allocatable :: eobj(:),sij(:),sik(:),sjk(:),rlag(:)
integer, allocatable :: il(:),ijn(:),ikn(:),jkn(:),ij2(:)
integer, dimension(2) :: kk
integer :: ix,jx,kx,ll,mm,nn,nl,nl2,ii,jj

! Initialize arrays.
    cc3=0
    dd=0
    coli=0
    rowi=0
    valu=0

! Allocate grid search dimensions.
    nl=2*int(mxlag/(ndec*dt))+1
    nl2=nl*nl
    allocate (eobj(nl2),sij(nl2),sik(nl2),sjk(nl2),rlag(nl))
    allocate (ijn(nl2),ikn(nl2),jkn(nl2),il(nl),ij2(nl))

! Create index vectors for accessing correlation functions.
    il=(/ (ix, ix=1,nl) /)
    ijn=ndec*(reshape(spread(il,1,nl),(/ nl2 /))-int(nl/2))+nt
    ikn=ndec*(reshape(spread(il,2,nl),(/ nl2 /))-int(nl/2))+nt
    ij2=ndec*(il-int(nl/2))+nt
    do ll=1,nl
        jkn((ll-1)*nl+1:ll*nl)=-ll+il(:)
    enddo
    jkn=ndec*jkn+nt+1
    rlag=ndec*dt*real(il-int(nl/2)-1)
    if (verb) then
        print*,'MXLAG: +/-',dt*(ndec*int(nl/2))
        print*,'DRLAG:',ndec*dt
    end if

! Remove mean and normalize seismograms.
    do ix=1,ns
        scomp(:,ix)=scomp0(:,ix)-sum(scomp0(:,ix))/nt
        scomp(:,ix)=scomp(:,ix)/norm2(scomp(:,ix))
    enddo

! First precompute correlation coefficients and find location of maximum
    if (verb) then
        print*,'Precomputing correlations'
    end if
    do ix=1,ns-1
        ii=ns*(ix-1)-(ix*(ix-1))/2
        do jx=ix+1,ns
            call correlate(scomp(:,ix),scomp(:,jx),ccij(:,ii+jx-ix),nt)
        enddo
    enddo

! Loop over all stations
    ll=1
    mm=1
    nn=1
! 1st loop.
    do ix=1,ns-2
        if (verb) then
            print*,'Waveforms to process left: ',ns-1-ix
        end if
        ii=ns*(ix-1)-(ix*(ix-1))/2

! 2nd loop.
        do jx=ix+1,ns-1
            sij=ccij(ijn,ii+jx-ix)
            jj=ns*(jx-1)-(jx*(jx-1))/2

! 3rd loop.
            do kx=jx+1,ns
                sik=ccij(ikn,ii+kx-ix)
                sjk=ccij(jkn,jj+kx-jx)


! Define EOBJ function and find minimum location.
                eobj=(1+2*sij*sjk*sik-sij*sij-sjk*sjk-sik*sik)
                kk=minloc(transpose(reshape(eobj, (/ nl,nl /))))

! Compute correlation coefficients between time series and their rank 2 approximations
                gmat(:,1)=scomp(:,ix)
                gmat(:,2)=cshift(scomp(:,jx),-int(rlag(kk(1))/dt))
                gmat(:,3)=cshift(scomp(:,kx),-int(rlag(kk(2))/dt))
                call ccorf3(gmat,nt,cc3(nn:nn+2))

! Fill rows,columns and values for COO spars format.
                rowi(mm)=ll
                coli(mm)=ix
                valu(mm)=1.0
                rowi(mm+1)=ll
                coli(mm+1)=jx
                valu(mm+1)=-1.0
                rowi(mm+2)=ll+1
                coli(mm+2)=ix
                valu(mm+2)=1.0
                rowi(mm+3)=ll+1
                coli(mm+3)=kx
                valu(mm+3)=-1.0
                dd(ll)=rlag(kk(1))
                dd(ll+1)=rlag(kk(2))
                ll=ll+2
                nn=nn+3
                mm=mm+4
            enddo
        enddo
    enddo

! Final row for 0 sum constraint to full rank.
    rowi(mm:)=ll
    valu(mm:)=1.0
    do jx=0,ns-1
       coli(jx+mm)=jx+1
    enddo
    dd(ll)=0.0

! Correct all array indices by -1 for python indexing.
    rowi=rowi-1
    coli=coli-1

! Deallocate memory.
    deallocate (eobj,sij,sik,sjk,rlag,ijn,ikn,jkn,il,ij2)

end subroutine
!-----------------
subroutine mccc_ssf(scomp0,dt,mxlag,ndec,rowi,coli,valu,dd,cc,ns,nt)

! Input:
! scomp0: Input seismogram matrix of NS seismograms x NT samples
! dt: Sampling interval (seconds)
! mxlag: Maximum time lag to search for correlation (seconds)
! ndec: Number of samples to decimate time lag search (samples)
!
! Output:
! rowi: sparse matrix row indices
! coli: sparse matrix column indices
! valu: Values of G matrix (Eq. 3 in Bostock et al., 2021, BSSA)
! dd: Pairwise relative lag times (Eq. 3)
! cc: Averaged cross correlation coefficients hat(C)ij
!     at optimal lag times (NS x NS)
! ns: Number of seismograms
! nt: Number of samples

implicit none
real(8), intent(in) :: mxlag,dt
integer, intent(in) :: ns,nt
integer, intent(in) :: ndec
real(8), intent(in), dimension(nt,ns) :: scomp0
real(8), intent(out), dimension(ns,ns) :: cc
real(8), intent(out), dimension(ns*(ns-1)*(ns-2)/3+1) :: dd
real(8), intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: valu
integer, intent(out), dimension(ns+2*ns*(ns-1)*(ns-2)/3) :: rowi,coli
real(8), dimension(nt,ns) :: scomp
real(8), dimension(2*nt,ns*(ns-1)/2) :: ccij
real(8), dimension(nt,3) :: gmat
real(8), allocatable :: eobj(:),sij(:),sik(:),sjk(:),rlag(:)
real(8), dimension(3) :: cct
integer, allocatable :: il(:),ijn(:),ikn(:),jkn(:)
integer :: ix,jx,kx,ll,mm,nl,nl2,ii,jj
integer, dimension(2) :: kk

! Initialize arrays.
    cc=0
    dd=0
    coli=0
    rowi=0
    valu=0

! Allocate grid search dimensions.
    nl=2*int(mxlag/(ndec*dt))+1
    nl2=nl*nl
    allocate (eobj(nl2),sij(nl2),sik(nl2),sjk(nl2),rlag(nl))
    allocate (ijn(nl2),ikn(nl2),jkn(nl2),il(nl))

! Create index vectors for accessing correlation functions.
    il=(/ (ix, ix=1,nl) /)
    ijn=ndec*(reshape(spread(il,1,nl),(/ nl2 /))-int(nl/2))+nt
    ikn=ndec*(reshape(spread(il,2,nl),(/ nl2 /))-int(nl/2))+nt
    do ll=1,nl
        jkn((ll-1)*nl+1:ll*nl)=-ll+il(:)
    enddo
    jkn=ndec*jkn+nt+1
    rlag=ndec*dt*real(il-int(nl/2)-1)
!    print*,'MXLAG: +/-',dt*(ndec*int(nl/2))
!    print*,'DRLAG:',ndec*dt

! Remove mean and normalize seismograms.
    do ix=1,ns
        scomp(:,ix)=scomp0(:,ix)-sum(scomp0(:,ix))/nt
        scomp(:,ix)=scomp(:,ix)/norm2(scomp(:,ix))
    enddo

! First precompute correlation coefficients
!    print*,'Precomputing correlations'
    do ix=1,ns-1
        ii=ns*(ix-1)-(ix*(ix-1))/2
        do jx=ix+1,ns
            call correlate(scomp(:,ix),scomp(:,jx),ccij(:,ii+jx-ix),nt)
        enddo
    enddo

! Loop over all stations
    ll=1
    mm=1
! 1st loop.
    do ix=1,ns-2
!        print*,'Processing waveform: ',ix
        ii=ns*(ix-1)-(ix*(ix-1))/2

! 2nd loop.
        do jx=ix+1,ns-1
            sij=ccij(ijn,ii+jx-ix)
            jj=ns*(jx-1)-(jx*(jx-1))/2

! 3rd loop.
            do kx=jx+1,ns
                sik=ccij(ikn,ii+kx-ix)
                sjk=ccij(jkn,jj+kx-jx)


! Define EOBJ function and find minimum location.
                eobj=(1+2*sij*sjk*sik-sij*sij-sjk*sjk-sik*sik)
                kk=minloc(transpose(reshape(eobj, (/ nl,nl /))))

! Compute correlation coefficients between time series and their rank 2 approximations
                gmat(:,1)=scomp(:,ix)
                gmat(:,2)=cshift(scomp(:,jx),-int(rlag(kk(1))/dt))
                gmat(:,3)=cshift(scomp(:,kx),-int(rlag(kk(2))/dt))
                call ccorf2(gmat,nt,cct)

! Augment summed coefficients. Traces that are not well represented by linear combination of other traces will have low sum.
                cc(ix,jx)=cc(ix,jx)+cct(1)
                cc(ix,kx)=cc(ix,kx)+cct(2)
                cc(jx,kx)=cc(jx,kx)+cct(3)

! Fill rows,columns and values for COO spars format.
                rowi(mm)=ll
                coli(mm)=ix
                valu(mm)=1.0
                rowi(mm+1)=ll
                coli(mm+1)=jx
                valu(mm+1)=-1.0
                rowi(mm+2)=ll+1
                coli(mm+2)=ix
                valu(mm+2)=1.0
                rowi(mm+3)=ll+1
                coli(mm+3)=kx
                valu(mm+3)=-1.0
                dd(ll)=rlag(kk(1))
                dd(ll+1)=rlag(kk(2))
                ll=ll+2
                mm=mm+4
            enddo
        enddo
    enddo

! Final row for 0 sum constraint to full rank.
    rowi(mm:)=ll
    valu(mm:)=1.0
    do jx=0,ns-1
       coli(jx+mm)=jx+1
    enddo
    dd(ll)=0.0

! Correct all array indices by -1 for python indexing.
    rowi=rowi-1
    coli=coli-1

! Compute average CC by normalizing sum and inverting Fisher transform.
! Note each triple produces 3*n*(n-1)*(n-2)/6 cij estimates
! that are distributed equally over n*(n-1)/2 cij possibilities leading to
! a multiplicity of n-2 for each cij
    cc=cc/(ns-2)
    cc=(exp(2*cc)-1)/(exp(2*cc)+1)

! Deallocate memory.
    deallocate (eobj,sij,sik,sjk,rlag,ijn,ikn,jkn,il)

end subroutine
!-----------------

subroutine mccc_ppf(scomp0,dt,mxlag,ndec,verb,rowi,coli,valu,dd,cc,ns,nt)

! Input:
! scomp0: Input seismogram matrix of NS seismograms x NT samples
! dt: Sampling interval (seconds)
! mxlag: Maximum time lag to search for correlation (seconds)
! ndec: Number of samples to decimate time lag search (samples)
! verb: Output progress messages to screen
!
! Output:
! rowi: sparse matrix row indices
! coli: sparse matrix column indices
! valu: Values of G matrix (Eq. 3 in Bostock et al., 2021, BSSA)
! dd: Pairwise relative lag times (Eq. 3)
! cc: Cross correlation coefficients at optimal lag times (NS x NS)
! ns: Number of seismograms
! nt: Number of samples

implicit none
real(8), intent(in) :: mxlag,dt
integer, intent(in) :: ns,nt
integer, intent(in) :: ndec
logical, intent(in) :: verb
real(8), intent(in), dimension(nt,ns) :: scomp0
real(8), intent(out), dimension(ns,ns) :: cc
real(8), intent(out), dimension(ns*(ns-1)/2+1) :: dd
real(8), intent(out), dimension(ns+ns*(ns-1)) :: valu
integer, intent(out), dimension(ns+ns*(ns-1)) :: rowi,coli
real(8), dimension(nt,ns) :: scomp
real(8), dimension(2*nt) :: sccij
real(8), allocatable :: eobj(:),sij(:),rlag(:)
integer, allocatable :: il(:),ijn(:)
integer :: ix,jx,ll,mm,nl,kk



! Initialize arrays.
! arrays up front.
    cc=0
    dd=0
    coli=0
    rowi=0
    valu=0

! Allocate grid search dimensions.
! nl = number of lag times
! nt = number of time samples
    nl=2*int(mxlag/(ndec*dt))+1
    allocate (eobj(nl),sij(nl),rlag(nl))
    allocate (il(nl), ijn(nl))

! Look only at the user-defined portion of the correlation functions
! Create index vector for accessing correlation functions.
    il=(/ (ix, ix=1,nl) /)
    ijn=ndec*(il-int(nl/2))+nt

! vector of time lags
    rlag=ndec*dt*real(il-int(nl/2)-1)
    if (verb) then
        print*,'MXLAG: +/-',dt*(ndec*int(nl/2))
        print*,'DRLAG:',ndec*dt
    end if

! Remove mean and normalize seismograms.
    do ix=1,ns
        scomp(:,ix)=scomp0(:,ix)-sum(scomp0(:,ix))/nt
        scomp(:,ix)=scomp(:,ix)/norm2(scomp(:,ix))
    enddo

! Loop over all stations
    ll=1
    mm=1
! 1st loop.
    do ix=1,ns-1
    if (verb) then
        print*,'Processing waveform: ',ix
        print*,'waveforms left: ', ns*ns - ix
    end if

! 2nd loop.
        do jx=ix+1,ns
            call correlate(scomp(:,ix),scomp(:,jx),sccij,nt)
            sij=sccij(ijn)
            eobj=(1-sij*sij)

! Find EOBJ function minimum location
            kk=minloc(eobj,dim=1)

! Store correlation coefficients.
            cc(ix,jx)=sij(kk)
            cc(jx,ix)=sij(kk)

! Fill rows,columns and values for COO spars format.
            rowi(mm)=ll
            coli(mm)=ix
            valu(mm)=1.0
            rowi(mm+1)=ll
            coli(mm+1)=jx
            valu(mm+1)=-1.0
            dd(ll)=rlag(kk)
            ll=ll+1
            mm=mm+2
        enddo
    enddo

! Final row for 0 sum constraint to full rank.
    rowi(mm:)=ll
    valu(mm:)=1.0
    do jx=0,ns-1
       coli(jx+mm)=jx+1
    enddo
    dd(ll)=0.0

! Correct all array indices by -1 for python indexing.
    rowi=rowi-1
    coli=coli-1

! Deallocate memory.
    deallocate (eobj,sij,rlag,il,ijn)

end subroutine

!--------------------------
subroutine ccorf(gmat,nt,cc)
!subroutine ccorf(a,b,c,gmat,nt,cc)
    implicit none
!    real(8), intent(in) :: a,b,c
    integer, intent(in) :: nt
    real(8), intent(in), dimension(1:nt,3) :: gmat
    real(8), intent(out) :: cc
    real(8), dimension(2,2) :: disc
    real(8), dimension(nt) :: gmatp
    real(8), dimension(3) :: v1,v2,ev
    real(8), dimension(2) :: dvec,coff
!    real(8) :: x1,x2,phi,d,e,f,de,ef,abc,abc2,bac2,cba2,d2,e2,f2,sx2
    real(8) :: x1,x2,phi,d,e,f,de,ef,d2,e2,f2,sx2
    real(8) :: ev1,ev2,m1,m2,cij
    real(8) :: ev3,m3
    real(8), dimension(3) :: v3
    real(8), parameter :: pi = 3.141592653589793
    integer :: ii,ij,ik,i0,i1,i2

! Compute off-diagonal elements of correlation matrix
    d=dot_product(gmat(:,1),gmat(:,2))
    e=dot_product(gmat(:,2),gmat(:,3))
    f=dot_product(gmat(:,3),gmat(:,1))

! Auxiliary variables
!    abc=a+b+c
!    abc2=2*a-b-c
!    bac2=2*b-a-c
!    cba2=2*c-b-a
! Major simplifications if a,b,c=1
! Note abc2 etc = 0 for matrix form we have adopted.
!    abc=3.0

    d2=d*d
    f2=f*f
    e2=e*e
    ef=e*f
    de=d*e

! Major
!    x1=a*a+b*b+c*c-a*b-a*c-b*c+3*(d2+e2+f2)
!    x2=-abc2*bac2*cba2+9*(cba2*d2+bac2*f2+abc2*e2)-54*(de*f)
    x1=3*(d2+e2+f2)
    x2=-54*(de*f)
    sx2=2*sqrt(x1)

! Note else condition is x2==0.
    if (x2 > 0) then
        phi=atan(sqrt(4*x1**3-x2*x2)/x2)
    elseif (x2 < 0) then
       phi=atan(sqrt(4*x1**3-x2*x2)/x2)+pi
    else
       phi=pi/2
    end if

!    ev(1)=(abc-sx2*cos(phi/3))/3
!    ev(2)=(abc+sx2*cos((phi-pi)/3))/3
!    ev(3)=(abc+sx2*cos((phi+pi)/3))/3
    ev(1)=(3-sx2*cos(phi/3))/3
    ev(2)=(3+sx2*cos((phi-pi)/3))/3
    ev(3)=(3+sx2*cos((phi+pi)/3))/3

! Sort 3 eigenvalues in descending (singular) order  ev1 is largest
! ev2 is medium, ev3 is smallest.
    i0=minloc(ev,1)
    i2=maxloc(ev,1)
    i1=6-i0-i2
    ev3=ev(i0)
    ev1=ev(i2)
    ev2=ev(i1)

! Normalization - may not be necessary but double check. Note we don't need
! ev3,m3 for rank 2 expansion so don't calculate.
!    m3=(d*(c-ev3)-ef)/(f*(b-ev3)-de)
!    m1=(d*(c-ev1)-ef)/(f*(b-ev1)-de)
!    m2=(d*(c-ev2)-ef)/(f*(b-ev2)-de)
!    v3(1)=(ev3-c-e*m3)/f
!    v3(2)=m1
!    v3(3)=1
!    v3=v3/norm2(v3)
!    v1(1)=(ev1-c-e*m1)/f
!    v1(2)=m1
!    v1(3)=1
!    v1=v1/norm2(v1)
!    v2(1)=(ev2-c-e*m2)/f
!    v2(2)=m2
!    v2(3)=1
!    v2=v2/norm2(v2)
    m3=(d*(1.0-ev3)-ef)/(f*(1.0-ev3)-de)
    m1=(d*(1.0-ev1)-ef)/(f*(1.0-ev1)-de)
    m2=(d*(1.0-ev2)-ef)/(f*(1.0-ev2)-de)
    v3(1)=(ev3-1.0-e*m3)/f
    v3(2)=m1
    v3(3)=1.0
    v3=v3/norm2(v3)
    v1(1)=(ev1-1.0-e*m1)/f
    v1(2)=m1
    v1(3)=1.0
    v1=v1/norm2(v1)
    v2(1)=(ev2-1.0-e*m2)/f
    v2(2)=m2
    v2(3)=1.0
    v2=v2/norm2(v2)


! Compute reconstruction coefficients.
    cc=0
    do ii = 1,3
       ij=modulo(ii,3)+1
       ik=modulo(ii+1,3)+1
       disc=reshape((/ v2(ik), -v2(ij), -v1(ik), v1(ij) /), shape(disc))
       dvec=(/ v1(ii), v2(ii) /)
       coff=matmul(disc,dvec)/(v1(ij)*v2(ik)-v1(ik)*v2(ij))
       gmatp=coff(1)*gmat(:,ij)+coff(2)*gmat(:,ik)
! Exploit fact that norm(gmat(,i)) is 1
!       cc=cc+dot_product(gmat(:,ii),gmatp)/norm2(gmatp,gmatp)
! Fisher transform to get a better, unbaissed estimate after inverse transform.
       cij=dot_product(gmat(:,ii),gmatp)/norm2(gmatp)
!       print*,'CCOR',cij
       cc=cc+log((1+cij)/(1-cij))/2.0

    enddo
    cc=cc/3

end subroutine

!--------------------------
subroutine ccorf3(gmat,nt,cc3)
!subroutine ccorf(a,b,c,gmat,nt,cc)
    implicit none
!    real(8), intent(in) :: a,b,c
    integer, intent(in) :: nt
    real(8), intent(in), dimension(1:nt,3) :: gmat
    real(8), intent(out), dimension(3) :: cc3
    real(8), dimension(2,2) :: disc
    real(8), dimension(nt) :: gmatp
    real(8), dimension(3) :: v1,v2,ev
    real(8), dimension(2) :: dvec,coff
!    real(8) :: x1,x2,phi,d,e,f,de,ef,abc,abc2,bac2,cba2,d2,e2,f2,sx2
    real(8) :: x1,x2,phi,d,e,f,de,ef,d2,e2,f2,sx2
    real(8) :: ev1,ev2,m1,m2
!    real(8) :: ev3,m3
!    real(8), dimension(3) :: v3
    real(8), parameter :: pi = 3.141592653589793
    integer :: ii,ij,ik,i0,i1,i2

! Compute off-diagonal elements of correlation matrix
    d=dot_product(gmat(:,1),gmat(:,2))
    e=dot_product(gmat(:,2),gmat(:,3))
    f=dot_product(gmat(:,3),gmat(:,1))

! Auxiliary variables
!    abc=a+b+c
!    abc2=2*a-b-c
!    bac2=2*b-a-c
!    cba2=2*c-b-a
! Major simplifications if a,b,c=1
! Note abc2 etc = 0 for matrix form we have adopted.
!    abc=3.0

    d2=d*d
    f2=f*f
    e2=e*e
    ef=e*f
    de=d*e

! Major
!    x1=a*a+b*b+c*c-a*b-a*c-b*c+3*(d2+e2+f2)
!    x2=-abc2*bac2*cba2+9*(cba2*d2+bac2*f2+abc2*e2)-54*(de*f)
    x1=3*(d2+e2+f2)
    x2=-54*(de*f)
    sx2=2*sqrt(x1)

! Note else condition is x2==0.
    if (x2 > 0) then
        phi=atan(sqrt(4*x1**3-x2*x2)/x2)
    elseif (x2 < 0) then
       phi=atan(sqrt(4*x1**3-x2*x2)/x2)+pi
    else
       phi=pi/2
    end if

!    ev(1)=(abc-sx2*cos(phi/3))/3
!    ev(2)=(abc+sx2*cos((phi-pi)/3))/3
!    ev(3)=(abc+sx2*cos((phi+pi)/3))/3
    ev(1)=(3-sx2*cos(phi/3))/3
    ev(2)=(3+sx2*cos((phi-pi)/3))/3
    ev(3)=(3+sx2*cos((phi+pi)/3))/3

! Sort 3 eigenvalues in descending (singular) order  ev1 is largest
! ev2 is medium, ev3 is smallest.
    i0=minloc(ev,1)
    i2=maxloc(ev,1)
    i1=6-i0-i2
!   ev3=ev(i0)
    ev1=ev(i2)
    ev2=ev(i1)

! Normalization - may not be necessary but double check. Note we don't need
! ev3,m3 for rank 2 expansion so don't calculate.
!    m3=(d*(c-ev3)-ef)/(f*(b-ev3)-de)
!    m1=(d*(c-ev1)-ef)/(f*(b-ev1)-de)
!    m2=(d*(c-ev2)-ef)/(f*(b-ev2)-de)
!    v3(1)=(ev3-c-e*m3)/f
!    v3(2)=m1
!    v3(3)=1
!    v3=v3/norm2(v3)
!    v1(1)=(ev1-c-e*m1)/f
!    v1(2)=m1
!    v1(3)=1
!    v1=v1/norm2(v1)
!    v2(1)=(ev2-c-e*m2)/f
!    v2(2)=m2
!    v2(3)=1
!    v2=v2/norm2(v2)
!   m3=(d*(1.0-ev3)-ef)/(f*(1.0-ev3)-de)
    m1=(d*(1.0-ev1)-ef)/(f*(1.0-ev1)-de)
    m2=(d*(1.0-ev2)-ef)/(f*(1.0-ev2)-de)
!   v3(1)=(ev3-1.0-e*m3)/f
!   v3(2)=m1
!   v3(3)=1.0
!   v3=v3/norm2(v3)
    v1(1)=(ev1-1.0-e*m1)/f
    v1(2)=m1
    v1(3)=1.0
    v1=v1/norm2(v1)
    v2(1)=(ev2-1.0-e*m2)/f
    v2(2)=m2
    v2(3)=1.0
    v2=v2/norm2(v2)


! Compute reconstruction coefficients.
    cc3=0
    do ii = 1,3
       ij=modulo(ii,3)+1  ! 2, 3, 1
       ik=modulo(ii+1,3)+1  ! 3, 1, 2
       disc=reshape((/ v2(ik), -v2(ij), -v1(ik), v1(ij) /), shape(disc))
       dvec=(/ v1(ii), v2(ii) /)
       ! division by zero possible with synthetic data
       coff=matmul(disc,dvec)/(v1(ij)*v2(ik)-v1(ik)*v2(ij))
       gmatp=coff(1)*gmat(:,ij)+coff(2)*gmat(:,ik)
! Exploit fact that norm2(gmat(,i)) is 1
       cc3(ii)=dot_product(gmat(:,ii),gmatp)/norm2(gmatp)
!       print*,'CCOR3',cc3(ii)
    enddo

end subroutine
!--------------------

subroutine correlate(fn1,fn2,ccfn,nt)
! Compute and return correlation function CCFN from input functions FN1,FN2
implicit none
include 'fftw3.f90'
integer, intent(in) :: nt
real*8, intent(in), dimension(nt) :: fn1,fn2
real*8, intent(out), dimension(2*nt) :: ccfn
complex*16, dimension(2*nt) :: cfn1,cfn2
real*8, dimension(2*nt) :: rfn
real*8, parameter :: pi = 3.14159265358979323846
integer*8 :: plan_for1,plan_for2,plan_bac

! Initialize and set up plans.
rfn=0.0
cfn1=(0.0,0.0)
cfn2=(0.0,0.0)
call dfftw_plan_dft_r2c_1d(plan_for1,2*nt,rfn,cfn1,fftw_r2hc,fftw_estimate)
call dfftw_plan_dft_r2c_1d(plan_for2,2*nt,rfn,cfn2,fftw_r2hc,fftw_estimate)
call dfftw_plan_dft_c2r_1d(plan_bac,2*nt,cfn1,rfn,fftw_hc2r,fftw_estimate)

rfn(1:nt)=fn1(:)
call dfftw_execute_dft_r2c(plan_for1,rfn,cfn1)
rfn(1:nt)=fn2(:)
call dfftw_execute_dft_r2c(plan_for2,rfn,cfn2)
cfn1=cfn1*dconjg(cfn2)/real(2*nt,kind=8)
call dfftw_execute_dft_c2r(plan_bac,cfn1,rfn)
ccfn(1:nt)=rfn(nt+1:2*nt)
ccfn(nt+1:2*nt)=rfn(1:nt)

! Destroy plans.
call dfftw_destroy_plan(plan_for1)
call dfftw_destroy_plan(plan_for2)
call dfftw_destroy_plan(plan_bac)

end subroutine

!--------------------------
subroutine ccorf2(gmat,nt,cc)
! Reconstruct S waveform from pairs of two others
! Input:
! gmat: Matrix of 3 seimograms
!       Time aligned along 1st, event aligned along 2nd dimension.
! nt: Number of samples
! Return:
! cc:  cross correlation coefficients between reconstructed waveforms Cijk, Cjki
!      and Ckij.

!subroutine ccorf(a,b,c,gmat,nt,cc)
    implicit none
!    real(8), intent(in) :: a,b,c
    integer, intent(in) :: nt
    real(8), intent(in), dimension(1:nt,3) :: gmat
    real(8), dimension(3), intent(out) :: cc(3)
    real(8), dimension(3) :: cct(3)
    real(8), dimension(2,2) :: disc
    real(8), dimension(nt) :: gmatp
    real(8), dimension(3) :: v1,v2,ev
    real(8), dimension(2) :: dvec,coff
!    real(8) :: x1,x2,phi,d,e,f,de,ef,abc,abc2,bac2,cba2,d2,e2,f2,sx2
    real(8) :: x1,x2,phi,d,e,f,de,ef,d2,e2,f2,sx2
    real(8) :: ev1,ev2,m1,m2,cij
!    real(8) :: ev3,m3
!    real(8), dimension(3) :: v3
    real(8), parameter :: pi = 3.141592653589793
    integer :: ii,ij,ik,i0,i1,i2

! Compute off-diagonal elements of correlation matrix (Eq. 8, Bostock et al. 2021)
    d=dot_product(gmat(:,1),gmat(:,2))
    e=dot_product(gmat(:,2),gmat(:,3))
    f=dot_product(gmat(:,3),gmat(:,1))

! Auxiliary variables
!    abc=a+b+c
!    abc2=2*a-b-c
!    bac2=2*b-a-c
!    cba2=2*c-b-a
! Major simplifications if a,b,c=1
! Note abc2 etc = 0 for matrix form we have adopted.
!    abc=3.0

! Compute singular values and vectors following Deledalle et al. (2017)
    d2=d*d
    f2=f*f
    e2=e*e
    ef=e*f
    de=d*e

! Major
!    x1=a*a+b*b+c*c-a*b-a*c-b*c+3*(d2+e2+f2)
!    x2=-abc2*bac2*cba2+9*(cba2*d2+bac2*f2+abc2*e2)-54*(de*f)
    x1=3*(d2+e2+f2)
    x2=-54*(de*f)
    sx2=2*sqrt(x1)

! Note else condition is x2==0.
    if (x2 > 0) then
        phi=atan(sqrt(4*x1**3-x2*x2)/x2)
    elseif (x2 < 0) then
        phi=atan(sqrt(4*x1**3-x2*x2)/x2)+pi
    else
        phi=pi/2
    end if

! Eignevalues lambda. Note a = b = c = 1 (Eq. 7 in Deledalle et al., 2017)
!    ev(1)=(abc-sx2*cos(phi/3))/3
!    ev(2)=(abc+sx2*cos((phi-pi)/3))/3
!    ev(3)=(abc+sx2*cos((phi+pi)/3))/3
    ev(1)=(3-sx2*cos(phi/3))/3
    ev(2)=(3+sx2*cos((phi-pi)/3))/3
    ev(3)=(3+sx2*cos((phi+pi)/3))/3

! Sort 3 eigenvalues in descending (singular) order  ev1 is largest
! ev2 is medium, ev3 is smallest.
    i0=minloc(ev,1)
    i2=maxloc(ev,1)
    i1=6-i0-i2
!   ev3=ev(i0)
    ev1=ev(i2)
    ev2=ev(i1)

! Normalization - may not be necessary but double check.
!    m3=(d*(c-ev3)-ef)/(f*(b-ev3)-de)
!    m1=(d*(c-ev1)-ef)/(f*(b-ev1)-de)
!    m2=(d*(c-ev2)-ef)/(f*(b-ev2)-de)
!    v3(1)=(ev3-c-e*m3)/f
!    v3(2)=m1
!    v3(3)=1
!    v3=v3/norm2(v3)
!    v1(1)=(ev1-c-e*m1)/f
!    v1(2)=m1
!    v1(3)=1
!    v1=v1/norm2(v1)
!    v2(1)=(ev2-c-e*m2)/f
!    v2(2)=m2
!    v2(3)=1
!    v2=v2/norm2(v2)

! Eq. 11 in Deledalle et al. (2017)
    m1=(d*(1.0-ev1)-ef)/(f*(1.0-ev1)-de)
    m2=(d*(1.0-ev2)-ef)/(f*(1.0-ev2)-de)

! Note we don't need ev3,m3 for rank 2 expansion.
! So we don't calculate it.
!    m3=(d*(1.0-ev3)-ef)/(f*(1.0-ev3)-de)
!    v3(1)=(ev3-1.0-e*m3)/f
!    v3(2)=m1
!    v3(3)=1.0
!    v3=v3/norm2(v3)

! Eq. 10 in Deledalle et al. (2017)
    v1(1)=(ev1-1.0-e*m1)/f
    v1(2)=m1
    v1(3)=1.0
    v1=v1/norm2(v1)
    v2(1)=(ev2-1.0-e*m2)/f
    v2(2)=m2
    v2(3)=1.0
    v2=v2/norm2(v2)


! Compute reconstruction coefficients.
    cc=0
    do ii = 1,3
       ij=modulo(ii,3)+1
       ik=modulo(ii+1,3)+1
       disc=reshape((/ v2(ik), -v2(ij), -v1(ik), v1(ij) /), shape(disc))
       dvec=(/ v1(ii), v2(ii) /)
       coff=matmul(disc,dvec)/(v1(ij)*v2(ik)-v1(ik)*v2(ij))
       gmatp=coff(1)*gmat(:,ij)+coff(2)*gmat(:,ik)
! Exploit fact that norm(gmat(,i)) is 1
!       cc=cc+dot_product(gmat(:,ii),gmatp)/norm2(gmatp,gmatp)
! Fisher transform to get a better, unbaissed estimate after inverse transform.
       cij=dot_product(gmat(:,ii),gmatp)/norm2(gmatp)
!       print*,'CCOR',cij
       cct(ii)=log((1+cij)/(1-cij))/2.0

    enddo

    cc(1)=(cct(1)+cct(2))/2
    cc(2)=(cct(1)+cct(3))/2
    cc(3)=(cct(2)+cct(3))/2

end subroutine

!-------------------------
subroutine mccc_ppc(scomp0,dt,mxlag,ndec,rowi,coli,valu,dd,cc,ns,nt)
implicit none
real(8), intent(in) :: mxlag,dt
integer, intent(in) :: ns,nt
integer, intent(in) :: ndec
real(8), intent(in), dimension(nt,ns) :: scomp0
real(8), intent(out), dimension(ns,ns) :: cc
real(8), intent(out), dimension(ns*(ns-1)/2+1) :: dd
real(8), intent(out), dimension(ns+ns*(ns-1)) :: valu
integer, intent(out), dimension(ns+ns*(ns-1)) :: rowi,coli
real(8), dimension(nt,ns) :: scomp
real(8), dimension(2*nt) :: sccij
real(8), allocatable :: eobj(:),sij(:),rlag(:)
integer, allocatable :: il(:),ijn(:)
integer :: ix,jx,ll,mm,nl,kk

!!!! NOTE THIS VERSION EMPLOYS CCMAX RATHER THAN 1-S1**2 AS
!!!! EOBJ FUNCTION TO HELP MITIGATE CYCLE SKIPPING. ALSO
!!!! CC IS STORED AS VECTOR LIKE DD NOT MATRIX
! Initialize arrays.
! arrays up front.

    cc=0
    dd=0
    coli=0
    rowi=0
    valu=0

! Allocate grid search dimensions.
    nl=2*int(mxlag/(ndec*dt))+1
    allocate (eobj(nl),sij(nl),rlag(nl))
    allocate (il(nl), ijn(nl))

! Create index vector for accessing correlation functions.
    il=(/ (ix, ix=1,nl) /)
    ijn=ndec*(il-int(nl/2))+nt

    rlag=ndec*dt*real(il-int(nl/2)-1)
!    print*,'MXLAG: +/-',dt*(ndec*int(nl/2))
!    print*,'DRLAG:',ndec*dt

! Remove mean and normalize seismograms.
    do ix=1,ns
        scomp(:,ix)=scomp0(:,ix)-sum(scomp0(:,ix))/nt
        scomp(:,ix)=scomp(:,ix)/norm2(scomp(:,ix))
    enddo

! Loop over all stations
    ll=1
    mm=1
! 1st loop.
    do ix=1,ns-1
!        print*,'Processing waveform: ',ix

! 2nd loop.
        do jx=ix+1,ns
            call correlate(scomp(:,ix),scomp(:,jx),sccij,nt)
            sij=sccij(ijn)
!            eobj=(1-sij*sij)
            eobj=(sij)

! Find EOBJ function minimum location - USE MAXIMUM FOR PPC SINCE YOU
! ARE SEARCHING FOR MAX CC
!            kk=minloc(eobj,dim=1)
            kk=maxloc(eobj,dim=1)

! Store correlation coefficients.
            cc(ix,jx)=sij(kk)


! Fill rows,columns and values for COO spars format.
            rowi(mm)=ll
            coli(mm)=ix
            valu(mm)=1.0
            rowi(mm+1)=ll
            coli(mm+1)=jx
            valu(mm+1)=-1.0
            dd(ll)=rlag(kk)
            ll=ll+1
            mm=mm+2
        enddo
    enddo

! Final row for 0 sum constraint to full rank.
    rowi(mm:)=ll
    valu(mm:)=1.0
    do jx=0,ns-1
       coli(jx+mm)=jx+1
    enddo
    dd(ll)=0.0

! Correct all array indices by -1 for python indexing.
    rowi=rowi-1
    coli=coli-1

! Deallocate memory.
    deallocate (eobj,sij,rlag,il,ijn)

end subroutine

!--------------------------
! END FILE: core.f90
