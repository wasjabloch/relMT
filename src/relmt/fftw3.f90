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
! FILE: fftw3.f90

integer ( kind = 4 ), parameter :: fftw_r2hc = 0
integer ( kind = 4 ), parameter :: fftw_hc2r = 1
integer ( kind = 4 ), parameter :: fftw_dht = 2
integer ( kind = 4 ), parameter :: fftw_redft00 = 3
integer ( kind = 4 ), parameter :: fftw_redft01 = 4
integer ( kind = 4 ), parameter :: fftw_redft10 = 5
integer ( kind = 4 ), parameter :: fftw_redft11 = 6
integer ( kind = 4 ), parameter :: fftw_rodft00 = 7
integer ( kind = 4 ), parameter :: fftw_rodft01 = 8
integer ( kind = 4 ), parameter :: fftw_rodft10 = 9
integer ( kind = 4 ), parameter :: fftw_rodft11 = 10
integer ( kind = 4 ), parameter :: fftw_forward = -1
integer ( kind = 4 ), parameter :: fftw_backward = +1
integer ( kind = 4 ), parameter :: fftw_measure = 0
integer ( kind = 4 ), parameter :: fftw_destroy_input = 1
integer ( kind = 4 ), parameter :: fftw_unaligned = 2
integer ( kind = 4 ), parameter :: fftw_conserve_memory = 4
integer ( kind = 4 ), parameter :: fftw_exhaustive = 8
integer ( kind = 4 ), parameter :: fftw_preserve_input = 16
integer ( kind = 4 ), parameter :: fftw_patient = 32
integer ( kind = 4 ), parameter :: fftw_estimate = 64
integer ( kind = 4 ), parameter :: fftw_estimate_patient = 128
integer ( kind = 4 ), parameter :: fftw_believe_pcost = 256
integer ( kind = 4 ), parameter :: fftw_dft_r2hc_icky = 512
integer ( kind = 4 ), parameter :: fftw_nonthreaded_icky = 1024
integer ( kind = 4 ), parameter :: fftw_no_buffering = 2048
integer ( kind = 4 ), parameter :: fftw_no_indirect_op = 4096
integer ( kind = 4 ), parameter :: fftw_allow_large_generic = 8192
integer ( kind = 4 ), parameter :: fftw_no_rank_splits = 16384
integer ( kind = 4 ), parameter :: fftw_no_vrank_splits = 32768
integer ( kind = 4 ), parameter :: fftw_no_vrecurse = 65536
integer ( kind = 4 ), parameter :: fftw_no_simd = 131072
! END FILE: fftw3.f90
