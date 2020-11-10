function out=prox_Ky_Fan(X,k,alpha)
%PROX_KY_FAN computes the proximal operator of the function alpha* (Ky Fan norm)
%
%  Usage: 
%  out = PROX_KY_FAN(X,k,alpha)
%  ===========================================
%  INPUT:
%  X - mxn matrix to be projected 
%  k - positive integer
%  alpha - positive scalar
%  ===========================================
%  Assumptions:
%  X is symmetric
%  k in {1,...,min(m,n)}
%  ===========================================
%  Output:
%  out - proximal operator at X

% This file is part of the FOM package - a collection of first order methods for solving convex optimization problems
% Copyright (C) 2017 Amir and Nili Beck
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

if (nargin < 3)
    error ('usage: prox_Ky_Fan(X,k,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_Ky_Fan(X,k,alpha) - alpha should be positive')
end

eps = 1e-10 ; % defalut value for eps : 1e-10
if ((k < 1) || ( k > min(size(X))) || (abs(round(k) - k)  > eps))
    error('usage: prox_sum_k_largest_abs(X,k,alpha) - k should be in {1,...,min(size(X))}')
end

[U,S,V] = svd(X) ;
sdiag = spdiags(S,0);
onevec = ones(size(sdiag)) ;

newS = spdiags(proj_l1ball_box (sdiag/alpha,onevec,k,onevec),0,S) ;
out = X - alpha * U * newS *V' ;

end

