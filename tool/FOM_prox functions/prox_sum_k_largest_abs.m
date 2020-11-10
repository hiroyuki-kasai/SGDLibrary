function out=prox_sum_k_largest_abs(x,k,alpha)
%PROX_SUM_K_LARGEST_ABS computes the proximal operator of the function
%                     alpha*(sum of k largest absolute values of x(:))
%
%  Usage: 
%  out = PROX_SUM_K_LARGEST_ABS(x,k,alpha)
%  ===========================================
%  INPUT:
%  x - point to be projected (vector/matrix)
%  k - positive integer
%  alpha - positive scalar
%  ===========================================
%  Assumptions:
%  k in {1,...,length(x(:))}
%  ===========================================
%  Output:
%  out - proximal operator at x

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
    error ('usage: prox_sum_k_largest_abs(x,k,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_sum_k_largest_abs(x,k,alpha) - alpha should be positive')
end

eps = 1e-10 ; % default value
if ((k < 1) || ( k > length(x(:))) || (abs(round(k) - k)  > eps))
    error('usage: prox_sum_k_largest_abs(x,k,alpha) - k should be in {1,...,length(x(:))}')
end

out = x - alpha * proj_l1ball_box (x/alpha,ones(size(x)),k,ones(size(x))) ;

end

