function out=prox_sum_k_largest_eigenvalues(X,k,alpha)
% PROX_SUM_K_LARGEST_EIGENVALUES computes the proximal operator of the 
%                      function alpha* (sum of k largest eigenvalues)
%
%  Usage: 
%  out = PROX_SUM_K_LARGEST_EIGENVALUES(X,k,alpha)
%  ===========================================
%  INPUT:
%  X - nxn matrix to be projected
%  k - positive integer
%  alpha - positive scalar
%  ===========================================
%  Assumptions:
%  X is symmetric
%  k in {1,...,n}
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
    error ('usage: prox_sum_k_largest_eigenvalues(X,k,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_sum_k_largest_eigenvalues(X,k,alpha) - alpha should be positive')
end

eps = 1e-10 ; % defalut value for eps : 1e-10
if ((size(X,1) ~= size(X,2)) || (norm( X - X') > eps))
    error('prox_sum_k_largest_eigenvalues(X,k,alpha) - X should be a symmetric matrix') ;
end

len = size(X,1) ;
if ((k < 1) || ( k > len) || (abs(round(k) - k)  > eps))
    error('prox_sum_k_largest_eigenvalues(X,k,alpha) - k should be in {1,...,length(X)}')
end

X = 0.5 * (X + X');
[V,D] = eig(X) ;

out = X - alpha * V*diag(proj_hyperplane_box(diag(D)/alpha,ones(len,1),k,0,1))*V' ;



