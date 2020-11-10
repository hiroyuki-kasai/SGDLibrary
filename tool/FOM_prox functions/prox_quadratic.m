function out=prox_quadratic(x,A,b,alpha)
%PROX_QUADRATIC computes the proximal operator of the function alpha*(0.5*x'*A*x+b'*x)
%
%  Usage: 
%  out = PROX_QUADRATIC(X,A,b,alpha)
%  ===========================================
%  INPUT:
%  x - vector to be projected
%  alpha - positive scalar
%  A - positive semidefinite matrix
%  b - vector
%  ===========================================
%  Assumptions:
%  A is psd
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

if (nargin < 4)
    error ('usage: prox_quadratic(x,A,b,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_quadratic(x,A,b,alpha) - alpha should be positive')
end

eps = 1e-10 ;
   
if (norm( A - A') > eps)
    error('usage: prox_quadratic(x,A,b,alpha) - A should be a symmetric matrix') ;
end
A = A + eps * eye(size(A,1)) ;
A = (A + A') / 2;

[~,p] = chol(A) ;

if (p > 0)
    error('usage: prox_quadratic(x,A,b,alpha) - A should be positive semidefinite"') ;
end

n = length(x) ;

out = (eye(n) + alpha * A) \ (x - alpha * b) ; 

end

