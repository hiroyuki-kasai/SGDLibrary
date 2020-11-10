function out=prox_norm2_linear(x,A,alpha)
%PROX_NORM2_LINEAR computes the proximal operator of the function alpha*(norm(A*x,2))
%
%  Usage: 
%  out = PROX_NORM2_LINEAR(x,A,alpha)
%  ===========================================
%  INPUT:
%  x - vector to be projected
%  A - matrix with a full row rank
%  alpha - positive scalar
%  ===========================================
%  Assumptions:
%  A with full row rank
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
    error ('usage: prox_norm2_linear(x,A,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_norm2_linear(x,A,alpha) - alpha should be positive')
end

if (rank(A) ~= size(A,1))
    error('usage: prox_norm2_linear(x,A,alpha) - Rows of A should be linearly independent') ;
end

B= A* A';
ineye = eye(size(B)) ;
eps = 1e-10 ; % default value

if (norm( B\A*x)) <= alpha
    out = x - A'* (B\A*x) ;
else
    f= @(lam)   norm( (B + lam*ineye)\(A*x)) ^2 - alpha ^2;
    
    lambda_min = 0 ;
    
    lambda_max = 1;
    while(f(lambda_max)>0)
        lambda_max = lambda_max *2  ;
    end
    
    final_lam = bisection(f,lambda_min,lambda_max,eps) ;
    out=  x - A'* ((B+final_lam * ineye)\(A*x))  ;

end

