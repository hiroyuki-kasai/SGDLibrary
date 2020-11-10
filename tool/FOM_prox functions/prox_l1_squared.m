function out=prox_l1_squared(x,alpha)
%PROX_L1_SQUARED computes the proximal operator of the function alpha*(norm(x(:),1)^2)
%
%  Usage: 
%  out = PROX_L1_SQUARED(x,alpha)
%  ===========================================
%  INPUT:
%  x - point to be projected (vector/matrix)
%  alpha - positive scalar
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

if (nargin < 2)
    error ('usage: prox_l1_squared(x,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_l1_squared(x,alpha) - alpha should be positive')
end

%setting eps to defalut value : 1e-10
eps = 1e-10 ;

if (norm(x) < eps)
    out = x;
    return ;
end

%defining f on mu - to be used by the bisetion
f=@(mu)   sum(sum( max(sqrt(alpha) *abs(x)/sqrt(mu) - 2*alpha,0))) - 1 ;

mu_min = 0 ;
mu_max = 1 ;
while(f(mu_max)> 0)
    mu_max = mu_max * 2 ;
end

final_mu = bisection(f,mu_min,mu_max,eps) ;
lam = max(sqrt(alpha) * abs(x)/sqrt(final_mu) - 2 *alpha,0) ;

out = (lam .* x) ./ (lam + 2 * alpha) ;

end

