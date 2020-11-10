function out = proj_hyperplane_box(x,a,b,l,u)
%PROJ_HYPERPLANE_BOX computes the orthogonal projection onto the intersection of a hyperplane and a box {x:<a,x>=b,l<=x<=u}
%
%  Usage:
%  out = PROJ_HYPERPLANE_BOX(x,a,b,[l],[u])
%  ===========================================
%  Input:
%  x - point to be projected (vector/matrix)
%  a - vector/matrix
%  b - scalar
%  l - lower bound (vector/matrix/scalar) [default: -inf]
%  u - upper bound (vector/matrix/scalar) [default: inf]
%  ===========================================
%  Assumptions:
%  The intersection of the hyperplane and the box is nonempty
%  ===========================================
%  Output:
%  out - projection vector

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
    error ('usage: proj_hyperplane_box(x,a,b,[l],[u])') ;
end

if (nargin < 5)
    %upper bound is not given, setting to defalut value :inf
    u = inf ;
end
if ((nargin < 4) || (isempty( l)))
    %lower bound is not given, setting to defalut value :-inf
    l = -inf;
end

%checking that sum (l) < b < sum (u)


sumlb = trace(a'*((l .* (sign(a)>0)) + (u .* (sign(a)<0)))) ;
sumub = trace(a'*((u .* (sign(a)>0)) + (l .* (sign(a)<0)))) ;

if ((sumlb > b) || (any(any(l > u))) ||  (sumub < b))
    error('Set is infeasible') ;
end

%solve with equality
%defining f on lambda - to be used by the bisetion

eps = 1e-10 ;

f= @(lam)   trace(a'*min(max(x-lam*a,l),u))-b;
lambda_min = -1;
while(f(lambda_min)<0)
    lambda_min = lambda_min *2  ;
end

lambda_max = 1;
while(f(lambda_max)>0)
    lambda_max = lambda_max *2  ;
end

final_lam = bisection(f,lambda_min,lambda_max,eps) ;
out=  min(max(x-final_lam*a,l),u) ;
