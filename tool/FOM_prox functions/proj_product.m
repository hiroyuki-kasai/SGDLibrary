function out = proj_product(x,r)
%PROJ_PRODUCT computes the orthogonal projection onto product superlevel set {x>0: prod(x) >= r} 
%                                               %
%  Usage: 
%  out = PROJ_PRODUCT(x,r)
%  =============================================================
%  INPUT:
%  x - point to be projected (vector/matrix)
%  r - a positive scalar 
%  ==============================================================
%  Assumptions:
%  r > 0
%  ==============================================================
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

%reading the user x and setting defalut values when required.
if (nargin < 2)
    error ('usage: proj_product(x,r)') ;
end

%default value for eps
eps = 1e-10 ;

if (r <= 0)
    error('Set is infeasible') ;
end

%checking feasibility of x

if (prod(x) >= r)
    out = x;
    return
end

%defining f on lamda - to be used by the bisetion

f= @(lam)   -sum(sum((log(0.5*(x + sqrt(x.^2+4*lam)))))) + log(r)     ;

lamda_min = 0;

lamda_max = 1;
while(f(lamda_max)>0)
    lamda_max = lamda_max *2  ;
end

final_lam = bisection(f,lamda_min,lamda_max,eps) ;
out=   0.5*(x + sqrt(x.^2+4* final_lam));
