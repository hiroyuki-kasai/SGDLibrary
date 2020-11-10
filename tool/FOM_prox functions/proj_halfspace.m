function [ out ] = proj_halfspace(x,a,b)
%PROJ_HALFSPACE computes the orthogonal projection onto the halfspace {x:<a,x><=b}
%
%  Usage: 
%  out = PROJ_HALFSPACE(x,a,b)
%  ===========================================
%  Input:
%  x - point to be projected (vector/matrix)
%  a - vector/matrix
%  b - scalar
%  ===========================================
%  Assumptions:
%  If a = 0 then b >= 0
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
    error ('usage: proj_halfspace(x,a,b)') ;
end

eps = 1e-10;
if (norm(a) < eps)
    if (b >= 0)
        out = x ;
    else
        error('Set is infeasible') ;
    end
else
    out = x- max((trace(a' * x) -b),0) / (norm(a,'fro')^2) * a ;
end

end

