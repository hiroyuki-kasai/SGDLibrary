function [ out ] = proj_box( x,l,u )
%PROJ_BOX computes the orthogonal projection onto the box {x:l<=x<=u}
%
%  Usage: 
%  out = PROJ_BOX(x,l,u)
%  ===========================================
%  Input:
%  x - point to be projected (vector/matrix)
%  l - lower bound (vector/matrix/scalar)
%  u - upper bound (vector/matrix/scalar)
%  ===========================================
%  Assumptions:
%  l<=u
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

%reading the user x and 
if (nargin < 3)
    error ('usage: proj_box( x,l,u )') ;
end

if any(any((l > u)))
    error('Set is infeasible') ;
end

out= min(max(l,x),u) ;
end

