function [ out ] = proj_affine_set(x,A,b)
%PROJ_AFFINE_SET computes the orthogonal projection onto the affine set {x:Ax=b}
%
%  Usage:
%  out = PROJ_AFFINE_SET(x,A,b)
%  ===========================================
%  Input:
%  x - point to be projected (vector)
%  A - mxn matrix
%  b - m-length vecotr
%  ===========================================
%  Assumptions:
%  A with full row rank
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

%reading the user 
if (nargin < 3)
    error ('usage: proj_affine_set(x,A,b)') ;
end

if (rank(A) ~= size(A,1))
    error('usage: proj_affine_set(x,A,b) - Rows of A should be linearly independent') ;
end

out = x - A' * ((A*A') \(A* x - b)) ;

end

