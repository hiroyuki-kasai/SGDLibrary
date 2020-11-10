function out = proj_halfspace_box(x,a,b,lb,ub)
%PROJ_HALFSPACE_BOX computes the orthogonal projection onto the intersection of a halfspace and a box
%                   {x : <a,x> <= b, lb<=x<=ub}
%
%  Usage: 
%  out = PROJ_HALFSPACE_BOX(x,a,b,[lb],[ub])
%  ===========================================
%  Input:
%  x - point to be projected (vector/matrix)
%  a - vector/matrix
%  b - scalar
%  lb - lower bound (vector/matrix/scalar) [default: -inf]
%  ub - upper bound (vector/matrix/scalar) [default: inf]
%  ===========================================
%  Assumptions:
%  The intersection of the halfspace and the box is nonempty
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

%reading the user x and setting defalut values when required.
if (nargin < 3)
    error ('usage: proj_halfspace_box(x,a,b,[lb],[ub]') ;
end

if (nargin < 5)
    %upper bound is not given, setting to defalut value :inf
    ub = inf ;
end
if ((nargin < 4) || (isempty( lb)))
    %lower bound is not given, setting to defalut value :0
    lb = -inf;
end

if  (any(any(lb > ub))) 
    error('Set is infeasible') ;
end

if trace(a'*min(max(x,lb),ub)) <= b
    out =  min(max(x,lb),ub) ;
else
    %use proj_hyperplane_box
    out = proj_hyperplane_box(x,a,b,lb,ub) ;
end



