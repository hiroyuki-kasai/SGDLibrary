function [ out ] = proj_Euclidean_ball( x,c,r )
%PROJ_EUCLIDEAN_BALL computes the orthogonal projection onto the Euclidean ball {x:||x-c||<=r}
%
%  Usage: 
%  out = PROJ_EUCLIDEAN_BALL(x,[c],[r])
%  ===========================================
%  Input:
%  x - point to be projected (vector/matrix)
%  c - center of the ball (vector/matrix) [default: c=0]
%  r - positive radius (a positive scalar) [default: r=1]
%  ===========================================
%  Assumptions:
%  l2 or Frobenius norm
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
if (nargin < 1)
     error ('usage: proj_Euclidean_ball(input,[c],[r])') ;
end

if (nargin < 3)
    %setting default value of r to 1
    r = 1;
end

if ((nargin < 2) || (isempty (c)))
    %setting default value of c to zeros
    c  = zeros(size(x)) ;
end

out=  c+r *(x-c) / max(norm(x-c,'fro'),r) ; 
end

