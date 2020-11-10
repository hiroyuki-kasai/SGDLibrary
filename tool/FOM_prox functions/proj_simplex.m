function out = proj_simplex(x,r,eq_flag)
%PROJ_SIMPLEX computes the orthogonal projection onto the r-simplex {x:sum(x)=r,x>=0} 
%                                                or r-full simplex {x:sum(x)<=r,x>=0}
%
%  Usage: 
%  out = PROJ_SIMPLEX(x,[r],[eq_flag])
%  =============================================================
%  Input:
%  x - point to be projected (vector/matrix)
%  r - a positive scalar [default: 1]
%  eq_flag - a flag that determines whether the projection is onto the r-simplex ('eq') or 
%                 the r-full simplex ('ineq') [defualt: 'eq']
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

%reading the user x and setting defalut values when reqired.
if (nargin < 1)
    error ('usage: proj_simplex(x,r,[eq_flag])') ;
end

if (( nargin < 3) || (isempty(eq_flag)) )
    %eq_flag is not given, setting to default value: true
    eq_flag = 'eq' ;
end

if ((nargin < 2) || (isempty(r)))
    %r is not given, setting to default value: 1
    r = 1 ;
end

if (r < 0) 
    error('Set is infeasible') ;
end

if (strcmp(eq_flag,'eq') == 1)
    %call proj_hyperplane_box
   out = proj_hyperplane_box(x,ones(size(x)),r,0) ; 
else
    if (strcmp(eq_flag,'ineq') == 1)
        %call proj_halfspace_box
        out = proj_halfspace_box(x,ones(size(x)),r,0) ;
    else
        error('usage: proj_simplex(x,r,[eq_flag]) - eq_flag should be either eq or ineq') ;
    end
end

