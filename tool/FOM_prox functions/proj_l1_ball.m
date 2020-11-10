function out=proj_l1_ball(x,r)
%PROJ_L1_BALL computes the orthogonal projection onto the l1 ball {x: norm(x(:),1) <= r} 
%                                               
%  Usage: 
%  out = PROJ_L1_BALL(x,[r])
%  =============================================================
%  x:
%  x - point to be projected (vector/matrix)
%  r - a positive scalar [default: 1]
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

if (nargin < 1)
     error ('usage: prox_l1_ball(x,[r])') ;
end

if (nargin < 2)
    %setting default value to 1
    r = 1;
end

out = sign(x) .* proj_simplex(abs(x),r,'ineq') ;

end
