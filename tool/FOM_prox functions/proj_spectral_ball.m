function out = proj_spectral_ball(X,r)
%PROJ_SPECTRAL_BALL computes the orthogonal projection onto the spectral-norm ball
%                             {X:norm(X,2) <= r}
%
%  Usage: 
%  out = PROJ_SPECTRAL_BALL(X,[r])
%  ===========================================
%  INPUT:
%  X - matrix to be projected
%  r -  radius (scalar) - [default: 1]
%  ===========================================
%  Assumptions:
%  r > 0
%  ===========================================
%  Output:
%  out - projection matrix

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
    error ('usage: proj_spectral_ball(X,[r])') ;
end

if (nargin < 2)
    % r is not given, setting to default value: 1
    r = 1 ;
end

if ( r < 0 ) 
   error('Set is infeasible') ;
end

[U,S,V] = svd(X) ;

newS = spdiags(min(spdiags(S,0),r),0,S) ;
out = U * newS *V' ;
