function out=prox_spectral(X,alpha)
%PROX_SPECTRAL computes the proximal operator of the function alpha*norm(X,2)
%
%  Usage: 
%  out = PROX_SPECTRAL(X,alpha)
%  ===========================================
%  INPUT:
%  X - matrix to be projected
%  alpha - positive scalar
%  ===========================================
%  Output:
%  out - proximal operator at X

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

if (nargin < 2)
    error ('usage: prox_spectral(X,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_spectral(X,alpha) - alpha should be positive')
end

[u,s,v] = svd(X) ;
s = prox_linf(s,alpha) ;
out = u * s * v' ;

end
