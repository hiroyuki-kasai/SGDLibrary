function out = proj_spectral_box_sym(X,l,u)
%PROJ_SPECTRAL_BOX_SYM computes the orthogonal projection onto the spectral box 
%                             {X symmetric :X-lI psd, uI-X psd}
%
%  Usage: 
%  out = PROJ_SPECTRAL_BOX_SYM(X,l,u)
%  ===========================================
%  Input:
%  X - matrix to be projected
%  l - lower bound (scalar)
%  u - upper bound (scalar) 
%  ===========================================
%  Assumptions:
%  l<=u
%  X symmetric
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

if (nargin < 3)
    error ('usage: proj_spectral_box_sym(X,l,u)') ;
end


eps = 1e-10 ; % defalut value for eps : 1e-10
if ((size(X,1) ~= size(X,2)) || (norm( X - X') > eps))
    error('usage: proj_spectral_box_sym(X,l,u) - X should be a symmetric matrix') ;
end

if (l > u) 
    error('Set is infeasible') ;
end

X = 0.5 * (X + X');

[V,D] = eig(X) ;

newD = diag(min ( max( diag(D),l),u)) ;
out = V * newD *V' ;
