function out = proj_psd(X)
%PROJ_PSD computes the orthogonal projection onto the cone of positive semidefinite matrics 
%                                     (psd cone)   {X: X psd }
%
%  Usage: 
%  out = PROJ_PSD(X)
%  ===========================================
%  INPUT:
%  X - matrix to be projected
%  ===========================================
%  Assumptions:
%  X is symmetric
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
    error ('usage: proj_psd(X)') ;
end

eps = 1e-10 ;   % defalut value for eps : 1e-10
if ((size(X,1) ~= size(X,2)) || (norm( X - X') > eps))
    error('usage: proj_psd(X) - X should be a symmetric matrix') ;
end

X = 0.5 * (X + X');

[V,D] = eig(X) ;

out = V * max(D,0) *V' ;
