function out = proj_spectahedron_box(X,r,l,u,eq_flag)
%PROJ_SPECTAHEDRON_BOX computes the orthogonal projection onto the spectahedron {X : Tr(X) = r, X psd, l<=eig(X)<=u} or
%                                                onto the full spectahedron {X : Tr(X)<= r, X psd, l<=eig(X)<=u}
%
%  Usage: 
%  out = PROJ_SPECTAHEDRON_BOX(X,[r],[l],[u],[eq_flag])
%  ===========================================
%  Input:
%  X - matrix to be projected
%  r - positive scalar [default: 1]
%  l - lower bound (scalar) [default: 0]
%  u - upper bound (scalar) [default: inf]
%  eq_flag - a flag that determines whether the projection is onto the spectahedron ('eq') or 
%                 the full spectahedron ('ineq') [defualt: 'eq']
%  ===========================================
%  Assumptions:
%  X symmetric
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

%reading the user X and setting defalut values when required.
if (nargin < 1)
    error ('usage: spectahedron_box(X,[r],[l],[u],[eq_flag])') ;
end

if (( nargin < 5) || (isempty(eq_flag)) )
    %eq_flag is not given, setting to default value: true
    eq_flag = 'eq' ;
end

if ((nargin < 4) || (isempty(u)))
    %u is not given, setting to default value: inf
    u=inf ;
end

if ((nargin < 3) || (isempty(l)))
    %l is not given, setting to default value: 0
    l=0 ;
end

if ((nargin < 2) || (isempty(r)))
    %r is not given, setting to default value: 1
    r = 1 ;
end

if ( r <= 0 ) 
   error('Set is infeasible') ;
end

eps = 1e-10 ; % defalut value for eps : 1e-10
if ((size(X,1) ~= size(X,2)) || (norm( X - X') > eps))
    error('usage: spectahedron_box(X,[r],[eq_flag]) - X should be a symmetric matrix') ;
end

X = 0.5 * (X + X');

[V,D] = eig(X) ;

if (strcmp(eq_flag,'eq') == 1)
    %call proj_simplex with eq
   out = V * diag(proj_hyperplane_box(diag(D),ones(length(X),1),r,l,u)) * V' ; 
   out=(out+out')/2;
else
    if (strcmp(eq_flag,'ineq') == 1)
        %call proj_simplex with ineq
        out = V * diag((proj_halfspace_box(diag(D),ones(length(X),1),r,l,u))) * V' ;
        out=(out+out')/2;
    else
        error('usage: spectahedron(X,[r],[eq_flag]) - eq_flag should be either eq or ineq') ;
    end
end

