function out=proj_two_halfspaces(x,a1,b1,a2,b2)
%PROJ_TWO_HALFSPACES computes the orthogonal projection onto intersection of two halfspaces {x:<a1,x><=b1,<a2,x><=b2}
%
%  Usage: 
%  out = PROJ_TWO_HALFSPACES(x,a1,b1,a2,b2)
%  ===========================================
%  Input:
%  x - point to be projected (vector/matrix)
%  a1 - vector/matrix
%  b1 - scalar
%  a2 - vector/matrix
%  b2 - scalar
%  ===========================================
%  Assumptions:
%  a1 and a2 are linearly independent
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

if (nargin < 5)
    error ('usage: proj_two_halfspaces(x,a1,b1,a2,b2)') ;
end
eps = 1e-10;

alpha = trace(a1'*x)-b1;
beta = trace(a2'*x)-b2 ;

if ((alpha <= 0) && (beta <=0))
    out=x;
else
    pai = trace(a1'*a2) ;
    mu = trace(a1'* a1) ;
    nu = trace(a2'* a2) ;
    ro = mu*nu - pai^2 ;
    if ( (abs(mu) < eps) || (abs(nu) < eps) || (abs(ro) < eps))
         error('usage: proj_two_halfspaces(x,a1,b1,a2,b2) - vectors a1 and a2 should be linearly independent ') ;
    end

    if ((alpha <= pai*beta/nu) && (beta >0))
        out = x - beta/nu*a2 ;
    else
        if ((beta <= pai*alpha/mu) && (alpha >0))
            out = x - alpha/mu*a1 ;
        else    
            out = x + alpha/ro*(pai*a2-nu*a1)+beta/ro*(pai*a1-mu*a2) ;
        end
        
    end
    
    
end


