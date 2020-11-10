function out=proj_Lorentz(x)
%PROJ_LORENTZ computes the orthogonal projection onto the Lorentz cone {x:||x(1,..,n)||<=x(n+1)}
%
%  Usage: 
%  out = PROJ_LORENTZ(x)
%  ===========================================
%  Input:
%  x - (n+1)-length vector to be projected 
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

%reading the user 
if (nargin < 1)
    error ('usage: proj_Lorentz(x)') ;
end
n = length(x)-1;
s = x(n+1) ;
x = x(1:n) ;

if (norm(x,s) >= abs(s))
    outx = (norm(x,2) + s)/(2*norm(x,2)) * x;
    outs = (norm(x,2) + s)/2 ;
else
    if (norm(x,2) <= s)
        outx = x ;
        outs = s ;
    else
        % s < norm(x,2) < -s
        outx = zeros(n,1) ;
        outs = 0 ;
    end
end

out = [outx; outs] ;

end

