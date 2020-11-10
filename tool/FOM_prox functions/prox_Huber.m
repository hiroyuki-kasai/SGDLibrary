function out=prox_Huber(x,mu,alpha)
%PROX_HUBER computes the proximal operator of the function alpha*H_(mu) 
%                      where H_(mu)=huber function with parameter mu
%
%  Usage: 
%  out = PROX_HUBER(x,mu,alpha)
%  ===========================================
%  INPUT:
%  x - point to be projected (vector/matrix)
%  mu - positive scalar
%  alpha - positive scalar
%  ===========================================
%  Assumptions:
%  mu is positive
%  ===========================================
%  Output:
%  out - proximal operator at x

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
    error ('usage: prox_huber(x,mu,alpha)') ;
end

if (alpha < 0)
    error('usage: prox_huber(x,mu,alpha) - alpha should be positive')
end

if (mu < 0)
    error('usage: prox_huber(x,mu,alpha) - mu should be positive')
end

out = x*( 1  - alpha/max(norm(x,'fro'),mu+alpha))   ;

end

