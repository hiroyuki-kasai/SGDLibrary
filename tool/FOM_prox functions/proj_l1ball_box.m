function out = proj_l1ball_box(x,w,r,u)
%PROJ_L1BALL_BOX computes the orthogonal projection onto the intersection of an l1 ball and a box
%                                                  {x: norm(w(:).*x(:),1)<=r, -u<=x<=u} 
%                                               
%  Usage: 
%  out = PROJ_L1BALL_BOX(x,[w],[r],[u])
%  =============================================================
%  INPUT:
%  x - point to be projected (vector/matrix)
%  w - weights vecotr - [default: 1]
%  r - positive scalar - [default: 1]
%  u - box paramters (scalar/vector/matrix)- [default: Inf]
%  ==============================================================
%  Assumptions:
%  r,u >= 0
%  w >= 0
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

%reading the user x and setting defalut values when required.
if (nargin < 1)
    error ('usage: proj_l1_inter_box(x,[w],[r],[u])') ;
end

if (nargin < 4)
    %box parameters are not given, setting to defalut value :inf
    u = inf*(ones(size(x))) ;
end

if ((nargin < 3) || (isempty(r)))
    %ball radius is not given, setting to defalut value :1
    r = 1;
end

if ((nargin < 2) || (isempty(w)))
    %weight vector is not given, setting to default value: ones
    w=ones(size(x)) ;
end

if  ((r < 0) || (min(min(u)) < 0))
      error('Set is infeasible') ;
end

if (min(min(w)) < 0)
    error('usage: proj_l1_inter_box(x,[w],[r],[u]) - w should be a non-negative vector or matrix');
end

%checking the simple projection 
out = min(max(x,-u),u) ;
if ((sum(sum(w .* abs (out)))) <= r)
    return;
end

%defining f on lambda - to be used by the bisetion

f= @(lam)   trace(w' * abs(min(max(abs(x)-lam*w,0),u) .* sign(x)))- r;
lambda_min = 0;

lambda_max = 1;
while(f(lambda_max)>0)
    lambda_max = lambda_max *2  ;
end

eps = 1e-10 ;
final_lam = bisection(f,lambda_min,lambda_max,eps) ;
out=  min(max(abs(x)-final_lam*w,0),u) .* sign(x) ;
