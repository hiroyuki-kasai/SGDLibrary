function out=bisection(f,min_val,max_val,eps)
%INPUT
%================
%f ................... a scalar function
%lb ................. the initial lower bound
%ub ................ the initial upper bound
%eps .............. tolerance parameter
%OUTPUT
%================
% z ................. a root of the equation f(x)=0

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

saved_fmin_val = f(min_val) ;
saved_fmax_val = f(max_val) ;

if (saved_fmin_val * saved_fmax_val >0)
    error('f(lb)*f(ub)>0')
end

if (min_val > max_val)
    error ('minimal value is bigger than maximal_value\n') ;
end

iter=0;
while (max_val-min_val>eps)
    
    %if the function is linear in this range, find the root
    r = (max_val-(saved_fmax_val/saved_fmin_val) * min_val) / (1-(saved_fmax_val/saved_fmin_val)) ;
    saved_froot = f(r) ;
    if (abs(saved_froot) < eps)
        out = r;
        return ;
    end
    
    iter=iter+1;
    mid=(min_val+max_val)/2;
    changed_limits = false ; %haven't changed the limits yet
    
    %checking if r maybe be one of the new range limits
    if (r > mid)
        %r may be the the new min_val
        if (saved_froot * saved_fmax_val <0 )
            min_val = r ;
            saved_fmin_val = saved_froot ;
            changed_limits = true ;
        end
    else
        %r may be the new max_val
        if ( saved_froot * saved_fmin_val < 0)
            max_val = r ;
            saved_fmax_val = saved_froot ;
            changed_limits = true ;
        end
    end
    
    if (~changed_limits)
        %didn't use r
        saved_fmid = f(mid) ;
        if(saved_fmin_val* saved_fmid >0)
            min_val=mid;
            saved_fmin_val = saved_fmid ;
        else
            max_val=mid;
            saved_fmax_val = saved_fmid ;
        end
    end
end

if (~changed_limits)
    out=mid ;
else
    out = r;
end


