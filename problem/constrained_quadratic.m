classdef constrained_quadratic
% This file defines constrained quadratic problem class
%
% Inputs:
%           Q       a positive definite matrix of size nxn
%           p       a column vector of size n
%           A       a matrix of size nxm
%           b       a column vector of size m
%           G       a matrix of size nxp
%           h       a column vector of size p
%
% Output:
%           Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(x) = 1/2 * x^T * Q * x - p^T * x.
%           s.t.
%           A * x = b, G * x <= h
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 22, 2020

    properties
        d;
        n;  
        name;
        dim;
        samples;
        Q;
        p;
        A;
        b;
        G;
        h;

    end
    
    methods
        function obj = constrained_quadratic(Q, p, A, b, G, h)
            obj.d = length(p);
            obj.n = obj.d;

            obj.name = 'constrained_quadratic';    
            obj.dim = obj.d;
            obj.samples = obj.n;    
            obj.Q = Q;     
            obj.p = p;  
            obj.A = A;     
            obj.b = b; 
            obj.G = G;     
            obj.h = h;             
        end
        

        function v = prox(obj, w, t)  
            % project y onto the feasible set Ax <= b, i.e. nearest feasible point x to y
            % minimize 0.5||x-y||^2
            % s.t.     Ax <= b   

            % equivalent to
            % min J   = 0.5 x'x - x'y + 0.5 y'y
            % st. Ax <= b

            %n = length(y);
            H = eye(obj.d);
            w0 = w;
            qp_options = optimset('Display','off');
            v = quadprog(H,-w,obj.G,obj.h,[],[],[],[],w0, qp_options);
        end


        function f = cost(obj, x)

            f = 1/2 * x' * obj.Q * x - obj.p' * x;
        end

        function g = grad(obj, x)

            g = obj.Q * x - obj.p;

        end  

        function g = full_grad(obj, x)

            g = obj.grad(x);

        end 

        function h = hess(obj, x)

            h = obj.Q;

        end

        function h = full_hess(obj, x)

            h = obj.hess(x);

        end
    end

end

