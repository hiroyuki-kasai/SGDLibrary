classdef general
% This file defines quadratic problem class
%
% Inputs:
%       f           cost function
%       g           gradient
%       h           hessian
%       hv          hessian-product 
%       d           dimension
%
% Output:
%       Problem     problem instance. 
%
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Oct. 29, 2016
% Modified by H.Kasai on Mar. 25, 2018


    properties
        f;
        g;
        h;
        hv;
        d;
        name;    
        dim;
        samples;
        hessain_w_independent;
    end
    
    methods
        function obj = general(f_in, g_in, h_in, hv_in, d)
            
            obj.f = f_in;
            obj.g = g_in;
            obj.h = h_in;
            obj.hv = hv_in;
            obj.d = d;

            obj.name = 'general';  
            obj.dim = obj.d;
            obj.samples = 0; 
            obj.hessain_w_independent = false;
            
        end

        function f = cost(obj, x)

            f = obj.f(x);
        end

        function g = grad(obj, x, indices)

            g = obj.g(x);

        end        

        function g = full_grad(obj, x)

            g = obj.g(x);

        end

        function h = hess(obj, x)

            h = obj.h(x);

        end

        function h = full_hess(obj, x)

            h = obj.hess(x);

        end

        function h = hess_vec(obj, x)

            h = obj.hv(x);

        end
    end

end

