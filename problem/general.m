classdef general
% This file defines a general problem class
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
% Modified by H.Kasai on Nov. 02, 2020


    properties
        f;
        g;
        h;
        hv;
        d;
        econst;     % equality constraints
        g_econst;   % gradient of equality constraints
        ineconst;   % inequality constraints
        g_ineconst; % gradient of inequality constraints
        lb;         % Upper bound of constraints
        ub;         % Lower bound of constraints
        lambda;
        ineconst_flag;  % indicate whehter the problem has inequality constraints, i.e., there exists econst and/or lb/ub;
        
        name;    
        dim;
        edim;       % dimension of equality constraints
        inedim;     % dimension of inequality constraints
        samples;
        hessain_w_independent;
    end
    
    methods
        function obj = general(f, g, h, hv, d, econst, g_econst, ineconst, g_ineconst, lb, ub, lambda)
            
            obj.f = f;
            obj.g = g;
            obj.h = h;
            obj.hv = hv;
            obj.d = d;
            obj.econst = econst;
            obj.g_econst = g_econst;               
            obj.ineconst = ineconst;
            obj.g_ineconst = g_ineconst;   
            obj.lb = lb;   
            obj.ub = ub;              
            obj.lambda = lambda;
                        
            obj.name = 'general';  
            obj.dim = obj.d;
            obj.samples = 0; 
            obj.hessain_w_independent = false;
            

            if ~isempty(obj.econst) && ~isempty(obj.g_econst)
                tmp_x = zeros(d,1)';
                obj.edim = length(obj.econst(tmp_x));
            else
                obj.edim = 0;                
            end
            
            if ~isempty(obj.ineconst) && ~isempty(obj.g_ineconst)
                tmp_x = zeros(d,1)';
                obj.inedim = length(obj.ineconst(tmp_x));
                obj.ineconst_flag = true;                
            else
                obj.inedim = 0;
                obj.ineconst_flag = false;                
            end 
            
%             if isempty(obj.lb)
%                 obj.lb = -inf;
%             end
%             
%             if isempty(obj.ub)
%                 obj.ub = inf;
%             end 

            if ~isempty(obj.lb) || ~isempty(obj.ub)
                obj.ineconst_flag = true;
            end
            
            
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
        
        function [econ, gecon] = equality_consts(obj, x)

            econ    = obj.econst;
            gecon   = obj.g_econst;            

        end
        
        function [inecon, ginecon] = inequality_consts(obj, x)

            inecon    = obj.ineconst;
            ginecon   = obj.g_ineconst;            

        end        
        
    end

end

