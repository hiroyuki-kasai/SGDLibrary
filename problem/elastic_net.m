classdef elastic_net
% This file defines the Elastic Net problem class. 
%
% Inputs:
%       A           dictionary matrix of size dxn.
%       b           observation vector of size dx1.
%       lambda1     l1-regularized parameter. 
%       lambda2     l2-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/2 * || A * w - b ||_2^2 + 1/2 * lambda2 * || w ||_2^2 + lambda1 * || w ||_1 ).
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Apr. 17, 2017
% Modified by H.Kasai on Mar. 25, 2017


    properties
        name;    
        dim;
        samples;
        lambda1;
        lambda2;
        d;
        n;
        A;
        b;
        Atb;
        AtA_l2;
        L;
        prox_flag;        
    end
    
    
    methods
        function obj = elastic_net(A, b, varargin)  
            
            obj.A = A;
            obj.b = b;
            
            if nargin < 3
                obj.lambda1 = 0.1;
                obj.lambda2 = 0.1;                
            elseif nargin < 4
                obj.lambda2 = varargin{1};                
            else
                obj.lambda1 = varargin{1};
                obj.lambda2 = varargin{2};                  
            end 
            
            if obj.lambda1 > 0 || obj.lambda2 > 0
                obj.prox_flag = true;
            else
                obj.prox_flag = false;
            end
            
            obj.d = size(obj.A, 2);
            obj.n = size(obj.A, 2);

            obj.name = 'elastic net';    
            obj.dim = obj.d;
            obj.samples = obj.n;

            obj.AtA_l2 = obj.A'*obj.A + obj.lambda2*eye(obj.n);

            obj.Atb = obj.A'*obj.b;


            fprintf('Calculated Lipschitz constant (L), i.e., max(eig(AtA)), .... ')
            obj.L = eigs(obj.A'*obj.A, 1);
            fprintf('is L=%f.\n', obj.L);
        end
    
        function v = prox(obj, w, t) % l1_soft_thresh
            v = soft_thresh(w, t * obj.lambda1);
        end    

        function f = cost(obj, w)
            reg = obj.reg(w);
            f = 1/2 * sum((obj.A * w - obj.b).^2) + 1/2 * obj.lambda2 * norm(w,2)^2 + obj.lambda1 * reg;
        end

        function r = reg(obj, w)
            r = norm(w, 1);
        end

        function r = residual(obj, w, i)
            %r = - A * w + b;
            %r = - AtA_l2 * w + Atb; 
            r = - (obj.A(:, i)' * obj.A + obj.lambda2) * w + obj.A(:, i)'*obj.b; 
        end

        function f = cost_batch(obj, w, indices)
            error('Not implemted yet.');        
        end

        function g = full_grad(obj, w)
            %g = A' * (A * w + lambda2 - b);
            g = obj.AtA_l2 * w - obj.Atb;
        end

        function g = grad(obj, w, indices)
            error('Not implemted yet.');
        end

        function h = hess(obj, w, indices)
            error('Not implemted yet.');        
        end

        function h = full_hess(obj, w)
            h = obj.AtA_l2;       
        end

        function hv = hess_vec(obj, w, v, indices)
            error('Not implemted yet.');
        end
        
    end
end

