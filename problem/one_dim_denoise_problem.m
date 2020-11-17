classdef one_dim_denoise_problem


    properties
        name;    
        dim;
        samples;
        lambda;
        Lap;
        LaptLap;
        b;
        n;
        d;
        prox_flag;
    end
    
    methods
        function obj = one_dim_denoise_problem(b, lambda, varargin)

            obj.name = 'one_dim_denoise_problem';  
            obj.b = b;
            obj.lambda = lambda;
            obj.samples = length(b);
            
            obj.n = obj.samples;
            obj.d = obj.samples;
            obj.dim = 1;
            obj.samples = obj.n;            

            obj.Lap = zeros(obj.d-1, obj.d);
            for i = 1 : obj.d-1
                obj.Lap(i,i) = 1;
                obj.Lap(i,i+1) = -1;
            end 
            obj.LaptLap = obj.Lap' * obj.Lap;
            
            obj.prox_flag = false;
        end

%         function v = prox_denoise(w, t)
%             
%             v = soft_thresh(w, t * obj.lambda);
%             
%         end    

        function f = cost(obj, w)
            
            f = obj.loss(w) + obj.lambda * obj.reg(w);
            
        end
        
        function l = loss(obj, w)
            
            l = 1/2 * norm(obj.b-w)^2;
        end          

        function r = reg(obj, w)
            
            % Lw = obj.Lap * w;
            % r = (Lw'*Lw)/2;
            % r = w'L'Lw/2
            r = w' * obj.LaptLap * w / 2;
            
        end
        
        function r = differentiable_reg(obj, w)
            
            r = w' * obj.LaptLap * w / 2;

        end
            

        function r = residual(obj, w)
            r = - (obj.b - w);
        end

        function g = full_grad(obj, w)
            g = - (obj.b - w) + obj.lambda * obj.reg_grad(w);
        end

        function g = grad(obj, w, indices)
            g = obj.full_grad(w);
        end
        
        function rg = reg_grad(obj, w)
            
            rg = obj.Lap' * obj.Lap * w;

         end         

        function h = hess(obj, w, indices)
            error('Not implemted yet.');        
        end

        function h = full_hess(obj, w)
            error('Not implemted yet.');
        end

        function hv = hess_vec(obj, w, v, indices)
            error('Not implemted yet.');
        end
        
        
        function w_opt = calc_solution(obj, w_init, options)
            
            w_opt = (eye(obj.d) + obj.lambda * obj.LaptLap) \ obj.b;
            
        end
        
    end


end

