classdef one_dim_tv_denoise_problem


    properties
        name;    
        dim;
        samples;
        lambda;
        sub_mode;
        Lap;
        LaptLap;
        b;
        n;
        d;
        prox_flag;
    end
    
    methods
        function obj = one_dim_tv_denoise_problem(b, lambda, sub_mode, varargin)

            obj.name = 'one_dim_tv_denoise_problem';  
            obj.b = b;
            obj.lambda = lambda;
            obj.sub_mode = sub_mode;
            obj.samples = length(b);
            
            obj.n = obj.samples;
            obj.d = obj.samples;
            obj.dim = obj.d;
            obj.samples = obj.n; 
            
            
            if nargin < 4
                obj.Lap = zeros(obj.d-1, obj.d);
                for i = 1 : obj.d-1
                    obj.Lap(i,i) = 1;
                    obj.Lap(i,i+1) = -1;
                end 
            else
                obj.Lap = varargin{1};
            end 
            obj.LaptLap = obj.Lap' * obj.Lap; 
            
            
            if strcmp(obj.sub_mode, 'l1')
                obj.prox_flag = true;
            else
                obj.prox_flag = false;
            end
        end
        
        function v = prox(obj, w, t)
            if strcmp(obj.sub_mode, 'l1')
                v = soft_thresh(w, t * obj.lambda);        
            else
                v = w;
            end
        end

        function f = cost(obj, w)
            
            f = obj.loss(w) + obj.lambda * obj.reg(w);
            
        end
        
        function l = loss(obj, w)
            
            l = 1/2 * norm(obj.b-w)^2;
        end          

        function r = reg(obj, w)
            
            if strcmp(obj.sub_mode, 'l2')
                % Lw = obj.Lap * w;
                % r = (Lw'*Lw)/2;
                % r = w'L'Lw/2
                r = w' * obj.LaptLap * w / 2;
            else
                r = norm(obj.Lap * w, 1);
            end
            
        end
        
        function r = differentiable_reg(obj, w)
            
            if strcmp(obj.sub_mode, 'l2')            
                r = w' * obj.LaptLap * w / 2;
            else
                r = 0;
            end

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
            
            if strcmp(obj.sub_mode, 'l2')            
                rg = obj.Lap' * obj.Lap * w;
            else
                rg = zeros(obj.d, 1);
            end

        end
        
        
        % calculate A(x)
        function v = grad_A_func(obj, w)
            v = obj.Lap * w;
        end        
        
        % calculate for input vector w: argmax_x { <x,w> - f(x) }
        % In this case, we calculate x' = Lap^T x + b.
        function v = grad_conj(obj, w)
            v = obj.Lap' * w + obj.b;
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

