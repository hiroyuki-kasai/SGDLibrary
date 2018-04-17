function func = loss_functions(name)

    if strcmp(name, 'logistic_loss')
        func = @logistic_loss;
    end
    
    if strcmp(name, 'logistic_loss_gradient')
        func = @logistic_loss_gradient;
    end
    
    if strcmp(name, 'logistic_loss_hessian')
        func = @logistic_loss_hessian;
    end
    
    if strcmp(name, 'logistic_loss_Hv')
        func = @logistic_loss_Hv;
    end
    

    function l = logistic_loss(w, X, Y, alpha)
        [n ,d] = size(X);
        z = X*w;
        l= - ((log_phi(z)'*Y) + (ones(n,1)-Y)'*(one_minus_log_phi(z)))/ n;
        l = l + 0.5 * alpha * (norm(w).^2);
    end

    function grad = logistic_loss_gradient(w, X, Y, alpha)
        [n ,d] = size(X);
        z = X*w;  
        h = phi(z);
        grad = X'*(h-Y)/n;
        grad = grad + alpha * w;
    end


    function H = logistic_loss_hessian(w, X, Y, alpha)
        [n ,d] = size(X);
        z = X*w;
        q = phi(z);
        h = q .* ( ones(n,1) - phi(z));
        H = (X'*(h .* X)) / n;
        H = H + alpha * eye(d, d);
    end

    function out = logistic_loss_Hv(w,X, Y, v, alpha)
        [n ,d] = size(X);
        z = X*w; 
        z = phi(-z);
        d_binary = z.*(ones(n,1) - z);
        wa = d_binary .* (X * v);
        Hv = X'* wa / n;
        out = Hv + alpha * v;
    end
    % ######## Auxiliary Functions: robust Sigmoid, log(sigmoid) and 1-log(sigmoid) computations ########

    function out = phi(t) % Author: Fabian Pedregosa
        % logistic function returns 1 / (1 + exp(-t))
        idx = t>0;
        [t_size, ~] = size(t);
        one = ones(t_size,1);
        out = zeros(t_size,1);
        out(idx) = 1.0 ./ (1 + exp(-t(idx)));
        exp_t = zeros(t_size,1);
        exp_t(~idx) = exp(t(~idx));
        out(~idx) = exp_t(~idx) ./ (one(~idx) + exp_t(~idx));
    end

    function out = log_phi(t)
    % log(Sigmoid): log(1 / (1 + exp(-t)))
        idx = t>0;
        [t_size, ~] = size(t);
        out = zeros(t_size,1);
        out(idx) = -log(1 + exp(-t(idx)));
        out(~idx) = t(~idx) - log(1 + exp(t(~idx)));
    end


    function out = one_minus_log_phi(t)
    % log(1-Sigmoid): log(1-1 / (1 + exp(-t)))
        idx = t>0;
        [t_size, ~] = size(t);
        out = zeros(t_size,1);
        out(idx) = -t(idx) -log(1 + exp(-t(idx)));
        out(~idx) = -log(1 + exp(t(~idx)));
    end

end
