function [A, b, x0, lambda, lambda_max] = generate_lasso_data(n, d, k, noise_level)

    [A,~] = qr(randn(n,d),0);                   
    A = A';                                    
    p = randperm(n); 
    p = p(1:k);                                 % select location of k nonzeros
    x0 = zeros(n,1); 
    x0(p) = randn(k,1);                         
    b = A*x0 + noise_level*randn(d, 1);                  % add random noise   
    lambda_max = norm( A'*b, 'inf' );
    lambda = 0.1*lambda_max;

end

