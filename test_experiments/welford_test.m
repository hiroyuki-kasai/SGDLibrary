function welford_test()
    clc;
    clear;
    close all;

    d = 50;
    N = 10000;
    data = randn(d,N);
    v_sol = var(data') * var(data')';
    fprintf('answer: %.16e\n', v_sol);
    

    M = zeros(d,1);
    S = 0;

    for k=1:N
        x = data(:,k);
        oldM = M;
        M = M + (x-M)/k;
        S = S + (x-M)'*(x-oldM);
    end

    v = S/(N-1);
    fprintf('result: %.16e\n',v);    
    
    
    M = zeros(d,1);
    S = 0;

    step = 10;
    for k=1:step:N

        start_idx = k;
        end_idx = k + step - 1;
        %fprintf('[%d]: start_idx=%d end_idx=%d\n', k, start_idx, end_idx);        
        x = data(:,start_idx:end_idx);
        oldM = M;
        M_rep = repmat(M, [1 step]);
        M = M + sum((x-M_rep),2)/end_idx;
        
        M_rep = repmat(M, [1 step]);
        oldM_rep = repmat(oldM, [1 step]);
        S = S + trace((x-M_rep)'*(x-oldM_rep));
    end

    v = S/(N-1);
    fprintf('result: %.16e\n',v);      
    
end

 

  