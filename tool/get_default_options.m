function options = get_default_options(d)

    options.stepsizefun     = @stepsize_alg;
    options.step_alg        = 'fix';
    options.step_init       = 0.01;
    options.lambda          = 0.1;    
    options.tol_optgap      = 1.0e-12;
    options.batch_size      = 10;
    options.max_epoch       = 100;
    %options.w_init          = randn(d,1);
    options.f_opt           = -Inf;
    options.permute_on      = 1;
    options.verbose         = 0;
    options.store_w         = false;

end

