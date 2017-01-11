function [ algs ] = sgd_solver_list(category)
% Return list of solvers.
%
% Inputs:
%       category    category to be returned. 
% Output:
%       algs        list of solvers in the category
%
% This file is part of GDLibrary.
%
% Created by H.Kasai on Nov. 0, 2016


    % supported algorithms by ‚SGDLibrary
    basic_algs = {'SGD'};
    vr_algs = {'SVRG','SAG','SAGA'};
    qnewton_algs = {'SQN','SVRG-SQN','SVRG-LBFGS','SS-SVRG', ...
        'oBFGS-Inf','oBFGS-Lim','Reg-oBFGS-Inf','Reg-oBFGS-Lim','Damp-oBFGS-Inf','Damp-oBFGS-Lim'};
    adagrad_algs = {'AdaGrad','RMSProp','AdaDelta','Adam','AdaMax'};
    else_algs = {'SVRG-BB'};
    
    switch category
        case 'BASIC' 
            algs = basic_algs;        
        case 'VR' 
            algs = vr_algs;
        case 'QN' 
            algs = qnewton_algs;
        case 'AD' 
            algs = adagrad_algs;            
        case 'ALL'
            %algs = [basic_algs, vr_algs, qnewton_algs, adagrad_algs, else_algs];
            algs = [basic_algs, vr_algs, qnewton_algs, adagrad_algs];
        otherwise
    end
    
end
