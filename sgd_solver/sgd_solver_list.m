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
% Created by H.Kasai on Nov. 1, 2016
% Modified by H.Kasai on Sep. 28, 2017


    % supported algorithms by SGDLibrary
    basic_algs = {'SGD','SGD-CM','SGD-CM-NAG','AdaGrad','RMSProp','AdaDelta','Adam','AdaMax'};
    vr_algs = {'SVRG','SAG','SAGA','SARAH'};
    qnewton_algs = {'SQN','SVRG-SQN','SVRG-LBFGS','SS-SVRG', ...
        'oBFGS-Inf','oBFGS-Lim','Reg-oBFGS-Inf','Reg-oBFGS-Lim','Damp-oBFGS-Inf','Damp-oBFGS-Lim'};
    else_algs = {'SVRG-BB','IQN'};
    
    switch category
        case 'BASIC' 
            algs = basic_algs;        
        case 'VR' 
            algs = vr_algs;
        case 'QN' 
            algs = qnewton_algs;        
        case 'ALL'
            algs = [basic_algs, vr_algs, qnewton_algs, else_algs];
        otherwise
    end
    
end
