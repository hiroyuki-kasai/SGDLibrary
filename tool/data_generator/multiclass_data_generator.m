function [data] = multiclass_data_generator(n, d, l, std)
% Data generator for multiclass classifier problem.
%
% Inputs:
%       n               number of samples per class.
%       d               number of dimensions.
%       l               number of classes.
%       std             standard devision of gaussian probability.
% Output:
%       data            data set
%       data.x_train    train data of x of size d x n*l.
%       data.y_train    train data of y of size l x n*l.
%       data.x_test     test data of x of size d x n*l.
%       data.y_test     test data of y of size l x n*l.
%
% This file is part of SGDLibrary.
%
% Created H.Kasai on Oct. 25, 2016

    x = [];
    y = [];

    % generate the data
    for i = 1:l
      % set centre
      mu = rand(d, 1); 
      % covariance matrix
      std = eye(d) * std; 
      xi = std * randn(d, 2*n) + repmat(mu, 1, 2*n);

      x = [x xi]; 
      y = [y (i * ones(1, 2*n))];
    end

    
    n_train = length(y)/2; 
    % shuffle data
    perm_idx = randperm(length(y)); 
    
    % split data into train and test data
    % train data
    train_indices = perm_idx(1:n_train); 
    data.x_train = x(:,train_indices); 
    data.y_train = y(train_indices); 

    % test data    
    test_indices = perm_idx(n_train+1:end);
    data.x_test = x(:,test_indices); 
    data.y_test = y(test_indices); 
    
end
