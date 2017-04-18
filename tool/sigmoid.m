function [y] = sigmoid(x)
% Sigmoid function.
%
% This file is part of SGDLibrary.
%
% Created H.Kasai on Oct. 17, 2016

    y = 1.0 ./ (1.0 + exp(-x));
end

