function [Tensor_Y, Tensor_Y_Noiseless] = generate_synthetic_tensor(tensor_dims, rank, inverse_snr, data_subtype)
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017
 
    %disp('# Generating synthetic dataset ....');

    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    total_slices    = tensor_dims(3);

    if strcmp(data_subtype, 'Static')

        %disp('## Static dataset ....');    

        A=rand(rows, rank);
        B=rand(cols, rank);
        C=rand(total_slices, rank);

        % Create observed tensor that follows PARAFAC model
        Tensor_Y_Noiseless = zeros(rows,cols,total_slices);
        for k=1:total_slices
            Tensor_Y_Noiseless(:,:,k)=A*diag(C(k,:))*B.';
        end
        
    elseif strcmp(data_subtype, 'Dynamic')
        
        %disp('## Dynamic dataset ....'); 
        
        A = randn(rows, rank);
        B = randn(cols, rank);
        C = randn(total_slices, rank);
        
        % Create observed tensor that follows PARAFAC model
        Tensor_Y_Noiseless = zeros(rows,cols,total_slices);
        for k=1:total_slices
            
            jj = rem(k + rank-2, rank-1) + 1;
            R = eye(rank);
            R(jj:jj+1,jj:jj+1) = [cos(angle) -sin(angle); sin(angle) cos(angle)];


            A = A * R;
            B = B * R;
            Tensor_Y_Noiseless(:,:,k)=A*diag(C(k,:))*B.';
        end
        
    else
        
        %disp('## Switching dataset ....'); 
        
        rank = 2;

        A = randn(rows, rank);
        B = randn(cols, rank);
        C = randn(total_slices, rank);
        
        %angle = pi/360;
        R = [cos(angle) -sin(angle); sin(angle) cos(angle)];

        % Create observed tensor that follows PARAFAC model
        Tensor_Y_Noiseless = zeros(rows,cols,total_slices);
        for k=1:total_slices
            if rem(k, switch_period) == 0
                A = A * R;
                B = B * R;
            end

            Tensor_Y_Noiseless(:,:,k)=A*diag(C(k,:))*B.';
        end            



    end


    %% Adding noise
%     snr_val = 35;
%     aux_var = realpow(10, snr_val / 20);
%     std_noise = std(Tensor_Y_Noiseless(:)) / aux_var;
%     noise = std_noise * randn(size(Tensor_Y_Noiseless));
%     Tensor_Y = Tensor_Y_Noiseless + noise;  

    %SNR=inf;         % To add noise or not on initial tensor. 
                     % Choose SNR=inf for a noise free model
    %Noise_tens=randn(I,J,K);
    %sigma=(10^(-SNR/20))*(norm(reshape(X,J*I,K),'fro')/norm(reshape(Noise_tens,J*I,K),'fro'));
    %X=X+sigma*Noise_tens;

    %noise = [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
    %inverse_snr = noise(noise_level);
    Tensor_Noise = randn(size(Tensor_Y_Noiseless));
    Norm_Tensor_Y_Noiseless = norm(reshape(Tensor_Y_Noiseless, rows*cols, total_slices),'fro');
    Norm_Tensor_Noise = norm(reshape(Tensor_Noise, rows*cols, total_slices),'fro');


    Tensor_Y = Tensor_Y_Noiseless + (inverse_snr * Norm_Tensor_Y_Noiseless / Norm_Tensor_Noise) * Tensor_Noise; % entries added with noise
end


