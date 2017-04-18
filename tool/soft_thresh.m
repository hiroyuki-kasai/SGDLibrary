function [ v ] = soft_thresh(w, t)
    %v = max( 0, x - t*opts.rho ) - max( 0, -x - t*opts.rho );

    v = sign(w) .* max(abs(w) - t,0);
end

