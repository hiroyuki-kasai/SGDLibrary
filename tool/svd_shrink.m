function [ v ] = svd_shrink(w, t, dim)
    L = reshape(w, dim);
    [U,S,V] = svd(L,'econ');
    s = diag(S);
    S = diag(sign(s) .* max(abs(s) - t,0));
    L = U*S*V';
    v = L(:);
end

