function [lambda, hout] = triSearch(sig, lb, ub, epsilon, thet)
N = length(thet);

flb = 1/N;
fub = ub*epsilon;

% [a, u, v, b]

while (ub-lb) > 1e-4
    lambda_u = lb + (ub-lb)/3;
    lambda_v = lb + 2*(ub-lb)/3;
    
    % First:
    h1 = lambda_u*epsilon;
    
    for i = 1:N
        inh(i) = sig - norm(thet(i), inf); % abs(thet(i));%
        inh2(i) = max(inh(i), 0);
        outh(i) = 1 - lambda_u*inh2(i);
        outh2(i) = max(outh(i), 0);
    end
    
    h2 = (1/N)*sum(outh2);

    h_u = h1 + h2;

    % Second:
    h1 = lambda_v*epsilon;
    
    for i = 1:N
        inh(i) = sig - norm(thet(i), inf);
        inh2(i) = max(inh(i), 0);
        outh(i) = 1 - lambda_v*inh2(i);
        outh2(i) = max(outh(i), 0);
    end
    
    h2 = (1/N)*sum(outh2);

    h_v = h1 + h2;
    
    
    if h_u < h_v
        ub = lambda_v;
    else
        lb = lambda_u;
    end
    
    
end

lambda = lambda_v;
hout = h_v;


end

