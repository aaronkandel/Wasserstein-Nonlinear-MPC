function J = obj_C(alpha, thet, mu, N)

J = sqrt(abs((1/(2*alpha))*(1+log(1/N* sum(exp(alpha*vecnorm(thet-mu, 1,1).^2))))));


end

