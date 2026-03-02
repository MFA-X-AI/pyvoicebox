function p = normpdf(x, mu, sigma)
    if nargin < 2, mu = 0; end
    if nargin < 3, sigma = 1; end
    z = (x - mu) ./ sigma;
    p = exp(-0.5 * z.^2) ./ (sigma * sqrt(2 * pi));
end
