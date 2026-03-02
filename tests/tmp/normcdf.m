function p = normcdf(x)
    p = 0.5 * erfc(-x / sqrt(2));
end
