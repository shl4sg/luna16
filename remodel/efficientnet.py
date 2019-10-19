from keras_efficientnets import *
from keras_efficientnets.optimize import optimize_coefficients
from keras_efficientnets.optimize import get_compound_coeff_func


results = optimize_coefficients(phi=1., max_cost=2.0, search_per_coeff=10, tol=1e-10)
cost_func = get_compound_coeff_func(phi=1.0, max_cost=2.0)

print("Num unique configs = ", len(results))
for i in range(20):  # print just the first 10 results out of 125 results
    print(i + 1, results[i], "Cost :", cost_func(results[i]))