def gradient_descent(x, fun, learning_rate=0.001):
    new_x = x.copy()
    new_objective_value, new_gradient_array = fun(new_x)

    for j in range(len(new_x)):
        new_x[j] = (new_x[j] - (learning_rate * new_gradient_array[j])) #Update step

    return new_x