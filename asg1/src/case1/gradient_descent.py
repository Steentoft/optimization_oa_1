def gradient_descent(x, fun, iterations=100, learning_rate=0.01):
    x_init_line = x.copy()

    best_line = x_init_line.copy()[1:-1].flatten()

    for iteration in range(iterations):
        next_line = best_line.copy()

        new_objective_value, new_gradient_array = fun(next_line)

        for j in range(len(next_line)):
            next_line[j] = (next_line[j] - (learning_rate * new_gradient_array[j])) #Update step

        if fun(next_line)[0] < fun(best_line)[0]:
            best_line = next_line.copy()

    return best_line