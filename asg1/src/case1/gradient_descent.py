def gradient_descent(x, fun, args):
    x_init_line = x.copy()

    iterations, learning_rate = args[0], args[1]

    best_line = x_init_line.copy().flatten()

    convergence_points = []

    for iteration in range(iterations):
        next_line = best_line.copy()

        new_objective_value, new_gradient_array = fun(next_line)

        for j in range(2, len(next_line) - 2):
            next_line[j] = (next_line[j] - (learning_rate * new_gradient_array[j])) #Update step

        if fun(next_line)[0] < fun(best_line)[0]:
            best_line = next_line.copy()

        convergence_points.append(fun(best_line)[0])

    return best_line, convergence_points