def gradient_descent(x, fun, args):
    x_init_line = x.copy()

    iterations, learning_rate = args[0], args[1]

    best_line = x_init_line.copy().flatten()

    convergence_points = []

    for iteration in range(iterations):

        new_objective_value, new_gradient_array = fun(best_line)

        for j in range(2, len(best_line) - 2):
            best_line[j] = (best_line[j] - (learning_rate * new_gradient_array[j]))

        convergence_points.append((iteration,fun(best_line)[0]))

    return best_line, convergence_points