def gradient_descent(x, fun, args):
    x_init_line = x.copy()

    iterations, learning_rate = args[0], args[1]

    best_line = x_init_line.copy().flatten()

    convergence_points = []

    for iteration in range(iterations):

        gradient = fun(best_line)[1]

        for j in range(2, len(best_line) - 2):
            best_line[j] = (best_line[j] - (learning_rate * gradient[j]))

        convergence_points.append((iteration, fun(best_line)[0]))

    return best_line, convergence_points