import math

def solve_quadratic(a: float, b: float, c: float) -> tuple[float | complex, float | complex]:
    '''
    Solve a quadratic equation of the form ax**2 + bx + c = 0.
    :param a: Coefficient of x**2.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: A tuple containing the roots (real or complex).
    '''
    discriminant = b**2 - 4*a*c

    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2 * a)
        root2 = (-b - math.sqrt(discriminant)) / (2 * a)

        return root1, root2
    elif discriminant == 0:
        root = -b / (2 * a)

        return root, root
    else:
        real = -b / (2 * a)
        imag = math.sqrt(discriminant) / (2 * a)

        return complex(real, imag), complex(real, imag)

def discriminant_analysis(a: float, b: float, c: float) -> str:
    '''
    Analyze the discriminant of the quadratic equation.
    :param a: Coefficient of x**2.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: A string describing the nature of the roots.
    '''
    discriminant = b ** 2 - 4 * a * c

    if discriminant > 0: return "Two different real roots."
    elif discriminant == 0: return "Two equivalent real roots."
    else: return "Two complex roots."

def vertex_form(a: float, b: float, c: float) -> tuple[float, float, str]:
    '''
    Calculate the vertex form of the quadratic equation.
    :param a: Coefficient of x**2.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: The vertex coordinates (h, k) and the expression in vertex form.
    '''
    discriminant = b ** 2 - 4 * a * c
    h = -b / (2 * a)
    k = -discriminant / (4 * a)

    return h, k, f"f(x) = {a}(x - {h})\u00b2 + {k}"