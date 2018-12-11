# -*- coding: utf-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Draw the hyperbolic tiling of the Poincare upper plane
by fundamental domains of the modular group PSL_2(Z).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A short introduction to the math:

PSL_2(Z) is an infinite group acts discretely on the upper plane by
fractional linear transformations:

              az + b
PSL_2(Z) = {  ------ , a,b,c,d in Z, ad-bc=1 }
              cz + d

This group has three generators A, B, C, where
A: z --> z+1
B: z --> z-1
C: z --> -1/z

Each element g in this group can be written as a word in ["A", "B", "C"],
for example "AAAAAC", "ACBBB", ...
To draw the hyperbolic tiling, one just starts from any fundamental domain D
(usually there is a classical choice of this domain), map it to g(D) for each
element g in the group (up to a given length), then draw all these g(D)s.
The main problem here is the word representation of g is generally not unique,
so it"s not obvious how to traverse each element only once without omitting any.

Here is the deep math: the modular group is an automatic group, i.e.
there exists a DFA such that the words accepted by the DFA are exactly
the elements of the group under the shortest-lex-order representation,
thus finding all elements in this group amounts to traversing a finite
directed graph, which is a much easier job. (we will use breadth-first search here)

reference: see the essay by Bill Casselman

    "https://www.math.ubc.ca/~cass/research/pdf/Automata.pdf"

"""
import collections
import cmath
import cairocffi as cairo


# three generators of the modular group, None means "infinity"
A = lambda z: None if z is None else z + 1
B = lambda z: None if z is None else z - 1
C = lambda z: None if z == 0j else 0j if z is None else -1 / z

# a fundamental domain for drawing the tiling of the modular group
FUND_TRIANGLE = [cmath.exp(cmath.pi * 1j / 3),
                 cmath.exp(cmath.pi * 2j / 3),
                 None]

# a fundamental domain for drawing the Cayley graph of the modular group
z = 1.32j
FUND_HEXAGON = [z,
                A(z),
                A(C(z)),
                A(C(A(z))),
                A(C(A(C(z)))),
                A(C(A(C(A(z)))))]

# The automaton that generates all words in the modular group,
# 0 is the starting state, each element g correspondes to a unique
# path starts from 0.　For example the path
# 0 --> 1 --> 3 -- > 4 --> 8
# correspondes to the element "ACAA" because the first step takes 0 to 1 is
# labelled by "A", the second step takes 1 to 3 is labelled by "C",
# the third step takes 3 to 4 is labelled by "A", ...
AUTOMATON = {0: {"A": 1, "B": 2, "C": 3},
             1: {"A": 1, "C": 3},
             2: {"B": 2, "C": 3},
             3: {"A": 4, "B": 5},
             4: {"A": 8},
             5: {"B": 6},
             6: {"B": 2, "C": 7},
             7: {"A": 4},
             8: {"A": 1, "C": 9},
             9: {"B": 5}}


def transform(symbol, domain):
    """Transform a domain by a generator `symbol`. `domain` is
       specified by a list of comlex numbers on its boundary.
    """
    func = {"A": A, "B": B, "C": C}[symbol]
    return [func(z) for z in domain]


def traverse(length, start_domain):
    """Tranverse domains g(D) for all group element g up to a
       given length. Here D = `start_domian`.
    """
    queue = collections.deque([("", 0, start_domain)])
    while queue:
        word, state, domain = queue.popleft()
        yield word, state, domain

        if len(word) < length:
            for symbol, to in AUTOMATON[state].items():
                queue.append((word + symbol, to, transform(symbol, domain)))


def arc_to(ctx, x1, y1):
    """Draw a geodesic line in the hyperbolic upper plane.
    """
    x0, y0 = ctx.get_current_point()
    dx, dy = x1 - x0, y1 - y0
    # if the geodesic line is a straight line
    if abs(dx) < 1e-8:
        ctx.line_to(x1, y1)
    else:
        # center of the geodesic circle
        center = 0.5 * (x0 + x1) + 0.5 * (y0 + y1) * dy / dx
        theta0 = cmath.phase(x0 - center + y0*1j)
        theta1 = cmath.phase(x1 - center + y1*1j)
        r = abs(x0 - center + y0*1j)

        # we must ensure that the arc ends at (x1, y1)
        if x0 < x1:
            ctx.arc_negative(center, 0, r, theta0, theta1)
        else:
            ctx.arc(center, 0, r, theta0, theta1)


def draw_domain(ctx,
                domain,
                face_color=None,
                edge_color=(0, 0, 0),
                line_width=0.1):
    # The points defining the domain may contain the infinity (None).
    # In this program the infinity always appear at the end,
    # we use 10000 as infinity when drawing lines.

    x0, y0 = domain[0].real, domain[0].imag
    if domain[-1] is None:
        x1 = domain[-2].real
        domain = domain[:-1] + [x1 + 10000*1j, x0 + 10000*1j]
    ctx.move_to(x0, y0)
    for z in domain[1:]:
        arc_to(ctx, z.real, z.imag)
    arc_to(ctx, x0, y0)
    ctx.close_path()

    if face_color is not None:
        ctx.set_source_rgb(*face_color)
        ctx.fill_preserve()

    ctx.set_line_width(line_width)
    ctx.set_source_rgb(*edge_color)
    ctx.stroke()


def draw_tiling(width, height, depth, xlim=(-2, 2), ylim=(0, 2)):
    surface = cairo.SVGSurface("modular_group.svg", width, height)
    ctx = cairo.Context(surface)
    xmin, xmax = xlim
    ymin, ymax = ylim
    ctx.scale(width * 1.0 / (xmax - xmin), height * 1.0 / (ymin - ymax))
    ctx.translate(abs(xmin), -ymax)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.set_line_join(2)
    ctx.move_to(xmin, 0)
    ctx.line_to(xmax, 0)
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(0.03)
    ctx.stroke()

    for word, _, triangle in traverse(depth, FUND_TRIANGLE):
        if word:
            if word[0] == 'C':
                fc_color = (1, 0.5, 0.75)
            else:
                fc_color = None
        else:
            fc_color = (0.5, 0.5, 0.5)

        draw_domain(ctx, triangle, face_color=fc_color, line_width=0.04/(len(word)+1))

    surface.finish()


def draw_cayley(width, height, depth, xlim=(-2, 2), ylim=(0, 2)):
    surface = cairo.SVGSurface("cayley_graph.svg", width, height)
    ctx = cairo.Context(surface)
    xmin, xmax = xlim
    ymin, ymax = ylim
    ctx.scale(width * 1.0 / (xmax - xmin), height * 1.0 / (ymin - ymax))
    ctx.translate(abs(xmin), -ymax)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.set_line_join(2)
    ctx.move_to(xmin, 0)
    ctx.line_to(xmax, 0)
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(0.03)
    ctx.stroke()

    for word, _, hexagon in traverse(depth, FUND_HEXAGON):
        fc_color = (1.0, 0.25, 0.25)
        draw_domain(ctx, hexagon, face_color=fc_color, line_width=0.03/(len(word)+1))

    for word, _, triangle in traverse(depth, FUND_TRIANGLE):
        draw_domain(ctx, triangle, face_color=None, line_width=0.03/(len(word)+1))

    surface.finish()


if __name__ == '__main__':
    draw_tiling(width=800, height=400, depth=15)
    draw_cayley(width=800, height=400, depth=15)
