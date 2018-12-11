# -*- coding: utf-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Perfectly random lozenge tiling of a hexagon using
Propp-Wilson's "coupling from the past" algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script samples a random lozenge tiling of a
(a x b x c) hexagon from the uniform distribution.
In the code a lozeng tiling is represented by a path
system consists of (c + 2) non-intersecting paths in the
ac-plane, where the i-th path (i=0, 1, ..., c+2) starts
at (0, i), moves up (in direction b) or moves down
(in direction a) in each step, and ends at (2a, c+i).

The updating rule of the Markov chain is: choose a random
path and a random position on this path, and try to push
up/down the path at this position. If the resulting path
system is not non-intersecting, i.e. does not represent a
tiling, then leave the path untouched.

        c/y-axis
            |    /\
            |   /  \
            | b/    \ a
            | /      \
            |/        \
            |          |
            |          |
          c |          | c
            |  x-axis  |
            |------    |
           O \        /
              \      /
             a \    / b
                \  /
                 \/
                  \
                   \
                   a-axis

To convert coordinates (z_a, z_c) in ac-plane to coordinates
(z_x, z_y) in the usual rectangular xy-plane:

    z_x = z_a * sqrt(3) / 2
    z_y = z_c - z_a / 2

REFERENCES:

    [1]. Blog post at
         "https://possiblywrong.wordpress.com/2018/02/23/coupling-from-the-past/"

    [2]. Wilson's paper
         "Mixing times of lozenge tiling and card shuffling Markov chains".

    [3]. Häggström's book
         "Markov chains and algorithmic applications"

    [4]. Book by David Asher Levin, Elizabeth Lee Wilmer, and Yuval Peres
         "Markov chains and mixing times".

"""
import random
import cairocffi as cairo
from tqdm import tqdm


def run_cftp(mc):
    """
    Sample a random state in a finite, irreducible Markov chain from its
    stationary distribution using monotone CFTP.
    `mc` is a Markov chain object that implements the following methods:
        1. `new_random_update`: return a new random updating operation.
        2. `update`: update a state by an updating operation.
        3. `min_max_state`: return the minimum and maximum states.
    """
    bar = tqdm(desc="Running cftp", unit=" steps")

    updates = [(random.getstate(), 1)]
    while True:
        # run two versions of the chain from the two min, max states
        # in each round.
        s0, s1 = mc.min_max_states()
        rng_next = None
        for rng, steps in updates:
            random.setstate(rng)
            for _ in range(steps):
                u = mc.new_random_update()
                mc.update(s0, u)
                mc.update(s1, u)
                bar.update(1)
            # save the latest random seed for future use.
            if rng_next is None:
                rng_next = random.getstate()
        # check if these two chains are coupled at time 0.
        if s0 == s1:
            break
        # if not coupled the look further back into the past.
        else:
            updates.insert(0, (rng_next, 2**len(updates)))

    random.setstate(rng_next)
    bar.close()
    return s0


class LozengeTiling(object):
    """
    This class builds the "monotone" Markov chain structure on the set
    of lozenge tilings of an (a x b x c) hexagon. A tiling is represented
    by c+2 pairwise non-intersecting paths, where the 0-th and (c+1)-th
    paths are fixed and are used for auxiliary purpose.
    """
    def __init__(self, size):
        """
        :size: a tuple of three integers, these are the side lengths of the hexagon.
        """
        self.size = size

    def min_max_states(self):
        """Return the minimum and maximum tilings. From a bird's view, the minimum
           tiling is the one that correspondes to a room filled full of boxes and
           the maximum tiling is the one that correspondes to an empty room.
        """
        a, b, c = self.size
        # the minimum state
        s0 = [[max(j - a, 0) if k == 0 else k + min(j, b)
               for j in range(a + b + 1)] for k in range(c + 2)]
        # the maximum state
        s1 = [[c + 1 + min(j, b) if k == c + 1 else k + max(j - a, 0)
               for j in range(a + b + 1)] for k in range(c + 2)]

        return s0, s1

    def new_random_update(self):
        """Return a new updating operation.
        """
        a, b, c = self.size
        return (random.randint(1, c),  # a random path
                random.randint(1, a + b - 1),  # a random position in this path
                random.randint(0, 1))  # a random direction (push up or push down)

    def update(self, s, u):
        """Update a state `s` by an operation `u`.
        """
        k, j, dy = u
        # try to push up
        if dy == 1:
            if (s[k][j - 1] == s[k][j] < s[k][j + 1] < s[k + 1][j]):
                s[k][j] += 1
        # try to push down
        else:
            if (s[k - 1][j] < s[k][j - 1] < s[k][j] == s[k][j + 1]):
                s[k][j] -= 1

    def get_tiles(self, s):
        """Return the vertices of all lozenges in the tiling defined by `s`.
        """
        a, b, c = self.size
        # three types of lozenges
        verts = {"L": [], "R": [], "T": []}

        for k in range(c + 1):
            for j in range(1, a + b + 1):
                if k > 0:
                    if s[k][j] == s[k][j - 1]:
                        verts["L"].append([(j + dx, s[k][j] + dy) for dx, dy in
                                           [(0, 0), (-1, 0), (-1, -1), (0, -1)]])
                    else:
                        verts["R"].append([(j + dx, s[k][j] + dy) for dx, dy in
                                           [(0, 0), (-1, -1), (-1, -2), (0, -1)]])
                for l in range(s[k][j] + 1, s[k + 1][j]):
                    verts["T"].append([(j + dx, l + dy) for dx, dy in
                                       [(0, 0), (-1, -1), (0, -1), (1, 0)]])
        return verts


def square_to_hex(verts):
    """Transform vertices in ac-plane to the usual xy-plane.
       :verts: a list of 2d points in ac-plane.
    """
    return [(HALFSQRT3 * x, y - 0.5 * x) for x, y in verts]


IMAGE_SIZE = 600
HEXAGON_SIZE = (20, 20, 20)

TOP_COLOR = (0.89, 0.1, 0.11)
LEFT_COLOR = (1, 0.5, 0)
RIGHT_COLOR = (0.21, 0.5, 0.7)
EDGE_COLOR = (0, 0, 0)

LINE_WIDTH = 0.12

HALFSQRT3 = 0.5 * 3**0.5

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMAGE_SIZE, IMAGE_SIZE)
ctx = cairo.Context(surface)

# put the center of the hexagon at the origin
a, b, c = HEXAGON_SIZE
ctx.translate(IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0)
extent = max(c, a * HALFSQRT3, b * HALFSQRT3) + 1
ctx.scale(IMAGE_SIZE / (extent * 2.0), -IMAGE_SIZE / (extent * 2.0))
ctx.translate(-b * HALFSQRT3, -c / 2.0)

T = LozengeTiling(HEXAGON_SIZE)
random_state = run_cftp(T)
for key, val in T.get_tiles(random_state).items():
    for verts in val:
        A, B, C, D = square_to_hex(verts)
        ctx.move_to(A[0], A[1])
        ctx.line_to(B[0], B[1])
        ctx.line_to(C[0], C[1])
        ctx.line_to(D[0], D[1])
        ctx.close_path()
        if key == "T":
            ctx.set_source_rgb(*TOP_COLOR)
        elif key == "L":
            ctx.set_source_rgb(*LEFT_COLOR)
        else:
            ctx.set_source_rgb(*RIGHT_COLOR)
        ctx.fill_preserve()
        ctx.set_line_width(LINE_WIDTH)
        ctx.set_source_rgb(*EDGE_COLOR)
        ctx.stroke()

surface.write_to_png("random_lozenge_tiling.png")
