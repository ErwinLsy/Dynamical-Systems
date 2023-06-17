import numpy as np  # noqa: D100
import matplotlib.pyplot as plt  # noqa: D100
from tqdm import tqdm  # noqa: D100


class Dynamics:  # noqa: D100
    """Define a discrete-time dynamical system.

    Parameters:
    f: the evolution function of the dynamical system
    Df: the derivative of the evolution function
    """

    def __init__(self, f, Df):  # noqa: N803, D107
        self.f = f
        self.Df = Df
        self.lyapunov_exponents_dict = dict({})

    def orbit(self, x0: tuple, r: tuple, n=10000) -> list:
        """Return a list of f^i(x0) for 0 <= i <= n-1.

        Parameters:
        x0: the initial value(state) of the orbit in tuple
        r: the parameters for the evolution function f in tuple 
        n: the number of evolutions, with default value 10000

        Notes:
        The length of the return list will be n.
        """
        ox = [x0] + [0 for _ in range(n-1)]
        for i in range(1, n):
            ox[i] = self.f(ox[i-1], r)
        return ox

    def lyapunov_exponents_over_evolution(self, x0: tuple, r: tuple, n=1000):
        """Save the lyapunov expoents info in a dict.

        Parameters:
        x0: the initial value in tuple
        r: the parameters for the evolution function f in tuple
        n: the number of evolutions
        """
        x = x0
        d = len(x)
        le_over_evolution_list = [[0 for _ in range(n)] for __ in range(d)]
        le_list = np.zeros(d)
        Q = np.identity(d)   # noqa: N806

        for i in range(1, n):
            x = self.f(x, r)
            B = self.Df(x, r)@Q.T   # noqa: N806
            Q, R = np.linalg.qr(B)   # noqa: N806
            le_list += np.log(np.absolute(np.diag(R)))
            for j in range(d):
                le_over_evolution_list[j][i] = le_list[j]/i
        ''' plot le_over_evolution
        for j in range(d):
            plt.plot(le_over_evolution_list[j], c='black')
        plt.show()
        '''
        self.lyapunov_exponents_dict[tuple([tuple(x0), tuple(r), n])] =\
            tuple(le_list/n)

    def lyapunov_exponents(self, x0, r, n=1000):
        """Return the tuple of the lyapunov expoents.

        Parameters:
        x0: the initial value in tuple
        r: the parameters for the evolution function f in tuple
        n: the number of evolutions
        """
        if (x0, r, n) not in self.lyapunov_exponents_dict:
            self.lyapunov_exponents_over_evolution(x0, r, n)
        return self.lyapunov_exponents_dict[(x0, r, n)]


class Dynamics1D(Dynamics):
    """Define a 1D discrete-time dynamical system.

    Parameters:
    f: the evolution function of the dynamical system
    Df: the derivative of the evolution function
    """

    def plot_orbit_over_evolution(self, x0, r, n=10000):
        """Plot the orbit over evolution.

        Parameters:
        x0: the initial value
        r: the parameter(s) for f
        n: the number of evolutions
        """
        ox = self.orbit(x0, r, n)
        plt.scatter(
            list(range(10000)), ox, s=.1, c='black', label=rf"$O_{x0}({r})$")
        plt.title(
            rf"The orbit $O_{x0}({r})$",
            loc='left', style='italic', fontsize=18)
        plt.xlabel("n: number of iterations", fontsize=18)
        plt.ylabel(r"$x_n$", fontsize=18)
        plt.show()

    def plot_average_distance_between_orbits(self, x0, x1, r, n=10000):
        """Plot the average distance between two orbits.

        Parameters:
        x0: one initial value
        x1: other initial value
        r: the parameter(s) for f
        n: the number of evolutions
        """
        ox0 = self.orbit(x0, r, n)
        ox1 = self.orbit(x1, r, n)
        sum_dist = abs(x0-x1)
        avg_dist = [abs(x0-x1)] + [0 for _ in range(n-1)]
        for i in range(1, n):
            sum_dist += abs(ox0[i] - ox1[i])
            avg_dist[i] = sum_dist/i
        print(avg_dist)
        plt.figure(figsize=(8, 8))
        plt.title(
            rf"Average distance between $O_{r}({x0})$ and $O_{r}({x1})$",
            loc='left', style='italic', fontsize=18)
        plt.xlabel("n: number of iterations", fontsize=18)
        plt.ylabel(r"$d_n$", fontsize=18)
        plt.legend(loc='upper left', fontsize=5)
        plt.plot(avg_dist, linewidth=.8, c='black')
        plt.show()


class Dynamics2D(Dynamics):
    """Define a 2D discrete-time dynamical system.

    Parameters:
    f: the evolution function of the dynamical system
    Df: the derivative of the evolution function
    """

    def lyapunov_dimension(self, x0, r, n=1000):
        """Return the lyapunov dimension for a 2D dynamical system.

        Parameters:
        x0: the initial value in tuple
        r: the parameters for the evolution function f in tuple
        n: the number of evolutions
        """
        if (x0, r, n) not in self.lyapunov_exponents_dict:
            self.lyapunov_exponents_over_evolution(x0, r, n)
        l1, l2 = self.lyapunov_exponents(x0, r, n)
        if l1 <= 0:
            return 0
        elif l1+l2 <= 0:
            return 1-l1/l2
        else:
            return 2

    def plot_attractor(self, x0, r, n=10000, n0=10000):
        """Plot an attractor of a dynamical system.

        Parameters:
        x0: the initial value in tuple
        r: the parameters for the evolution function f in tuple
        n: the number of evolutions
        n0: the number of pre-evolutions before plot
        """
        for _ in range(n0):
            x0 = self.f(x0, r)
        ox = self.orbit(x0, r, n)
        x, y = zip(*ox)
        plt.scatter(x, y, s=.1, c='black')
        plt.show()

    def plot_basin_of_attraction(self, r, loq, nums, n=1000):
        """Plot the basin of attraction of for attractor in given square.

        Parameters:
        r: the parameters for the evolution function f in tuple
        loq: the length of square with centre at (0,0)
        nums: the number of points in the square
        n0: the number of pre-evolutions before plot
        """
        x = []
        y = []
        for i in tqdm(np.linspace(-loq, loq, nums),
                      bar_format="Loading:{l_bar}{bar}[time left:{remaining}]"
                      ):
            for j in np.linspace(-loq, loq, nums):
                dl = self.lyapunov_dimension((i, j), r, n)
                if 1 < dl < 2:
                    x.append(i)
                    y.append(j)
        plt.scatter(x, y, s=0.5)
        plt.show()

    def plot_mle(self, x0, r_list, idx, n=1000):
        """Plot the maximal lyapunov exponents over one varied parameter of f.

        Parameters:
        x0: the initial value
        r_list: list of r
        idx: the index of the varied parameter in r
        n: the number of evolution in computing the lyapunov exponents
        """
        mle = [0 for _ in range(len(r_list))]
        i = 0
        for r in tqdm(r_list,
                      bar_format="Loading:{l_bar}{bar}[time left:{remaining}]"
                      ):
            mle[i] = self.lyapunov_exponents(x0, r, n)[0]
            i += 1
        plt.plot([r[idx] for r in r_list], mle, linewidth=.6, c='black')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title("Maximal Lyapunov exponents")
        plt.show()



logistics = Dynamics1D(lambda x, r: r*x*(1-x), None)
logistics.plot_orbit_over_evolution(0.3, 4, 10000)
logistics.plot_average_distance_between_orbits(0.3, 0.3000001, 4, 5000)
'''
f = lambda x, r:\
    np.array(
        [np.sin(x[0]*x[1]/r[1])*x[1]+np.cos(r[0]*x[0] - x[1]),
         x[0] + np.sin(x[1])/r[1]]
         )

df = lambda x, r:\
    np.array(
        [[np.cos(x[0]*x[1]/r[1])*x[1]*x[1]/r[1]-r[0]*np.sin(r[0]*x[0]-x[1]),
          np.cos(x[0]*x[1]/r[1])*x[0]*x[1]/r[1] + np.sin(r[0]*x[0]-x[1]) + np.sin(x[0]*x[1]/r[1])],
          [1,
           np.cos(x[1])/r[1]]])

ds = Dynamics2D(f, df)
x0 = (1, 1)
r = (-0.81, 0.798733)

r_list = [(-0.81, b) for b in np.linspace(0.797, 0.8, 100)]
ds.plot_basin_of_attraction(r,1,100,100)
'''
