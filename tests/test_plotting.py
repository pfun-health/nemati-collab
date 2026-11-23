import numpy as np

from pycardio.plotting import plot_cvc


def test_plot_cvc_includes_full_cdf_endpoints_for_phase_locked_beats():
    import matplotlib.pyplot as plt

    ax = plot_cvc(type("FeaturesStub", (), {"cvc": np.array([0.3, 0.3])})())
    line = next(line for line in ax.lines if line.get_label() == "data")
    np.testing.assert_array_equal(line.get_xdata(), [0, 0.3, 0.3, 1])
    np.testing.assert_array_equal(line.get_ydata(), [0, 0.5, 1, 1])
    plt.close(ax.figure)
