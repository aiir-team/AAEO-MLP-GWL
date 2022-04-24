# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from pathlib import Path
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
# from numpy import arange, pi, sin, cos, arctan
import platform

if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
    import matplotlib

    matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.


def draw_predict_line_with_error(data: list, error: list, filename: str, pathsave: str, exts: list):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # Import Data
    df = DataFrame({'y_true': data[0], 'y_pred': data[1]})
    list_data = [df.loc[:, "y_true"], df.loc[:, "y_pred"]]
    list_data[0].rename("Observed", inplace=True)
    list_data[1].rename("Predicted", inplace=True)

    # Draw Plot
    plt.rcParams['figure.figsize'] = 10, 6.5

    # sns.set(color_codes=True)
    my_fig = plt.figure(constrained_layout=True)
    gs = my_fig.add_gridspec(nrows=2, ncols=5)

    re_data = 100 * (data[0] - data[1]) / data[0]
    ax3 = my_fig.add_subplot(gs[0, :])
    sns.lineplot(data=re_data, ax=ax3)
    ax3.set(ylabel='RE(%)')

    ax1 = my_fig.add_subplot(gs[1, :3])
    sns.lineplot(data=list_data, ax=ax1)
    ax1.set(xlabel='Testing Examples', ylabel=r'Value', title='Prediction Performance: RMSE=' + str(error[1]))
    # ax1.set(xlabel='Months', ylabel=r'Streamflow ($m^3/sec$)')

    ax2 = my_fig.add_subplot(gs[1:, 3:])
    sns.regplot(x="y_true", y="y_pred", data=df, ax=ax2)
    ax2.set(xlabel=r'Observed', ylabel=r'Predicted', title='Linear Relationship: R=' + str(error[0]))
    # ax2.set(xlabel=r'Observed ($m^3/s$)', ylabel=r'Predicted ($m^3/s$)')
    ax2.legend(['Fit'])

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    # if platform.system() != "Linux":
    # plt.show()
    plt.close()

    # # plt.tight_layout()
    # plt.savefig(pathsave + filename + ".png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    return None


def draw_multiple_lines(list_lines: list, list_legends: list, list_colors: list, list_markers: list,
                       xy_labels:list, title: str, filename: str, pathsave: str, exts: list):
    plt.gcf().clear()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for idx, line in enumerate(list_lines):
        ax.plot(line[0], line[1], label=list_legends[idx])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    plt.xlabel(xy_labels[0])
    plt.ylabel(xy_labels[1])
    ax.set_title(title)
    ax.grid('on')

    for idx, ext in enumerate(exts):
        fig.savefig(pathsave + filename + ext, bbox_extra_artists=(lgd, ), bbox_inches='tight')
    if platform.system() != "Linux":
        fig.show()
    plt.close()


def draw_line(line: list, title: str, xy_labels: list, filename: str,
              pathsave: str, exts: list, verbose:False):
    plt.plot(line[0], line[1])
    plt.title(title)
    plt.xlabel(xy_labels[0])
    plt.ylabel(xy_labels[1])

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


# x = arange(-2 * pi, 2 * pi, 0.1)
# y1 = sin(x)
# y2 = cos(x)
# y3 = arctan(x)
# y4 = 2*sin(x)**2 + 1
# y5 = cos(x**2)-1
# y6 = sin(x) - cos(x)
# y7 = arctan(x) - sin(x)
#
# list_lines = [[x, y1], [x, y2], [x, y3], [x, y4], [x, y5], [x, y6], [x, y7]]
# list_legends = ["Sin", "Cos", "Inverse Tan", "Double Sin Square", "Cos X Square", "Sin x - Cos X", "Arctag x - Sin x"]
# title = "Trigonometry"
# xy_labels = ["x", "y"]
# draw_multiple_lines(list_lines, list_legends, None, None, xy_labels, title, "hello-world", ".", [".pdf", ".png"])



