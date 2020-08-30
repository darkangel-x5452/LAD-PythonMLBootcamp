import matplotlib.pyplot as plt
import numpy as np


def first_samples():
    x = np.linspace(0, 5, 11)
    y = x ** 2
    # BASIC METHOD
    # plt.subplot(1, 2, 1)
    # plt.plot(x,y)
    # plt.subplot(1, 2, 2)
    # plt.plot(x, y)

    # FUNCTIONAL METHOD
    fig = plt.figure()
    axes = fig.add_axes([0.1,0.1, 0.8,0.8])
    axes.plot(x,y)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_title('title')
    axes2 = fig.add_axes([0.1, 0.2, 0.5, 0.3])

    # AXES ARRAY
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # for axe in axes:
    #     axe.plot(x,y)

    fig = plt.figure(figsize=(3,2))
    ax =fig.add_axes([0,0,1,1])
    ax.plot(x,y, label='x2')
    ax.legend(loc=0)
    plt.tight_layout()

    # files
    # fig.savefig('my_picture.png', dpi=200)

    plt.show()


def second_samples():
    x = np.linspace(0, 5, 11)
    y = x ** 2

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(x,y,color='purple', linewidth=0.5, alpha=0.5, linestyle='--', marker='o', markersize=1, markerfacecolor='yellow') # can also use hex colours
    ax.set_xlim([0,1])
    ax.set_ylim([0, 1])


    plt.show()


def exercises():
    x = np.arange(0, 100)
    y = x * 2
    z = x ** 2

    fig = plt.figure(figsize=(3,2))
    # axes = fig.add_axes([0,0,1,1])
    # axes2 = fig.add_axes([0.2, 0.5, .4, .4])
    # axes.set_xlabel('x')
    # axes.set_ylabel('z')
    # # # axes.set_title('title')
    # axes2.set_xlabel('x')
    # axes2.set_ylabel('y')
    # axes2.set_title('zoom')
    # axes2.set_xlim(20,22)
    # axes2.set_xlim(30, 50)
    # axes.plot(x,z)
    # axes2.plot(x,y, color='red')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,2))
    axes[0].plot(x,y)
    axes[1].plot(x, z)
    plt.show()
    print('bye')

if __name__ == '__main__':
    # first_samples()
    # second_samples()
    exercises()