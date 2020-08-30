import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def first_samples():
    tips = sns.load_dataset('tips')
    tips.head()
    # sns.distplot(tips['total_bill'],kde=False, bins=30)
    # sns.jointplot(x='total_bill',y='tip',data=tips, kind='kde')
    # sns.pairplot(tips, hue='sex')
    sns.rugplot(tips['total_bill'])


    plt.show()
    print('bye')


def second_samples():
    # Create dataset
    dataset = np.random.randn(25)

    # Create another rugplot
    sns.rugplot(dataset)

    # Set up the x-axis for the plot
    x_min = dataset.min() - 2
    x_max = dataset.max() + 2

    # 100 equally spaced points from x_min to x_max
    x_axis = np.linspace(x_min, x_max, 100)

    # Set up the bandwidth, for info on this:
    url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

    bandwidth = ((4 * dataset.std() ** 5) / (3 * len(dataset))) ** .2

    # Create an empty kernel list
    kernel_list = []

    # Plot each basis function
    for data_point in dataset:
        # Create a kernel for each point and append to list
        kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
        kernel_list.append(kernel)

        # Scale for plotting
        kernel = kernel / kernel.max()
        kernel = kernel * .4
        plt.plot(x_axis, kernel, color='grey', alpha=0.5)

    plt.ylim(0, 1)
    plt.show()
    # To get the kde plot we can sum these basis functions.

    # Plot the sum of the basis function
    sum_of_kde = np.sum(kernel_list, axis=0)

    # Plot figure
    fig = plt.plot(x_axis, sum_of_kde, color='indianred')

    # Add the initial rugplot
    sns.rugplot(dataset, c='indianred')

    # Get rid of y-tick marks
    plt.yticks([])

    # Set title
    plt.suptitle("Sum of the Basis Functions")
    plt.show()
    print('bye')


def categorical_plot():
    tips = sns.load_dataset('tips')
    tips.head()
    # sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
    # sns.countplot(x='sex', data=tips)
    # sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
    # sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
    # sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex', split=True)
    # sns.violinplot(x='day', y='total_bill', data=tips )
    # sns.swarmplot(x='day', y='total_bill', data=tips, color='black')
    sns.catplot(x='day', y='total_bill', data=tips, kind='violin')



    plt.show()

    print('bye')


def matrix_plot():
    tips = sns.load_dataset('tips')
    # print(tips.head())
    flights = sns.load_dataset('flights')

    # tc = tips.corr()
    # sns.heatmap(tc, annot=True, cmap='coolwarm')

    fp = flights.pivot_table(index='month', columns='year', values='passengers')
    # sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=1)
    # sns.clustermap(fp, cmap='coolwarm', standard_scale=1)

    plt.show()
    print('bye')


def grids_plot():
    iris = sns.load_dataset('iris')
    print(iris.head())
    # sns.pairplot(iris)
    # g = sns.PairGrid(iris)
    # g.map(plt.scatter)
    # g.map_diag(sns.distplot)
    # g.map_upper(plt.scatter)
    # g.map_lower(sns.kdeplot)

    tips = sns.load_dataset('tips')
    g = sns.FacetGrid(data=tips,col='time',row='smoker')
    # g.map(sns.distplot, 'total_bill')
    g.map(plt.scatter, 'total_bill', 'tip')
    plt.show()
    print('bye')


def regress_plot():
    tips = sns.load_dataset('tips')
    # sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', markers=['o', 'v'], scatter_kws={'s': 100})
    # sns.lmplot(x='total_bill', y='tip', data=tips, col='day', row='time', hue='sex')
    sns.lmplot(x='total_bill', y='tip', data=tips, col='day', hue='sex', aspect=0.6, size=8)
    plt.show()
    print('bye')


def style_color():
    tips = sns.load_dataset('tips')

    # sns.set_style('whitegrid')
    # sns.countplot(x='sex', data=tips)
    # sns.despine()

    # plt.figure(figsize=(12,3)) # or below
    # sns.set_context('poster', font_scale=0.5)
    # sns.countplot(x='sex', data=tips)
    # sns.despine()

    sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='coolwarm')

    plt.show()
    print('bye')


def exercises():
    sns.set_style('whitegrid')
    titanic = sns.load_dataset('titanic')

    # sns.jointplot(x='fare', y='age', data=titanic)
    # sns.distplot(titanic['fare'], bins=60, kde=False)
    # sns.boxplot(x='class', y='age', data=titanic)
    # sns.swarmplot(x='class', y='age', data=titanic)
    # sns.countplot(data=titanic, x='sex')
    # sns.heatmap(titanic.corr(), cmap='coolwarm')
    g = sns.FacetGrid(data=titanic, col='sex')
    g.map(plt.hist, 'age')

    plt.show()
    print('bye')


if __name__ == '__main__':
    # first_samples()
    # second_samples()
    # categorical_plot()
    # matrix_plot()
    # grids_plot()
    # regress_plot()
    # style_color()
    exercises()
