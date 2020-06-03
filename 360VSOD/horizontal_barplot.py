"""
Horizontal bar plots
====================

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(style="whitegrid")

def sns_official():
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    # Load the example car crash dataset
    crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

    # Plot the total crashes
    sns.set_color_codes('pastel')
    sns.barplot(x='total', y='abbrev', data=crashes, label='Total', color='b')

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="alcohol", y="abbrev", data=crashes, label="Alcohol-involved", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="", xlabel="Automobile collisions per billion miles")
    sns.despine(left=True, bottom=True)

    plt.show()

def self_plot():
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    speed = [0.1, 17.5, 40, 48, 52, 69, 88]
    lifespan = [2, 8, 70, 1.5, 25, 12, 28]
    index = ['snail', 'pig', 'elephant',
             'rabbit', 'giraffe', 'coyote', 'horse']
    df = pd.DataFrame({'speed': speed, 'lifespan': lifespan, 'index': index}, index=index)

    # Plot the total crashes
    sns.set_color_codes('pastel')
    sns.barplot(x='speed', y='index', data=df, label='speed', color='b')

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x='lifespan', y='index', data=df, label='lifespan', color='b')

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 100), ylabel="Hello again !", xlabel="Hello Seaborn !")
    sns.despine(left=True, bottom=True)

    plt.show()


if __name__ == '__main__':
    #sns_official()
    self_plot()