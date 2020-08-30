import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.plotly as py

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
# fig.write_html('first_figure.html', auto_open=True)

def samples_1():
    # EXAMPLE PLOTLY EXPRESS
    # long_df = px.data.medals_long()
    # fig = px.bar(data_frame=long_df, x="nation", y="count", color="medal", title="Long-Form Input")
    # fig.show()

    # EXAMPLE PLOTLY GO
    # fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    # fig.write_html('first_figure.html', auto_open=True)

    data = dict(type='choropleth',
                locations=['AZ', 'CA', 'NY'],
                locationmode='USA-states',
                colorscale='Portland',
                text=['text1', 'text2', 'text3'],
                z=[1.0, 2.0, 3.0],
                colorbar={'title': 'Colorbar Title'})
    layout = dict(geo={'scope': 'usa'})
    # choromap = go.Figure(data=[data], layout=layout)
    # iplot(choromap)
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2011_US_AGRI_Exports')

    data = dict(type='choropleth',
                colorscale='ylgn',
                locations=df['code'],
                z=df['total exports'],
                locationmode='USA-states',
                text=df['text'],
                marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                colorbar={'title': "Millions USD"}
                )

    layout = dict(title='2011 US Agriculture Exports by State',
                  geo=dict(scope='usa',
                           showlakes=True,
                           lakecolor='rgb(85,173,240)')
                  )
    choromap = go.Figure(data=[data], layout=layout)
    iplot(choromap)

    print('bye')


def samples_2():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2014_World_GDP')
    print(df.head())

    data = dict(
        type='choropleth',
        locations=df['CODE'],
        z=df['GDP (BILLIONS)'],
        text=df['COUNTRY'],
        colorbar={'title': 'GDP Billions US'},
    )

    layout = dict(
        title='2014 Global GDP',
        geo=dict(
            showframe=False,
            projection={'type': 'natural earth'}
        )
    )

    choromap = go.Figure(data=[data], layout=layout)
    iplot(choromap)
    print('bye')


def exercises():
    # df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2014_World_Power_Consumption')
    # print(df.head())
    #
    # data = dict(type='choropleth',
    #             colorscale='ylgn',
    #             locations=df['Country'],
    #             z=df['Power Consumption KWH'],
    #             locationmode='country names',
    #             text=df['Text'],
    #             # marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
    #             colorbar={'title': "Power"}
    #             )
    #
    # layout = dict(title='Power',
    #               geo=dict(scope='world',
    #                        showcoastlines=True,
    #                        coastlinecolor='rgb(85,173,240)',
    #                        projection=dict(type='natural earth'))
    #               )
    # choromap = go.Figure(data=[data], layout=layout)
    # iplot(choromap)

    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2012_Election_Data')
    print(df.head())

    data = dict(type='choropleth',
                colorscale='ylgn',
                locations=df['State Abv'],
                z=df['Voting-Age Population (VAP)'],
                locationmode='USA-states',
                text=df['State'],
                # marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                colorbar={'title': "Power"}
                )

    layout = dict(title='Power',
                  geo=dict(scope='usa',
                           # showcoastlines=True,
                           # coastlinecolor='rgb(85,173,240)',
                           projection=dict(type='albers usa')
                           )
                  )
    choromap = go.Figure(data=[data], layout=layout)
    iplot(choromap)


    print('bye')


if __name__ == '__main__':
    # samples_1()
    # samples_2()
    exercises()