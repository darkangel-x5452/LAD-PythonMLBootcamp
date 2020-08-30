import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import __version__
import cufflinks as cf
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as plty
import plotly.express as px
import plotly.io as pio



def samples():
    # PLOTLY IS ANNOYING TO IMPLEMENT OFFLINE
    # EXAMPLE PLOTLY EXPRESS
    # long_df = px.data.medals_long()
    # fig = px.bar(data_frame=long_df, x="nation", y="count", color="medal", title="Long-Form Input")
    # fig.show()

    # EXAMPLE PLOTLY GO
    # fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    # fig.write_html('first_figure.html', auto_open=True)

    df = pd.DataFrame(np.random.randn(100,4), columns='A B C D'.split())
    df2 = pd.DataFrame({'Category': ['A', 'B','C'], 'Values': [32,43, 50]})
    fig = px.histogram(data_frame=df['A'], nbins=25)
    fig.show()

    print('bye')


if __name__ == '__main__':
    # plty.init_notebook_mode(connected=True)
    # plty.enable_mpl_offline(True)
    samples()


# 2.Possible Colorscale Error 2: In the "Real Data US Map Choropleth", when you are creating the data dictionary, make sure the colorscale line is = 'ylorbr', not 'YIOrbr'... so like this:
# colorscale='ylorbr'
# 3.Possible projection Error 3: In the "World Map Choropleth", when you are creating the layout, ensure that your projection line is = {'type':'mercator'} not Mercator with a capital...so like this:
# projection={'type':'mercator'}