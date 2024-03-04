import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
def visualization(data: pd.DataFrame):
    colors = px.colors.qualitative.Set3
    fig = px.pie(data, values='count', names='category', color='rating', color_discrete_map={rating: colors[i] for i, rating in enumerate(data['rating'].unique())})
    return fig