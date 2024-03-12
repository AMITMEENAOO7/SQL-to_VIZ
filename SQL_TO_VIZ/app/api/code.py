import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
def visualization(data: pd.DataFrame):
    colors = ['#4169E1', '#FFB347']
    fig = px.bar(data, x='staff_name', y='total_rentals', color='staff_name', color_discrete_sequence=colors)
    fig.update_layout(title='Total Rentals by Staff', xaxis_title='Staff Name', yaxis_title='Total Rentals')
    return fig