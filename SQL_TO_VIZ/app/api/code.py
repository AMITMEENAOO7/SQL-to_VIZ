import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
def visualization(data: pd.DataFrame):
    colors = px.colors.qualitative.Set3
    fig = px.bar(data, x='actor_id', y='actor_count', color='actor_id', color_discrete_sequence=colors, title='Bar Plot of Actor Count per Actor ID')
    fig.update_layout(plot_bgcolor='white')
    return fig