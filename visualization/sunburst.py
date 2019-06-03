# -*- coding: utf-8 -*-

from american_gut_project_pipeline.persist import load_dataframe, download_file
import plotly
import plotly.graph_objs as go


def plot_sunburst(df, sample_id):
    # example sample ID
    df_sample = df[['phylum', 'class', 'weight']][df['sample_id'] == int(sample_id)]

    # double group by
    df_sample = df_sample.groupby(['phylum', 'class']).sum()
    df_sample.reset_index(inplace=True)

    # define hierarchy for sunburst plot
    tax_labels = list(df_sample['class'])
    tax_parents = list(df_sample['phylum'])

    # values
    values = df_sample['weight'].tolist()

    # add kingdom to pylum relationship
    for phyla in list(df_sample['phylum'].unique()):
        tax_labels.append(phyla)
        tax_parents.append('Bacteria')
        values.append(1)

    # add kingdom as center of sunburst
    tax_labels.append('Bacteria')
    tax_parents.append("")

    # plot sunbust
    trace = go.Sunburst(
        labels=tax_labels,
        parents=tax_parents,
        values=values,
        outsidetextfont={"size": 20, "color": "#377eb8"},
        marker={"line": {"width": 2}})

    layout = go.Layout(margin=go.layout.Margin(t=0, l=0, r=0, b=0))

    plotly.offline.plot(go.Figure([trace], layout), filename='basic_sunburst_chart.html')


if __name__ == "__main__":
    # download processed data for visualization
    download_file('sampleid_to_tax.csv', 'default')
    df = load_dataframe('sampleid_to_tax.csv')
    df = df.dropna(how='any', axis=0)

    # cast sample id as int
    df['sample_id'] = df['sample_id'].apply(lambda x: int(x))

    # plot sunburst for sample_id = 47146
    plot_sunburst(df, 47146)
    