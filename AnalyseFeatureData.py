import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import requests
import branca.colormap as cm
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide", page_title="County Analytics Dashboard")

# Load data
df = pd.read_excel("C:/Users/Admin/Downloads/feature_Data.xlsx")
for col in ['Expenditures', 'Revenues', 'Indebtedness']:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
df['GeoFIPS'] = df['GeoFIPS'].astype(str).str.zfill(5)

df_sankey = pd.read_csv("c:/Users/Admin/Downloads/layer_score.csv")
df_sankey.columns = [col.strip() for col in df_sankey.columns]
df_sankey = df_sankey.rename(columns={"normalized_name": "City"})

dfr = df_sankey.dropna(subset=["City", "County", "Revenue Cluster", "Data and Analytics", "Action Planning", "Total Score"])

# Load Illinois counties GeoJSON
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
full_geojson = requests.get(url).json()
geojson = {
    "type": "FeatureCollection",
    "features": [f for f in full_geojson['features'] if f['id'].startswith("17")]
}

# Function: Folium single-county map
def create_single_county_map(data, column, legend_name, selected_fips):
    m = folium.Map(location=[40.0, -89.0], zoom_start=6, tiles='cartodbpositron')
    value = data[data['GeoFIPS'] == selected_fips][column].values[0]
    max_val = data[column].max()
    colormap = cm.linear.YlOrRd_09.scale(0, max_val)
    fill_color = colormap(value)

    for feature in geojson['features']:
        if feature['id'] == str(selected_fips):
            folium.GeoJson(
                feature,
                style_function=lambda x: {
                    'fillColor': fill_color,
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.8
                },
                tooltip=f"{legend_name}: {value}"
            ).add_to(m)

    colormap.caption = legend_name
    colormap.add_to(m)
    return m

# Function: Plotly Illinois choropleth
def make_il_county_choropleth(input_df, input_fips_col, input_value_col, input_color_theme):
    geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geojson_data = requests.get(geojson_url).json()
    input_df = input_df[input_df[input_fips_col].str.startswith('17')]

    choropleth = px.choropleth(
        input_df,
        geojson=geojson_data,
        locations=input_fips_col,
        color=input_value_col,
        color_continuous_scale=input_color_theme,
        scope="usa",
        labels={input_value_col: input_value_col.capitalize()},
        hover_name=input_fips_col
    )

    choropleth.update_geos(fitbounds="locations", visible=False)
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return choropleth

# Tabs
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#     "\U0001F4CD Geographical Visualization", 
#     "\U0001F9E9 County Sankey", 
#     "\U0001F4B0 Revenue Cluster Sankey", 
#     "\U0001F4CA Box Plots",
#     "\U0001F4DD City Report",
#     "\U0001F4C8 Report Type Boxplots"
# ])
tab1, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "\U0001F4CD Geographical Visualization", 
    "\U0001F4CA Box Plots",
    "\U0001F4DD City Report",
    "\U0001F4C8 Report Type Boxplots",
    "\U0001F4C8 Climate vs Non-Climate Boxplots",
    "\U0001F4C8 Demographic",
    "\U0001F4C8 Feature Score Analysis",
    "\U0001F4C8 Skokie & Niles Highlight"
])

# === Tab 1 ===
with tab1:
    st.header("\U0001F4CD County-Level Feature Analysis")
    st.markdown("Explore **median** and **range** values of selected features by county.")

    col1, col2 = st.columns([2, 3])
    with col1:
        selected_feature = st.selectbox("Select a Feature", ['Revenues', 'Expenditures', 'Indebtedness', 'median_income'])
    with col2:
        selected_county = st.selectbox("Select a County", sorted(df['County'].unique()))

    summary = df.groupby("County")[selected_feature].agg(['median', lambda x: x.max() - x.min()]).reset_index()
    summary.columns = ['County', 'Median', 'Range']
    summary['GeoFIPS'] = summary['County'].map(lambda x: df[df['County'] == x]['GeoFIPS'].iloc[0])
    selected_fips = summary[summary['County'] == selected_county]['GeoFIPS'].values[0]

    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.subheader(f"\U0001F5FA Median {selected_feature}")
        with st.expander("View Median Map", expanded=True):
            median_map = create_single_county_map(summary, 'Median', f"Median {selected_feature}", selected_fips)
            folium_static(median_map)

    with col_map2:
        st.subheader(f"\U0001F5FA Range of {selected_feature}")
        with st.expander("View Range Map", expanded=True):
            range_map = create_single_county_map(summary, 'Range', f"Range of {selected_feature}", selected_fips)
            folium_static(range_map)

    # Add Plotly Choropleth
    with st.expander("View Interactive Choropleth (Plotly)", expanded=False):
        choropleth_type = st.radio("Select Choropleth Type", ["Median", "Range"], horizontal=True)
        plotly_df = summary[['GeoFIPS', 'Median', 'Range']].copy()
        plotly_df['GeoFIPS'] = plotly_df['GeoFIPS'].astype(str).str.zfill(5)
        choropleth_plotly = make_il_county_choropleth(
            plotly_df,
            input_fips_col="GeoFIPS",
            input_value_col=choropleth_type,
            input_color_theme="Viridis"
        )
        st.plotly_chart(choropleth_plotly, use_container_width=True)

# # === Tab 2–4 unchanged ===
# # You already have working code for Sankey and box plots in tabs 2–4.


# with tab2:
#     st.header("\U0001F9E9 County-Level Sankey Diagram")
#     selected_county_name = st.selectbox("Select County", sorted(df_sankey['County'].dropna().unique()))

#     county_df = df_sankey[df_sankey['County'] == selected_county_name].copy()
#     county_df = county_df.dropna(subset=["City", "Governance", "Data and Analytics", "Action Planning", "Total Score"])

#     if county_df['Governance'].dtype == 'object':
#         county_df['Governance'] = county_df['Governance'].astype('category').cat.codes

#     score_cols = ['Governance', 'Data and Analytics', 'Action Planning', 'Total Score']
#     for col in score_cols:
#         max_val = county_df[col].max()
#         county_df[col + " (norm)"] = county_df[col] / max_val if max_val else 0

#     cities = county_df['City'].tolist()
#     n = len(cities)
#     city_idx = list(range(n))
#     gov_idx = list(range(n, 2 * n))
#     da_idx = list(range(2 * n, 3 * n))
#     ap_idx = list(range(3 * n, 4 * n))
#     total_idx = 4 * n

#     labels = cities + [f"{c} - GOV" for c in cities] + [f"{c} - DA" for c in cities] + [f"{c} - AP" for c in cities] + ["Total Score"]
#     source, target, values, link_labels = [], [], [], []

#     for i, city in enumerate(cities):
#         source += [city_idx[i], gov_idx[i], da_idx[i], ap_idx[i]]
#         target += [gov_idx[i], da_idx[i], ap_idx[i], total_idx]
#         values += [county_df.iloc[i][f'{col} (norm)'] for col in score_cols]
#         link_labels += [
#             f"{city} → Governance: {county_df.iloc[i]['Governance (norm)']:.2f}",
#             f"{city} GOV → DA: {county_df.iloc[i]['Data and Analytics (norm)']:.2f}",
#             f"{city} DA → AP: {county_df.iloc[i]['Action Planning (norm)']:.2f}",
#             f"{city} AP → Total: {county_df.iloc[i]['Total Score (norm)']:.2f}"
#         ]

#     fig = go.Figure(data=[go.Sankey(
#         arrangement="snap",
#         node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
#         link=dict(
#             source=source, target=target, value=values,
#             customdata=link_labels,
#             hovertemplate="%{customdata}<extra></extra>",
#             color=[
#                 'rgba(148, 103, 189, 0.6)' if "- GOV" in labels[t] else
#                 'rgba(31, 119, 180, 0.5)' if "- DA" in labels[t] else
#                 'rgba(255, 127, 14, 0.6)' if "- AP" in labels[t] else
#                 'rgba(44, 160, 44, 0.6)' for t in target
#             ]
#         )
#     )])
#     st.plotly_chart(fig, use_container_width=True)

# with tab3:
#     st.header("\U0001F4B0 Revenue Cluster Sankey Diagram")
#     selected_cluster = st.selectbox("Select Revenue Cluster", sorted(dfr['Revenue Cluster'].dropna().unique()))

#     cluster_df = dfr[dfr['Revenue Cluster'] == selected_cluster].copy()
#     cluster_df = cluster_df.dropna(subset=["City", "Governance", "Data and Analytics", "Action Planning", "Total Score"])

#     if cluster_df['Governance'].dtype == 'object':
#         cluster_df['Governance'] = cluster_df['Governance'].astype('category').cat.codes

#     for col in score_cols:
#         max_val = cluster_df[col].max()
#         cluster_df[col + " (norm)"] = cluster_df[col] / max_val if max_val else 0

#     cities = cluster_df['City'].tolist()
#     n = len(cities)
#     city_idx = list(range(n))
#     gov_idx = list(range(n, 2 * n))
#     da_idx = list(range(2 * n, 3 * n))
#     ap_idx = list(range(3 * n, 4 * n))
#     total_idx = 4 * n

#     labels = cities + [f"{c} - GOV" for c in cities] + [f"{c} - DA" for c in cities] + [f"{c} - AP" for c in cities] + ["Total Score"]
#     source, target, values, link_labels = [], [], [], []

#     for i, city in enumerate(cities):
#         source += [city_idx[i], gov_idx[i], da_idx[i], ap_idx[i]]
#         target += [gov_idx[i], da_idx[i], ap_idx[i], total_idx]
#         values += [cluster_df.iloc[i][f'{col} (norm)'] for col in score_cols]
#         link_labels += [
#             f"{city} → Governance: {cluster_df.iloc[i]['Governance (norm)']:.2f}",
#             f"{city} GOV → DA: {cluster_df.iloc[i]['Data and Analytics (norm)']:.2f}",
#             f"{city} DA → AP: {cluster_df.iloc[i]['Action Planning (norm)']:.2f}",
#             f"{city} AP → Total: {cluster_df.iloc[i]['Total Score (norm)']:.2f}"
#         ]

#     fig = go.Figure(data=[go.Sankey(
#         arrangement="snap",
#         node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
#         link=dict(
#             source=source,
#             target=target,
#             value=values,
#             customdata=link_labels,
#             hovertemplate="%{customdata}<extra></extra>",
#             color=[
#                 'rgba(148, 103, 189, 0.6)' if "- GOV" in labels[t] else
#                 'rgba(31, 119, 180, 0.5)' if "- DA" in labels[t] else
#                 'rgba(255, 127, 14, 0.6)' if "- AP" in labels[t] else
#                 'rgba(44, 160, 44, 0.6)' for t in target
#             ]
#         )
#     )])
#     st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("\U0001F4CA Box & Whisker Plots by Feature Clusters")

    df_plot = pd.read_csv("C:Users/Admin/Downloads/layer_score.csv")
    for col in ['Revenues', 'Expenditures', 'Indebtedness', 'median_income']:
        if df_plot[col].dtype == object:
            df_plot[col] = df_plot[col].replace('[\$,]', '', regex=True).astype(float)

    selected_feature = st.selectbox("Select Feature for Clustering", ["Revenues", "Expenditures", "Indebtedness", "median_income"])

    def cluster_label(value, feature):
        if feature == "median_income":
            return "0–50k" if value < 50000 else "50k–100k" if value <= 100000 else "100k+"
        return "0–50M" if value < 50_000_000 else "50M–250M" if value <= 250_000_000 else "250M+"

    cluster_col = f"{selected_feature} Cluster"
    df_plot[cluster_col] = df_plot[selected_feature].apply(lambda x: cluster_label(x, selected_feature))

    cluster_order_map = {
        "Revenues": ["0–50M", "50M–250M", "250M+"],
        "Expenditures": ["0–50M", "50M–250M", "250M+"],
        "Indebtedness": ["0–50M", "50M–250M", "250M+"],
        "median_income": ["0–50k", "50k–100k", "100k+"]
    }
    order = cluster_order_map[selected_feature]
    df_plot = df_plot.dropna(subset=[cluster_col, "Governance", "Data and Analytics", "Action Planning", "Total Score"])

    cols = ["Governance", "Data and Analytics", "Action Planning", "Total Score"]
    titles = ["Governance", "Data & Analytics", "Action Planning", "Total Score"]
    layout_cols = st.columns(2)
    for i in range(0, len(cols), 2):
        with layout_cols[0]:
            fig = px.box(df_plot, x=cluster_col, y=cols[i], points="all", category_orders={cluster_col: order}, title=f"{titles[i]} vs {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
        with layout_cols[1]:
            fig = px.box(df_plot, x=cluster_col, y=cols[i+1], points="all", category_orders={cluster_col: order}, title=f"{titles[i+1]} vs {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
with tab5:
    st.header("📝 City Score Comparison by Report Category")

    # Load data
    df_city = pd.read_csv("C:/Users/Admin/Downloads/layer_score.csv")
    df_city.columns = df_city.columns.str.strip()

    # Clean Report Type 1 column (remove trailing spaces)
    df_city["Report Type 1"] = df_city["Report Type 1"].str.strip()

    # Clean numeric columns
    score_categories = ["Governance", "Data and Analytics", "Action Planning", "Total Score"]
    for col in score_categories:
        df_city[col] = pd.to_numeric(df_city[col], errors="coerce")

    # Define report type mapping
    climate_types = ["Environmental Plan", "Sustainability Plan", "Climate Plan"]
    non_climate_types = ["Comprehensive Plan", "Strategic Plan"]

    # Dropdown 1: Select Climate/Non-Climate
    report_type_1_options = ["Climate", "Non Climate"]
    selected_type_1 = st.selectbox("Select Report Category", report_type_1_options)

    # Dropdown 2: Filter Report Type based on category
    if selected_type_1 == "Climate":
        filtered_df = df_city[df_city["Report Type 1"].str.strip() == "Climate"]
        allowed_types = [rt for rt in climate_types if rt in filtered_df["Report Type"].unique()]
    else:
        filtered_df = df_city[df_city["Report Type 1"].str.strip() == "Non Climate"]
        allowed_types = [rt for rt in non_climate_types if rt in filtered_df["Report Type"].unique()]

    selected_report_type = st.selectbox("Select Report Type", sorted(allowed_types))

    # Dropdown 3: Select City
    report_df = filtered_df[filtered_df["Report Type"] == selected_report_type].copy()
    city_options = sorted(report_df["normalized_name"].dropna().unique())
    selected_city = st.selectbox("Select City", city_options)

    # Get selected city's scores
    city_scores = report_df[report_df["normalized_name"] == selected_city].iloc[0]

    st.markdown(f"### Score Comparison for **{selected_city}** within **{selected_report_type}**")

    # Violin plots
    cols = st.columns(len(score_categories))
    for i, category in enumerate(score_categories):
        with cols[i]:
            fig = px.violin(
                report_df,
                y=category,
                box=True,
                points="all",
                title=category,
                labels={category: "Score"},
            )
            fig.add_shape(
                type="line",
                x0=-0.5, x1=0.5,
                y0=city_scores[category], y1=city_scores[category],
                line=dict(color="red", dash="dash"),
            )
            fig.add_annotation(
                x=0,
                y=city_scores[category],
                text=f"{selected_city}: {city_scores[category]}",
                showarrow=True,
                arrowhead=1,
                yshift=10
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Bar Plot Comparison")

    # Bar plots
    for category in score_categories:
        report_df["__color__"] = report_df["normalized_name"].apply(
            lambda name: "Selected City" if name == selected_city else "Other Cities"
        )

        fig = px.bar(
            report_df.sort_values(by=category, ascending=True),
            x=category,
            y="normalized_name",
            orientation="h",
            title=f"{category} Scores Across Cities – {selected_report_type}",
            labels={category: "Score", "normalized_name": "City"},
            color="__color__",
            color_discrete_map={"Selected City": "crimson", "Other Cities": "steelblue"},
        )

        fig.update_layout(
            height=400,
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.header("📦 Report Type vs Feature Score Distribution")

    # Load and clean data
    df_box = pd.read_csv("C:/Users/Admin/Downloads/layer_score.csv")
    df_box.columns = df_box.columns.str.strip()
    df_box["Report Type"] = df_box["Report Type"].str.strip()

    report_type_order = [
        "Environmental Plan", "Sustainability Plan", "Climate Plan",
        "Comprehensive Plan", "Strategic Plan"
    ]
    df_box = df_box[df_box["Report Type"].isin(report_type_order)]

    score_cols = ["Governance", "Data and Analytics", "Action Planning", "Total Score"]
    for col in score_cols:
        df_box[col] = pd.to_numeric(df_box[col], errors="coerce")
    df_box = df_box.dropna(subset=score_cols + ["Report Type", "normalized_name"])

    df_melted = df_box.melt(
        id_vars=["Report Type", "normalized_name"],
        value_vars=score_cols[:-1],
        var_name="Score Type",
        value_name="Score"
    )

    # === Total Score Box Plot ===
    fig2 = px.box(
        df_box,
        x="Report Type",
        y="Total Score",
        category_orders={"Report Type": report_type_order},
        title="Total Score Distribution by Report Type",
        points="all",
        hover_data=["normalized_name"]
    )
    fig2.update_layout(
        xaxis=dict(
            title=dict(text="Report Type", font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(text="Total Score", font=dict(color="black")),
            tickfont=dict(color="black"),
            range=[0, 60]
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

    # === Feature Score Box Plot ===
    fig = px.box(
        df_melted,
        x="Report Type",
        y="Score",
        color="Score Type",
        category_orders={"Report Type": report_type_order},
        title="Feature Score Distribution by Report Type",
        points="all",
        hover_data=["normalized_name"]
    )
    fig.update_layout(
        boxmode="group",
        xaxis=dict(
            title=dict(text="Report Type", font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(text="Score", font=dict(color="black")),
            tickfont=dict(color="black"),
            range=[0, 20]
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Combined Types ===
    st.header("📦 Combined Report Types vs Feature Score Distribution")

    df_box["Report Type Modified"] = df_box["Report Type"].replace({
        "Environmental Plan": "Climate-Related Plans",
        "Sustainability Plan": "Climate-Related Plans",
        "Climate Plan": "Climate-Related Plans"
    })
    combined_types = ["Climate-Related Plans", "Comprehensive Plan", "Strategic Plan"]
    df_box = df_box[df_box["Report Type Modified"].isin(combined_types)]

    df_melted_combined = df_box.melt(
        id_vars=["Report Type Modified", "normalized_name"],
        value_vars=score_cols[:-1],
        var_name="Score Type",
        value_name="Score"
    )

    fig_combined = px.box(
        df_melted_combined,
        x="Report Type Modified",
        y="Score",
        color="Score Type",
        category_orders={"Report Type Modified": combined_types},
        title="Feature Score Distribution: Combined Report Types",
        points="all",
        hover_data=["normalized_name"]
    )
    fig_combined.update_layout(
        boxmode="group",
        xaxis=dict(
            title=dict(text="Report Type", font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(text="Score", font=dict(color="black")),
            tickfont=dict(color="black"),
            range=[0, 20]
        )
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    fig_total_combined = px.box(
        df_box,
        x="Report Type Modified",
        y="Total Score",
        category_orders={"Report Type Modified": combined_types},
        title="Total Score Distribution: Combined Report Types",
        points="all",
        hover_data=["normalized_name"]
    )
    fig_total_combined.update_layout(
        xaxis=dict(
            title=dict(text="Report Type", font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(text="Total Score", font=dict(color="black")),
            tickfont=dict(color="black"),
            range=[0, 60]
        )
    )
    st.plotly_chart(fig_total_combined, use_container_width=True)

# === Top Quartile ===
    st.header("🏙️ Top Quartile Cities – Climate-Related Plans Only")

    df_climate = df_box[df_box["Report Type Modified"] == "Climate-Related Plans"].copy()
    df_climate = df_climate.dropna(subset=score_cols + ["normalized_name"])

    top_quartile_threshold = df_climate["Total Score"].quantile(0.75)
    df_top = df_climate[df_climate["Total Score"] >= top_quartile_threshold]

    df_top_melted = df_top.melt(
        id_vars=["normalized_name"],
        value_vars=score_cols[:-1],
        var_name="Feature",
        value_name="Score"
    )

    global_medians = df_climate[score_cols[:-1]].median().to_dict()

    fig_features = px.box(
        df_top_melted,
        x="Feature",
        y="Score",
        points="all",
        hover_data=["normalized_name"],
        title="Feature Scores for Top Quartile Climate-Related Cities"
    )
    for feature, median_val in global_medians.items():
        fig_features.add_shape(
            type="line",
            x0=feature,
            x1=feature,
            y0=0,
            y1=median_val,
            xref="x",
            yref="y",
            line=dict(color="red", dash="dash")
        )
        fig_features.add_annotation(
            x=feature,
            y=median_val,
            text=f"Median: {median_val:.1f}",
            showarrow=False,
            yshift=10,
            font=dict(color="red", size=10)
        )
    fig_features.update_layout(
        xaxis=dict(
            title=dict(text="Feature", font=dict(color="black")),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title=dict(text="Score", font=dict(color="black")),
            tickfont=dict(color="black")
        )
    )
    st.plotly_chart(fig_features, use_container_width=True)

    fig_top_total = px.box(
        df_top,
        y="Total Score",
        points="all",
        hover_data=["normalized_name"],
        title="Total Score for Top Quartile Climate-Related Cities"
    )
    fig_top_total.update_layout(
        xaxis=dict(tickfont=dict(color="black")),
        yaxis=dict(
            title=dict(text="Total Score", font=dict(color="black")),
            tickfont=dict(color="black")
        )
    )
    st.plotly_chart(fig_top_total, use_container_width=True)

with tab7:
    st.header("🏙️ City Score Comparison - Climate vs Non-Climate")

    # Load and clean data
    df_city = pd.read_csv("C:/Users/Admin/Downloads/layer_score.csv")
    df_city.columns = df_city.columns.str.strip()
    df_city["Report Type 1"] = df_city["Report Type 1"].str.strip()

    score_categories = ["Governance", "Data and Analytics", "Action Planning", "Total Score"]
    for col in score_categories:
        df_city[col] = pd.to_numeric(df_city[col], errors="coerce")

    # Dropdown 1: Climate / Non Climate
    category_choice = st.selectbox("Select Report Type 1 Category", ["Climate", "Non Climate"])

    # Filter cities accordingly
    filtered_df = df_city[df_city["Report Type 1"] == category_choice]
    city_list = sorted(filtered_df["normalized_name"].dropna().unique())

    # Dropdown 2: Select City
    selected_city = st.selectbox("Select City", city_list)

    # Filter city-specific data
    report_df = filtered_df[filtered_df["normalized_name"] == selected_city]
    if report_df.empty:
        st.warning("No data found for the selected city.")
    else:
        city_scores = report_df.iloc[0]

        st.markdown(f"### Score Comparison for **{selected_city}** under **{category_choice}** category")

        # Violin plots
        cols = st.columns(len(score_categories))
        for i, category in enumerate(score_categories):
            with cols[i]:
                fig = px.violin(
                    filtered_df,
                    y=category,
                    box=True,
                    points="all",
                    title=category,
                    labels={category: "Score"},
                )
                fig.add_shape(
                    type="line",
                    x0=-0.5, x1=0.5,
                    y0=city_scores[category], y1=city_scores[category],
                    line=dict(color="red", dash="dash"),
                )
                fig.add_annotation(
                    x=0,
                    y=city_scores[category],
                    text=f"{selected_city}: {city_scores[category]}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Bar Plot Comparison Across Cities")

        for category in score_categories:
            filtered_df["__color__"] = filtered_df["normalized_name"].apply(
                lambda name: "Selected City" if name == selected_city else "Other Cities"
            )

            fig = px.bar(
                filtered_df.sort_values(by=category, ascending=True),
                x=category,
                y="normalized_name",
                orientation="h",
                title=f"{category} Scores – {category_choice} Cities",
                labels={category: "Score", "normalized_name": "City"},
                color="__color__",
                color_discrete_map={"Selected City": "crimson", "Other Cities": "steelblue"},
            )

            fig.update_layout(
                height=400,
                yaxis=dict(tickfont=dict(size=10)),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

with tab8:
    st.header("🏙️ Feature Comparison by Report Type")

    df = pd.read_csv("C:/Users/Admin/Downloads/layer_score.csv")
    df.columns = df.columns.str.strip()
    df["Report Type 1"] = df["Report Type 1"].str.strip()
    df["normalized_name"] = df["normalized_name"].str.strip()

    df = df[df["normalized_name"].str.lower() != "chicago"]  # Remove Chicago

    df[["population", "Local taxes", "median_income", "General Obligation Bonds for End of the Year"]] = \
        df[["population", "Local taxes", "median_income", "General Obligation Bonds for End of the Year"]].replace('[\$,]', '', regex=True).apply(pd.to_numeric, errors="coerce")

    df["taxes_per_capita"] = df["Local taxes"] / df["population"]
    df = df.dropna(subset=["normalized_name", "Report Type 1"])
    feature_cols = ["population", "taxes_per_capita", "median_income", "General Obligation Bonds for End of the Year"]
    df = df.dropna(subset=feature_cols)

    category_choice = st.selectbox("Select Report Type Category", ["Climate", "Non Climate"], key="tab8_type")
    filtered_df = df[df["Report Type 1"] == category_choice]
    city_list = sorted(filtered_df["normalized_name"].unique())
    selected_city = st.selectbox("Select City", city_list, key="tab8_city")

    city_row = filtered_df[filtered_df["normalized_name"] == selected_city]
    if city_row.empty:
        st.warning("No data for the selected city.")
    else:
        city_row = city_row.iloc[0]
        st.markdown(f"### Feature Comparison for **{selected_city}** under **{category_choice}** category")

        col1, col2 = st.columns(2)
        for i, feature in enumerate(feature_cols):
            with col1 if i % 2 == 0 else col2:
                fig = px.violin(filtered_df, y=feature, box=True, points="all", title=feature)
                fig.add_shape(type="line", x0=-0.5, x1=0.5, y0=city_row[feature], y1=city_row[feature],
                              line=dict(color="red", dash="dash"))
                fig.add_annotation(x=0, y=city_row[feature], text=f"{selected_city}: {city_row[feature]:,.0f}",
                                   showarrow=True, arrowhead=1, yshift=10)
                st.plotly_chart(fig, use_container_width=True)
with tab9:

    st.subheader("📊 Demographics – Top Quartile Cities (Side-by-Side Comparison)")

    import pandas as pd
    import plotly.express as px
    import streamlit as st

    # Load and clean data
    df = pd.read_csv("C:/Users/Admin/Downloads/layer_score.csv")
    df.columns = df.columns.str.strip()
    df = df[df["normalized_name"].str.lower() != "chicago"]

    # Convert fields
    df["median_income"] = pd.to_numeric(df["median_income"].replace('[\$,]', '', regex=True), errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["Local taxes"] = pd.to_numeric(df["Local taxes"].replace('[\$,]', '', regex=True), errors="coerce")
    df["Total Score"] = pd.to_numeric(df["Total Score"], errors="coerce")
    df["taxes_per_capita"] = df["Local taxes"] / df["population"]

    # Drop missing data
    df = df.dropna(subset=["median_income", "population", "taxes_per_capita", "Total Score"])

    # Filter to top quartile by Total Score
    q3 = df["Total Score"].quantile(0.75)
    df_top = df[df["Total Score"] >= q3].copy()

    # Median values from full dataset
    medians = {
        "Median Income": df["median_income"].median(),
        "Population": df["population"].median(),
        "Taxes Per Capita": df["taxes_per_capita"].median()
    }

    # Create three separate plots
    def create_box(feature, label):
        fig = px.box(
            df_top,
            y=feature,
            points="all",
            hover_data=["normalized_name"],
            color_discrete_sequence=["lightblue"]
        )

        # Add red median line
        fig.add_shape(
            type="line",
            x0=-0.5, x1=0.5,
            y0=medians[label], y1=medians[label],
            line=dict(color="red", dash="dash", width=2)
        )

        # Add annotation
        fig.add_annotation(
            x=0,
            y=medians[label],
            text=f"Median: {medians[label]:,.0f}",
            showarrow=True,
            arrowhead=1,
            yshift=10,
            font=dict(color="red", size=10)
        )

        # Update layout
        fig.update_layout(
            title=label,
            font=dict(color="black"),
            xaxis=dict(
                showticklabels=False,
                title=dict(text=""),
                tickfont=dict(color="black")
            ),
            yaxis=dict(
                title=dict(text=label, font=dict(color="black")),
                tickfont=dict(color="black")
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    # Arrange in 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_box("median_income", "Median Income"), use_container_width=True)
    with col2:
        st.plotly_chart(create_box("population", "Population"), use_container_width=True)
    with col3:
        st.plotly_chart(create_box("taxes_per_capita", "Taxes Per Capita"), use_container_width=True)

with tab10:
    st.header("🌍 Climate Report Type – Skokie & Niles Highlighted")

    # Load and clean data
    df = pd.read_csv("C:/Users/Admin/Downloads/layer_score.csv")
    df.columns = df.columns.str.strip()
    df["Report Type 1"] = df["Report Type 1"].str.strip()
    df["normalized_name"] = df["normalized_name"].str.strip()
    df = df[df["normalized_name"].str.lower() != "chicago"]  # Remove Chicago

    # Score columns
    score_cols = ["Governance", "Data and Analytics", "Action Planning", "Total Score"]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Report Type 1"] == "Climate"]
    df = df.dropna(subset=score_cols + ["normalized_name"])

    target_cities = {"Skokie": "red", "Niles": "blue"}
    st.subheader("📈 Score Distributions – Skokie & Niles Highlighted")

    col1, col2 = st.columns(2)
    for i, score in enumerate(score_cols):
        with col1 if i % 2 == 0 else col2:
            fig = px.violin(df, y=score, box=True, points="all", title=f"{score} Distribution", hover_data=["normalized_name"])
            for city, color in target_cities.items():
                city_val = df.loc[df["normalized_name"].str.lower() == city.lower(), score]
                if not city_val.empty:
                    val = city_val.values[0]
                    fig.add_shape(type="line", x0=-0.5, x1=0.5, y0=val, y1=val, line=dict(color=color, dash="dash"))
                    fig.add_annotation(x=0, y=val, text=f"{city}: {val:.1f}",
                                    showarrow=True, arrowhead=1, yshift=10, font=dict(color=color))
            fig.update_layout(
                yaxis=dict(
                    title=dict(text="Score", font=dict(color="black")),
                    tickfont=dict(color="black"),
                    linecolor="black"
                ),
                xaxis=dict(
                    tickfont=dict(color="black"),
                    linecolor="black"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Demographic & Financial Feature Distributions – Skokie & Niles")

    # Convert fields
    df["median_income"] = pd.to_numeric(df["median_income"].replace('[\$,]', '', regex=True), errors="coerce")
    df["Local taxes"] = pd.to_numeric(df["Local taxes"].replace('[\$,]', '', regex=True), errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["taxes_per_capita"] = df["Local taxes"] / df["population"]

    demo_vars = ["population", "median_income", "Local taxes", "taxes_per_capita"]
    demo_labels = {
        "population": "Population",
        "median_income": "Median Income",
        "Local taxes": "Local Taxes",
        "taxes_per_capita": "Taxes Per Capita"
    }

    df = df.dropna(subset=demo_vars)

    col1, col2 = st.columns(2)
    for i, var in enumerate(demo_vars):
        with col1 if i % 2 == 0 else col2:
            fig = px.violin(df, y=var, box=True, points="all", title=f"{demo_labels[var]} Distribution", hover_data=["normalized_name"])
            for city, color in target_cities.items():
                city_val = df.loc[df["normalized_name"].str.lower() == city.lower(), var]
                if not city_val.empty:
                    val = city_val.values[0]
                    fig.add_shape(type="line", x0=-0.5, x1=0.5, y0=val, y1=val, line=dict(color=color, dash="dash"))
                    fig.add_annotation(x=0, y=val, text=f"{city}: {val:,.0f}",
                                    showarrow=True, arrowhead=1, yshift=10, font=dict(color=color))
            fig.update_layout(
                yaxis=dict(
                    title=dict(text=demo_labels[var], font=dict(color="black")),
                    tickfont=dict(color="black"),
                    linecolor="black"
                ),
                xaxis=dict(
                    tickfont=dict(color="black"),
                    linecolor="black"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("📈 Combined Score Distribution – Skokie & Niles Highlighted")

    # Melt the dataframe to long format
    df_melted = df.melt(
        id_vars=["normalized_name"],
        value_vars=["Governance", "Data and Analytics", "Action Planning"],
        var_name="Feature",
        value_name="Score"
    )

    fig = px.violin(
        df_melted,
        x="Feature",
        y="Score",
        box=True,
        points="all",
        hover_data=["normalized_name"],
        title="Governance, Data & Analytics, Action Planning – Combined Violin Plot"
    )

    # Add Skokie and Niles lines
    for city, color in target_cities.items():
        for feature in ["Governance", "Data and Analytics", "Action Planning"]:
            city_val = df.loc[df["normalized_name"].str.lower() == city.lower(), feature]
            if not city_val.empty:
                val = city_val.values[0]
                fig.add_shape(
                    type="line",
                    x0=feature,
                    x1=feature,
                    y0=val,
                    y1=val,
                    xref='x',
                    yref='y',
                    line=dict(color=color, dash="dash")
                )
                fig.add_annotation(
                    x=feature,
                    y=val,
                    text=f"{city}: {val:.1f}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10,
                    font=dict(color=color)
                )

    fig.update_layout(
        yaxis=dict(
            title=dict(text="Score", font=dict(color="black")),
            tickfont=dict(color="black"),
            linecolor="black"
        ),
        xaxis=dict(
            title=dict(text="Feature", font=dict(color="black")),
            tickfont=dict(color="black"),
            linecolor="black"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

