import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns


def load_and_filter_data(file_path, agricultural_land_indicator,
                         population_indicator, year):
    """
    Load and filter data based on indicators and year.

    Parameters:
    - file_path (str): Path to the CSV file.
    - agricultural_land_indicator (str): Indicator name for agricultural land.
    - population_indicator (str): Indicator name for population.
    - year (str): Year for data extraction.

    Returns:
    - pd.DataFrame: Merged and filtered data.
    """
    df = pd.read_csv(file_path, skiprows=3)
    data1 = df[df['Indicator Name'] == agricultural_land_indicator][[
        'Country Name', year]].rename(columns={year: 
                                               agricultural_land_indicator})
    data2 = df[df['Indicator Name'] == population_indicator][[
        'Country Name', year]].rename(columns={year:
                                               population_indicator})
    merged_data = pd.merge(data1, data2, on='Country Name',
                           how='outer').dropna().reset_index(drop=True)
    merge_trans = merged_data.transpose()
    return merged_data


def calculate_kmeans_inertia(data, max_clusters=10):
    """
    Calculate inertia for KMeans clustering.

    Parameters:
    - data (pd.DataFrame): Input data.
    - max_clusters (int): Maximum number of clusters.

    Returns:
    - list: Inertia values for different cluster counts.
    """
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias


def apply_kmeans_and_plot_subplots(df1, df2, cluster_columns, num_clusters=4,
                                   title1='', title2=''):
    """
    Apply KMeans clustering and plot subplots.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame for clustering and plotting.
    - df2 (pd.DataFrame): Second DataFrame for clustering and plotting.
    - cluster_columns (list): Columns for clustering.
    - num_clusters (int): Number of clusters.
    - title1 (str): Title for the first subplot.
    - title2 (str): Title for the second subplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for data_frame, subplot_title, axis in zip([df1, df2], [title1, title2],
                                               axes):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data_frame['cluster'] = kmeans.fit_predict(data_frame[cluster_columns])
        data_frame['cluster'] = data_frame['cluster'].astype('category')
        cluster_centers = kmeans.cluster_centers_
        scatter_plot = sns.scatterplot(
            x=cluster_columns[0], y=cluster_columns[1], hue='cluster',
            data=data_frame, ax=axis)
        axis.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                     marker='+', s=50, c='black', label='Cluster Centers')
        axis.set_title(subplot_title)
        axis.set_ylabel('Total Population (Millions)')  # Updated y-axis label
        # Updated x-axis label
        axis.set_xlabel('Agricultural Land (% of Land Area)')
        legend_handles = [Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=f'C{i}', markersize=8) for
                          i in range(num_clusters)]
        legend_handles.append(
            Line2D([0], [0], marker='+', color='black', markersize=8))
        legend_labels = [
            f'Cluster {i + 1}' for i in range(num_clusters)] + [
                'Cluster Centers']
        axis.legend(handles=legend_handles, labels=legend_labels, 
                    loc='upper left', fontsize='small', handlelength=0.5,
                    handletextpad=0.5)
    plt.tight_layout()
    plt.savefig('Cluster_plot.png')


def plot_polynomial_regression_with_error(x, y, degree, ax, title,
                                          curve_color, data_color):
    """
    Plot data with polynomial regression and error range.

    Parameters:
    - x (pd.DataFrame): Input features.
    - y (pd.Series): Target variable.
    - degree (int): Polynomial degree.
    - ax (matplotlib.axes._subplots.AxesSubplot): Plot axes.
    - title (str): Plot title.
    - curve_color (str): Color for the fitted curve.
    - data_color (str): Color for actual data points.
    """
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    x_pred = poly_features.transform(
        pd.DataFrame(all_years_extended, columns=['Year']))
    forecast_values = model.predict(x_pred)
    n_bootstraps = 1000
    bootstrapped_predictions = np.zeros((n_bootstraps, len(x_pred)))
    for i in range(n_bootstraps):
        indices = np.random.choice(len(x), len(x))
        x_bootstrapped = x.iloc[indices]
        y_bootstrapped = y.iloc[indices]
        x_poly_bootstrapped = poly_features.transform(x_bootstrapped)
        model.fit(x_poly_bootstrapped, y_bootstrapped)
        bootstrapped_predictions[i, :] = model.predict(x_pred)
    lower_bound = np.percentile(bootstrapped_predictions, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_predictions, 97.5, axis=0)
    ax.plot(x, y, marker='.', linestyle='-',
            label='Actual Data', color=data_color)
    ax.plot(all_years_extended, forecast_values, label='Fitted Curve',
            linestyle='-', color=curve_color)
    prediction_2025 = forecast_values[-1]
    ax.plot(2025, prediction_2025, marker='o', markersize=8,
            label=f'Prediction for 2025: {prediction_2025:.2f}', color='black')
    ax.fill_between(all_years_extended, lower_bound, upper_bound,
                    color=curve_color, alpha=0.3,
                    label='95% Confidence Interval')
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage of Land Area')
    ax.set_xlim(1980, 2030)
    ax.set_xticks(range(1980, 2031, 5))
    ax.grid(True, linestyle='--', alpha=0.8)
    ax.legend(fontsize=7)


# Example usage
file_path = '/Users/user1/Documents/ADS3/API_19_DS2_en_csv_v2_6300757/API_19_DS2_en_csv_v2_6300757.csv'
agricultural_land_indicator = 'Agricultural land (% of land area)'
population_indicator = 'Population, total'
year_2005 = '2005'
year_2020 = '2020'

data_2005 = load_and_filter_data(file_path, agricultural_land_indicator,
                                 population_indicator, year_2005)
data_2020 = load_and_filter_data(file_path, agricultural_land_indicator,
                                 population_indicator, year_2020)

x_2005 = data_2005[[agricultural_land_indicator, population_indicator]]
x_2020 = data_2020[[agricultural_land_indicator, population_indicator]]

if not x_2005.empty:
    inertias_2005 = calculate_kmeans_inertia(x_2005)
    plt.plot(range(1, 11), inertias_2005, marker='o', label=f'{year_2005} - X')
else:
    print("DataFrame x_2005 is empty. Check your data filtering for 2005.")

if not x_2020.empty:
    inertias_2020 = calculate_kmeans_inertia(x_2020)
    plt.plot(range(1, 11), inertias_2020, marker='o', label=f'{year_2020} - X')
else:
    print("DataFrame x_2020 is empty. Check your data filtering for 2020.")

plt.title('Elbow Plot for KMeans Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()
plt.show()

cluster_columns = [agricultural_land_indicator, population_indicator]

apply_kmeans_and_plot_subplots(data_2005, data_2020, cluster_columns,
                               num_clusters=4,
                               title1=f'{agricultural_land_indicator} and {population_indicator} in {year_2005}',
                               title2=f'{agricultural_land_indicator} and {population_indicator} in {year_2020}')

sns.set_palette("inferno")

df = pd.read_csv(file_path, skiprows=3)
selected_countries = ['India', 'United Kingdom']
indicator_name = agricultural_land_indicator
data_selected = df[(df['Country Name'].isin(selected_countries)) & 
                   (df['Indicator Name'] == indicator_name)].reset_index(
    drop=True)

data_forecast = data_selected.melt(
    id_vars=['Country Name', 'Indicator Name'], var_name='Year',
    value_name='Value')

data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]
data_forecast['Year'] = data_forecast['Year'].astype(int)
data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)
data_forecast = data_forecast[(data_forecast['Year'] >= 1980) & (
    data_forecast['Year'] <= 2020)]

predictions = {}
all_years_extended = list(range(1980, 2026))

actual_data_colors = ['red', 'blue', 'purple']
curve_color = 'Chartreuse'

for country, data_color in zip(selected_countries, actual_data_colors):
    fig, ax = plt.subplots(figsize=(6, 4))
    country_data = data_forecast[data_forecast['Country Name'] == country]
    x_country = country_data[['Year']]
    y_country = country_data['Value']
    plot_polynomial_regression_with_error(
        x_country, y_country, degree=3, ax=ax,
                                          title=f'Agricultural Land Area Forecast for {country}',
                                          curve_color=curve_color,
                                          data_color=data_color)
    filename = f"Total_Population_Forecast_{country.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')