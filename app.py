from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the cleaned dataset
data_cleaned = pd.read_csv('cleaned_nike_data.csv')

# Transaction Analysis

# Top 10 Products by Average Rating
top_10_by_avg_rating = data_cleaned.groupby('name')['avg_rating'].mean().sort_values(ascending=False).head(10)

# Top 10 Products by Number of Reviews
top_10_by_num_reviews = data_cleaned['name'].value_counts().head(10)

# Plotting Top 10 Products by Average Rating
def plot_top_10_avg_rating(top_10_data):
    fig = px.bar(top_10_data, x=top_10_data.index, y=top_10_data.values,
                 title="Top 10 Products by Average Rating", labels={'x': 'Product Name', 'y': 'Average Rating'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

# Plotting Top 10 Products by Number of Reviews
def plot_top_10_num_reviews(top_10_data):
    fig = px.bar(top_10_data, x=top_10_data.index, y=top_10_data.values,
                 title="Top 10 Products by Number of Reviews", labels={'x': 'Product Name', 'y': 'Number of Reviews'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

# Inventory Analysis

# Plot Availability Counts using a bar graph
def plot_availability(data):
    availability_counts = data['availability'].value_counts()
    fig = px.bar(availability_counts, x=availability_counts.index, y=availability_counts.values,
                 title="Product Availability (In Stock vs Out Of Stock)", labels={'x': 'Availability', 'y': 'Count'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

# Plot Price Distribution using Seaborn
def plot_price_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price'], bins=20, kde=True, color='blue')
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Plot Available Sizes using a bar graph
def plot_available_sizes(data):
    available_sizes_counts = data['available_sizes'].value_counts().head(10)
    fig = px.bar(available_sizes_counts, x=available_sizes_counts.index, y=available_sizes_counts.values,
                 title="Top 10 Available Sizes", labels={'x': 'Size', 'y': 'Count'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

# Plot Available Colors using a pie chart
def plot_available_colors(data):
    color_counts = data['color'].value_counts().head(10)
    fig = px.pie(names=color_counts.index, values=color_counts.values, title="Top 10 Available Colors", template='plotly_white')
    return fig.to_html(full_html=False)

# Create summary data
def create_summary():
    total_sales = data_cleaned['price'].count()
    total_revenue = data_cleaned['price'].sum()
    average_price = data_cleaned['price'].mean()
    top_product = data_cleaned.groupby('name')['price'].sum().idxmax()
    
    summary_points = [
        f"Total Sales: {total_sales}",
        f"Total Revenue: ${total_revenue:,.2f}",
        f"Average Price: ${average_price:,.2f}",
        f"Top Selling Product: {top_product}",
        f"Total Unique Products: {data_cleaned['name'].nunique()}",
        f"Total Available Stock: {data_cleaned['availability'].value_counts().get('In Stock', 0)}"
    ]
    return summary_points

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transaction_analysis')
def transaction_analysis():
    avg_rating_graph = plot_top_10_avg_rating(top_10_by_avg_rating)
    num_reviews_graph = plot_top_10_num_reviews(top_10_by_num_reviews)
    return render_template('transaction_analysis.html', avg_rating_graph=avg_rating_graph, num_reviews_graph=num_reviews_graph)

@app.route('/inventory_analysis')
def inventory_analysis():
    availability_graph = plot_availability(data_cleaned)
    price_distribution_graph = plot_price_distribution(data_cleaned)
    available_sizes_graph = plot_available_sizes(data_cleaned)
    available_colors_graph = plot_available_colors(data_cleaned)
    
    return render_template('inventory_analysis.html',
                           availability_graph=availability_graph,
                           price_distribution_graph=price_distribution_graph,
                           available_sizes_graph=available_sizes_graph,
                           available_colors_graph=available_colors_graph)

@app.route('/summary')
def summary():
    summary_points = create_summary()
    return render_template('summary.html', summary_points=summary_points)

if __name__ == '__main__':
    app.run(debug=True)
