from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import urllib, base64

def linear_regression_view(request):
    # Load dataset (replace with the path of your dataset)
    df = pd.read_csv('linearregression.csv')

    # Check for missing values
    missing_values = df.isnull().sum()

    # Get the first 5 rows of the dataset (head)
    df_head = df.head().to_dict(orient='records')

    # Data visualization: Scatter plot between total_bill and tip
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_bill'], df['tip'], color='green')
    plt.title('Scatter plot of Total Bill vs Tip')
    plt.xlabel('Total Bill')
    plt.ylabel('Tip')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    total_bill_tip_uri = urllib.parse.quote(base64.b64encode(buf.read()))

    # Data visualization: Scatter plot between sex and tip
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sex'], df['tip'], color='purple')
    plt.title('Scatter plot of Sex vs Tip')
    plt.xlabel('Sex')
    plt.ylabel('Tip')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    sex_tip_uri = urllib.parse.quote(base64.b64encode(buf.read()))

    # Data visualization: Scatter plot between smoker and tip
    plt.figure(figsize=(10, 6))
    plt.scatter(df['smoker'], df['tip'])
    plt.title('Scatter plot of Smoker vs Tip')
    plt.xlabel('Smoker')
    plt.ylabel('Tip')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    smoker_tip_uri = urllib.parse.quote(base64.b64encode(buf.read()))

    # Data visualization: Scatter plot between day and tip
    plt.figure(figsize=(10, 6))
    plt.scatter(df['day'], df['tip'], color='red')
    plt.title('Scatter plot of Day vs Tip')
    plt.xlabel('Day')
    plt.ylabel('Tip')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    day_tip_uri = urllib.parse.quote(base64.b64encode(buf.read()))

    # Split data for linear regression
    X = df[['total_bill']]  # Independent variable
    y = df['tip']  # Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model building
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Convert actual and predicted values to a DataFrame for comparison
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  # Show first 10 values

    # Prepare the CSV response if POST request is made
    if request.method == 'POST':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="regression_results.csv"'
        
        # Write the DataFrame to the CSV response
        comparison.to_csv(path_or_buf=response, index=False)

        return response

    # Show the results on the webpage
    return render(request, 'regression/result.html', {
        'mse': mse,
        'r2': r2,
        'missing_values': missing_values.to_dict(),
        'df_head': df_head,  # Pass the first 5 rows of the dataset to the template
        'total_bill_tip_plot_uri': total_bill_tip_uri,
        'sex_tip_plot_uri': sex_tip_uri,
        'smoker_tip_plot_uri': smoker_tip_uri,
        'day_tip_plot_uri': day_tip_uri,
        'comparison': comparison.to_dict(orient='records')  # Convert DataFrame to list of dicts for HTML rendering
    })
