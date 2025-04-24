import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
from keras.utils import to_categorical
from pandas.plotting import parallel_coordinates
import plotly.express as px
import seaborn as sns
import os

# Load data
data = pd.read_csv('network_traffic_data.txt', delimiter='\t')

# Output directory
output_dir = "output_analysis"
os.makedirs(output_dir, exist_ok=True)

# Preprocess
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input for Conv1D (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Save visualizations
plt.figure(figsize=(10, 6))
plt.plot(X_train[0])
plt.title('Network Traffic Visualization')
plt.xlabel('Time')
plt.ylabel('Traffic')
plt.savefig(os.path.join(output_dir, 'line_plot.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(X_train.flatten(), bins=50, alpha=0.7, label='Training')
plt.hist(X_test.flatten(), bins=50, alpha=0.7, label='Testing')
plt.title('Histogram of Network Traffic Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(output_dir, 'histogram.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(X_train.reshape(X_train.shape[0], X_train.shape[1]))
plt.title('Boxplot of Network Traffic Features')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.savefig(os.path.join(output_dir, 'boxplot.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0, 0], X_train[:, 1, 0], c=y_train.argmax(axis=1), cmap='viridis', alpha=0.7)
plt.title('Scatter Plot of First Two Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
plt.close()

pd.plotting.scatter_matrix(pd.DataFrame(X_train.reshape(X_train.shape[0], X_train.shape[1]), 
columns=['feature1', 'feature2', 'feature3']), figsize=(10, 10), diagonal='kde')
plt.suptitle('Pair Plot of Features')
plt.savefig(os.path.join(output_dir, 'pair_plot.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Features')
plt.savefig(os.path.join(output_dir, 'heatmap.png'))
plt.close()

plt.figure(figsize=(6, 6))
data['label'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Traffic Distribution')
plt.ylabel('')
plt.savefig(os.path.join(output_dir, 'pie_chart.png'))
plt.close()

plt.figure(figsize=(12, 6))
parallel_coordinates(data, 'label', color=['blue', 'red'])
plt.title('Parallel Coordinates Plot')
plt.savefig(os.path.join(output_dir, 'parallel_coordinates.png'))
plt.close()

# Tree Map
fig = px.treemap(data, path=[px.Constant("All Traffic"), 'label'], values='feature1')
fig.update_layout(margin=dict(t=25, l=25, r=25, b=25))
fig.write_html(os.path.join(output_dir, 'treemap.html'))

# CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

# Save performance to file
with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
    f.write(f'Test Accuracy: {accuracy:.2f}\n')

print("All plots and results saved to 'output_analysis'. Opening 'int dashboard.html' to view integrated output.")

file_path = "network_traffic_data.txt"
txt_file = "traffic_analysis_report.txt"
html_file = "traffic_analysis_report.html"

data = pd.read_csv(file_path, delimiter='\t')

# Traffic distribution
botnet_count = (data['label'] == 'botnet').sum()
normal_count = (data['label'] == 'normal').sum()

# Ratio calculation
botnet_ratio = botnet_count / (botnet_count + normal_count)
normal_ratio = normal_count / (botnet_count + normal_count)

# Display statistics
print("Network Traffic Analysis Report")
print("="*40)
print(f"Total Records: {len(data)}")
print(f"Botnet Count: {botnet_count}")
print(f"Normal Count: {normal_count}")
print(f"Botnet Ratio: {botnet_ratio:.2%}")
print(f"Normal Ratio: {normal_ratio:.2%}")
print("\nFeature Statistics:")
print(data.describe())

# Save the analysis report
output_file = "traffic_analysis_report.txt"
with open(output_file, "w") as f:
    f.write("Network Traffic Analysis Report\n")
    f.write("="*40 + "\n")
    f.write(f"Total Records: {len(data)}\n")
    f.write(f"Botnet Count: {botnet_count}\n")
    f.write(f"Normal Count: {normal_count}\n")
    f.write(f"Botnet Ratio: {botnet_ratio:.2%}\n")
    f.write(f"Normal Ratio: {normal_ratio:.2%}\n")
    f.write("\nFeature Statistics:\n")
    f.write(data.describe().to_string())
    
print(f"\nReport saved to {output_file}")
# Read the file
with open(txt_file, "r") as file:
    content = [line.strip() for line in file.readlines() if line.strip()]

# Safe parsing with fallback values
def get_line_value(index, default="N/A"):
    try:
        line = content[index]
        print(f"Parsing line {index}: {line}")  # DEBUG LINE
        return line.split(":")[1].strip()
    except (IndexError, ValueError) as e:
        print(f"Error at line {index}: {e}")  # DEBUG LINE
        return default


# Start building HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Traffic Analysis Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet" />
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: #f6f7fb;
      color: #2c3e50;
    }
    header {
      background-color: #ffffff;
      padding: 2rem 4rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #eee;
    }
    header h1 {
      font-size: 1.8rem;
      font-weight: 800;
      color: #0f172a;
    }
    .container {
      max-width: 1200px;
      margin: 2rem auto;
      padding: 0 2rem;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 2rem;
      margin-bottom: 3rem;
    }
    .stat-card {
      background: white;
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.05);
      transition: transform 0.2s ease;
    }
    .stat-card:hover {
      transform: translateY(-5px);
    }
    .stat-title {
      font-size: 1rem;
      color: #64748b;
      margin-bottom: 0.5rem;
    }
    .stat-value {
      font-size: 1.6rem;
      font-weight: 700;
    }
    .botnet { color: #e11d48; }
    .normal { color: #10b981; }

    .feature-section {
      background: white;
      border-radius: 16px;
      padding: 2rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    }
    .feature-section h2 {
      font-size: 1.5rem;
      margin-bottom: 1rem;
      color: #0f172a;
    }
    .report-content {
      font-family: monospace;
      font-size: 0.9rem;
      background: #f9fafb;
      padding: 1rem;
      border-radius: 8px;
      white-space: pre-wrap;
      overflow-x: auto;
    }
    footer {
      text-align: center;
      padding: 2rem;
      font-size: 0.9rem;
      background-color: #1e293b;
      color: white;
      margin-top: 4rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Network Traffic Analysis</h1>
    <a href="output_analysis/dashboard.html" style="text-decoration: none; background: #0f172a; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600;">Go to Dashboard</a>
  </header>

  <div class="container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">Total Records</div>
        <div class="stat-value" id="totalRecords">""" + get_line_value(2) + """</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Botnet Traffic</div>
        <div class="stat-value botnet" id="botnetCount">""" + get_line_value(3) + """</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Normal Traffic</div>
        <div class="stat-value normal" id="normalCount">""" + get_line_value(4) + """</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Botnet Ratio</div>
        <div class="stat-value botnet" id="botnetRatio">""" + get_line_value(5) + """</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Normal Ratio</div>
        <div class="stat-value normal" id="normalRatio">""" + get_line_value(6) + """</div>
      </div>
    </div>

    <div class="feature-section">
      <h2>Feature Statistics</h2>
      <div class="report-content" id="featureStats">
"""

# Append feature stats lines (from line 8 onward)
for line in content[8:]:
    html_content += line + "<br>"

html_content += """
      </div>
    </div>
  </div>

  <footer>
    PROJECT BY PRANAV, KUSHAL, HARI, DATTA | 2025
  </footer>

</body>
</html>
"""

# Save HTML file
with open(html_file, "w") as file:
    file.write(html_content)

print(f"Enhanced HTML report saved as {html_file}")

dashboard_path = os.path.join(output_dir, 'dashboard.html')
with open(dashboard_path, 'w') as f:
    f.write(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Traffic Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: #f4f6f9;
                color: #333;
                padding: 40px;
            }}
            h2 {{
                margin-top: 40px;
                color: #3A506B;
            }}
            img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            iframe {{
                width: 100%;
                height: 600px;
                border: none;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Network Traffic Visualizations</h1>
        <h2>Line Plot</h2>
        <img src="line_plot.png" alt="Line Plot">

        <h2>Histogram</h2>
        <img src="histogram.png" alt="Histogram">

        <h2>Boxplot</h2>
        <img src="boxplot.png" alt="Boxplot">

        <h2>Scatter Plot</h2>
        <img src="scatter_plot.png" alt="Scatter">

        <h2>Pair Plot</h2>
        <img src="pair_plot.png" alt="Pair Plot">

        <h2>Heatmap</h2>
        <img src="heatmap.png" alt="Heatmap">

        <h2>Traffic Distribution Pie Chart</h2>
        <img src="pie_chart.png" alt="Pie Chart">

        <h2>Parallel Coordinates Plot</h2>
        <img src="parallel_coordinates.png" alt="Parallel Coordinates">

        <h2>Tree Map</h2>
        <iframe src="treemap.html"></iframe>

        <h2>Model Accuracy</h2>
        <pre>{open(os.path.join(output_dir, 'model_metrics.txt')).read()}</pre>
    </body>
    </html>
    """)

dashboard_file = os.path.abspath("int_dashboard.html")

webbrowser.open("int_dashboard.html")
