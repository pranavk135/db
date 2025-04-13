
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

print("All plots and results saved to 'output_analysis'. Open 'dashboard.html' to view integrated output.")
