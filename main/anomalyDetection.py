import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def detect_anomalies_isolation_forest(data, time_column, contamination=0.05):
    # Preprocess the data
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    numeric_data = numeric_data.fillna(0)  # Replace missing values with 0 or use appropriate imputation method

    # Convert non-numeric values to floats
    for column in numeric_data.columns:
        numeric_data[column] = pd.to_numeric(numeric_data[column], errors='coerce')
        numeric_data[column] = numeric_data[column].astype(float)

    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(numeric_data)

    # Train the Isolation Forest model
    isolation_forest = IsolationForest(contamination=contamination)
    isolation_forest.fit(data_normalized)

    # Predict anomalies
    anomaly_scores = isolation_forest.decision_function(data_normalized)
    anomalies = data[anomaly_scores < 0]

    return anomalies






df = pd.read_csv("/Users/adilrasheed/Downloads/raw_mill_trip_removed_zero.csv", index_col = 0)
# Drop the 'time' column
# df = df.drop(columns='time')

df = df.reset_index(drop=True)
df = df[(df != 0).all(1)]

# Reset the index
df = df.reset_index(drop=True)

time_column = 'time'
anomalies = detect_anomalies_isolation_forest(df, time_column, contamination=0.03)


# Plot the original data and the anomalous points against time and save the images
for column_name in anomalies.columns:
    if column_name != time_column:
        plt.plot(df[time_column], df[column_name], label='Normal')
        plt.plot(anomalies[time_column], anomalies[column_name], 'ro', label='Anomalous')
        plt.xlabel('Time')
        plt.ylabel(column_name)
        plt.title('Anomalous Points for {}'.format(column_name))
        plt.legend()
        
        #Format the time ticks on the x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Save the image
        plt.savefig('anomalies_{}.png'.format(column_name))  # Replace 'anomalies_{}.png' with the desired file name
        plt.close()  # Close the figure to free up resources





