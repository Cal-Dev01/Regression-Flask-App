import pandas as pd
import numpy as np

# Generate random data for regression demo
np.random.seed(42)
size = np.random.uniform(500, 4000, 100)
price = 50 * size + np.random.normal(0, 50000, 100)

# Create DataFrame
regression_data = pd.DataFrame({
    'size': size,
    'price': price
})

# Save to CSV
regression_file_path = 'regression_data.csv'
regression_data.to_csv(regression_file_path, index=False)

print(f"CSV file generated and saved as {regression_file_path}")
