import pandas as pd
import matplotlib.pyplot as plt

# Load the data from Excel file
data = pd.read_excel('./weight_contrast.xlsx')

# Assuming the data contains a column 'ACC' that we need to plot and the x values are described in the user's request
x_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
y_values = (data['ACC'][:10]/100)  # Assuming there are at least 10 ACC values

# Create the plot
plt.figure(figsize=(6,3))
plt.plot(x_values, y_values, marker='o')
# plt.title('ACC Values Over a Coordinates')
# plt.xlabel('a Values')
plt.ylabel('ACC')
# plt.grid(True)
# plt.savefig('./best weights.jpg',dpi=5000)
# plt.savefig('./best weights.pdf',dpi=2000)
plt.show()
