# TEA-of-GDHC-using-DL

Overview
This project implements a Physically Informed Neural Network (PINN) to model and analyze Geothermal District Heating and Cooling (GDHC) systems. By integrating physical laws into a deep learning framework, the model predicts electricity and heat outputs based on operational parameters with enhanced accuracy. The project also performs comprehensive calculations of the Levelized Cost of Electricity (LCOE), Levelized Cost of Heat (LCOH), and conducts a Life Cycle Analysis (LCA), culminating in a detailed Techno-Economic Assessment (TEA) of GDHC systems.

Features
Extended Datasets: Synthetic and real-world data covering various operational scenarios, economic factors, and environmental impacts.
Physically Informed Neural Network: A deep learning model that incorporates physical constraints to improve predictive performance.
Techno-Economic Calculations: Detailed computations of LCOE, LCOH, Net Present Value (NPV), Internal Rate of Return (IRR), and payback period.
Life Cycle Analysis: Assessment of greenhouse gas emissions associated with materials and processes in GDHC systems.
Visualization Tools: Comprehensive plots and graphs for model evaluation and analysis results.
Getting Started
Prerequisites
Python 3.7 or higher
Required Python libraries:
numpy
pandas
tensorflow
scikit-learn
matplotlib
seaborn
numpy-financial

Installation
Clone the repository:
git clone https://github.com/SamJWHu/TEA-of-GDHC-using-DL.git
cd TEA-of-GDHC-using-DL
Install dependencies:
pip install -r requirements.txt


Usage
Data Preparation:

Load and preprocess the extended datasets located in the data/ directory.
Datasets include:
operational_data_extended.csv
economic_data_extended.csv
environmental_data_extended.csv
Model Training:

Run the training scripts in the src/ directory.
The model uses early stopping and model checkpointing to prevent overfitting.
Model Evaluation:

Evaluate the model's performance on the test set.
Generate predictions and compare them with true values.
Analysis:

Perform techno-economic calculations using the provided scripts.
Conduct life cycle analysis to assess environmental impacts.
Visualization:

Generate plots for training/validation loss, predicted vs. true outputs, residuals, and more.
Visualization scripts are available in the src/ directory.
Project Structure
data/: Contains operational, economic, and environmental datasets.
src/: Python scripts and notebooks for model training, evaluation, and analysis.
plots/: Generated figures and visualizations.
README.md: Project description and instructions.
requirements.txt: List of required Python packages.
Key Results
Model Accuracy: The PINN model accurately predicts GDHC system outputs across various operational scenarios.
Economic Insights: Calculated LCOE and LCOH values demonstrate the financial viability of GDHC systems, especially with renewable energy incentives.
Environmental Impact: LCA results show reduced greenhouse gas emissions compared to conventional energy sources.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

License
This project is licensed under the MIT License.

Contact
For questions or collaboration opportunities, don't hesitate to get in touch with Sam Jiun-Wei Hu.
