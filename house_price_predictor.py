import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('house_price_dataset.csv')

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Display the dataset
print("Dataset:\n", df)

# Step 4: Define features (X) and target (y)
X = df[["area", "bedrooms", "age"]]  # features
y = df["price"]                      # target/output

# Step 5: Check shapes
print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model= LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained succesfully")


y_pred = model.predict(X_test)
print("\nPredicted prices for test set:",y_pred)
print("Actual prices:", list(y_test))

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error(MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 score:", r2)


print("n\Predict a new house price:")
area = float(input("Enter Area(sqft):"))
bedrooms = int(input("Enter your number of bedrooms:"))
age = int(input("Enter house age(years):"))

new_house = pd.DataFrame([[area, bedrooms, age]], columns=["area", "bedrooms", "age"])
predicted_price = model.predict(new_house)
print(f"Predicted price for the house: $ {predicted_price[0]:,.2f}")