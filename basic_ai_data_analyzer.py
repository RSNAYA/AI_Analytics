import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("student_scores.csv")
X = data[["Test1", "Test2"]]
y = data["FinalExam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

plt.plot(y_test.values, label="Actual Scores", marker="o")
plt.plot(y_pred, label="Predicted Scores", marker="x")
plt.title("Actual vs Predicted Final Exam Scores")
plt.xlabel("Student")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_graph.png")
plt.show()