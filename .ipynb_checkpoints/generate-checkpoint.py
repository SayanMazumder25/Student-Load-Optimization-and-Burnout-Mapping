import numpy as np
import pandas as pd

np.random.seed(42)

n = 80000

test = pd.DataFrame({
    "Study Hours": np.random.randint(0, 11, n),
    "Gym Hours": np.random.randint(0, 11, n),
    "Debate Prep Hours": np.random.randint(0, 11, n),
    "Sleep Hours": np.random.randint(0, 11, n),
    "Reported Stress Level": np.random.randint(0, 11, n),
    "Academic Score": np.random.randint(0, 11, n)
})

test.to_csv("test.csv", index=False)

print("test.xlsx generated successfully!")
print("Rows:", len(test))
print(test.head(10))