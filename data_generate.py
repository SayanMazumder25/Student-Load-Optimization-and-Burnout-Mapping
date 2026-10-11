import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n = 10000

# 1. Generate Base Variables
student_ids = [f"STU_{str(i).zfill(5)}" for i in range(1, n+1)]

# Study, Debate, and Gym hours
study = np.random.normal(4.5, 1.5, n).clip(0, 10)
debate = np.random.exponential(1.5, n).clip(0, 6) # Right-skewed: many do little, some do a lot
gym = np.random.normal(1.2, 0.8, n).clip(0, 3)

# 2. Build the Correlations (The hidden story you will uncover)
# Sleep suffers as extracurricular load increases
sleep = np.random.normal(8.0, 1.0, n) - (debate * 0.3) - (gym * 0.2)
sleep = sleep.clip(3, 10)

# Stress spikes with low sleep and high cognitive/physical load
stress_continuous = 10 - sleep + (debate * 0.5) + (gym * 0.3) + np.random.normal(0, 1, n)
stress = stress_continuous.clip(1, 10).round()

# Academic Score: Increases with study hours, but crashes severely if stress hits 8 or higher (Burnout)
score = 50 + (study * 6) + (debate * 1.5) 
burnout_penalty = np.where(stress >= 8, (stress - 7) * 8, 0) # The tipping point
score = (score - burnout_penalty + np.random.normal(0, 4, n)).clip(0, 100).round()

# 3. Assemble the DataFrame
df = pd.DataFrame({
    'Student_ID': student_ids,
    'Daily_Study_Hours': np.round(study, 1),
    'Debate_Prep_Hours': np.round(debate, 1),
    'Gym_Hours': np.round(gym, 1),
    'Average_Sleep_Hours': np.round(sleep, 1),
    'Reported_Stress_Level': stress,
    'Academic_Score': score
})

# 4. Inject Controlled Chaos (For your Pandas cleaning pipeline)
# Introduce missing values (NaNs)
df.loc[np.random.choice(n, 450, replace=False), 'Average_Sleep_Hours'] = np.nan
df.loc[np.random.choice(n, 200, replace=False), 'Gym_Hours'] = np.nan

# Introduce string corruption into the numeric Stress Level column
stress_chaos = {10.0: 'TEN', 9.0: 'Nine', 1.0: 'one', 8.0: 'EIGHT'}
df['Reported_Stress_Level'] = df['Reported_Stress_Level'].replace(stress_chaos)

# Corrupt some Student IDs (lowercase, trailing spaces)
messy_indices = np.random.choice(n, 150, replace=False)
df.loc[messy_indices, 'Student_ID'] = df.loc[messy_indices, 'Student_ID'].str.lower() + " "

# 5. Export to CSV
df.to_csv('campus_burnout_data.csv', index=False)
print("Successfully generated campus_burnout_data.csv with 10,000 rows.")