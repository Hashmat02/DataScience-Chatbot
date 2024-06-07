import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('df.csv')

df['gender'] = df['Sex']

plt.figure(figsize=(8, 6))
df.groupby(['gender', 'Survived']).size().unstack().plot(kind='bar', stacked=True)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Survival Count by Gender')
plt.legend(title='Survived', labels=['No', 'Yes'])

plt.savefig('EDA_Agent/graphs/gender_vs_survived.png')