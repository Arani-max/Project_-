# 1. Импорт библиотек (без seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_columns', None)
plt.style.use('default')  # Используем стандартный стиль вместо seaborn

# 2. Загрузка данных
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# 3. Первичный осмотр (оставляем без изменений)
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о столбцах:")
print(df.info())

print("\nСтатистическое описание числовых данных:")
print(df.describe())

# 4. Обработка данных (оставляем без изменений)
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic_BP', 'Diastolic_BP']] = df[['Systolic_BP', 'Diastolic_BP']].astype(int)
df = df.drop('Blood Pressure', axis=1)

# 5. Разведочный анализ (EDA) с использованием matplotlib вместо seaborn
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Распределение ключевых показателей', fontsize=16)

# Гистограмма продолжительности сна
axes[0, 0].hist(df['Sleep Duration'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Продолжительность сна')
axes[0, 0].set_xlabel('Часы')
axes[0, 0].set_ylabel('Количество')

# Гистограмма качества сна
axes[0, 1].hist(df['Quality of Sleep'], bins=10, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('Качество сна')
axes[0, 1].set_xlabel('Оценка (1-10)')

# Гистограмма уровня стресса
axes[0, 2].hist(df['Stress Level'], bins=10, edgecolor='black', alpha=0.7, color='red')
axes[0, 2].set_title('Уровень стресса')
axes[0, 2].set_xlabel('Уровень (1-10)')

# Гистограмма ежедневных шагов
axes[1, 0].hist(df['Daily Steps'], bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_title('Ежедневные шаги')
axes[1, 0].set_xlabel('Количество шагов')

# Столбчатая диаграмма для пола
gender_counts = df['Gender'].value_counts()
axes[1, 1].bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'], alpha=0.7)
axes[1, 1].set_title('Распределение по полу')
axes[1, 1].set_xlabel('Пол')
axes[1, 1].set_ylabel('Количество')

# Столбчатая диаграмма для расстройств сна
disorder_counts = df['Sleep Disorder'].value_counts()
axes[1, 2].bar(disorder_counts.index, disorder_counts.values, color=['gray', 'red', 'blue'], alpha=0.7)
axes[1, 2].set_title('Расстройства сна')
axes[1, 2].set_xlabel('Тип расстройства')
axes[1, 2].set_ylabel('Количество')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 6. Корреляционный анализ с использованием matplotlib
numeric_cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                'Stress Level', 'Daily Steps', 'Heart Rate', 'Systolic_BP', 'Diastolic_BP']
correlation_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap='coolwarm')

# Добавляем подписи
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.index)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix.index)

# Добавляем значения в ячейки
for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", color="black")

ax.set_title('Матрица корреляций', fontsize=16)
plt.colorbar(im)
plt.tight_layout()
plt.show()

# 7. Группировка и агрегация (оставляем без изменений)
print("Средние показатели по полу:")
gender_stats = df.groupby('Gender').agg({
    'Sleep Duration': 'mean',
    'Quality of Sleep': 'mean',
    'Stress Level': 'mean',
    'Daily Steps': 'mean',
    'Heart Rate': 'mean'
}).round(2)
print(gender_stats)

# 8. Экспорт данных
df.to_csv('sleep_data_processed.csv', index=False)
print("Данные успешно обработаны и сохранены в sleep_data_processed.csv")
