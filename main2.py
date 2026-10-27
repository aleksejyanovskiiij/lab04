import pandas as pd
import numpy as np
import scipy.stats as stats

ALPHA = 0.05 # для p-значения

def format_p(p):
    if abs(p) < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"

indicators = [
    "осведомленность",
    "скрытые фигуры",
    "пропущенные слова",
    "арифметика",
    "понятливость",
    "исключение изображений",
    "аналогии",
    "числовые ряды",
    "умозаключения",
    "геометрическое сложение",
    "заучивание слов",
    "средний IQ",
    "экстраверсия- интроверсия",
    "нейротизм",
    "средняя отметка",
]

df = pd.read_excel("Психологи.xlsx", sheet_name="Лист2")
df.columns = [" ".join(str(column).split()) for column in df.columns]

print("Размер набора данных:", df.shape)
print("Первые строки таблицы:")
print(df.head())
print("Количество юношей и девушек:")
print(df["пол"].value_counts().to_dict())
print()

male_sample = df.loc[df["пол"] == "м"]
female_sample = df.loc[df["пол"] == "ж"]

# сравнение показателей по полу с помощью t-тестов
results = []

for col in indicators:
    male_values = male_sample[col].dropna().to_numpy()
    female_values = female_sample[col].dropna().to_numpy()

    print(f"Показатель {col}")
    print(f"Среднее юношей: {format_p(np.mean(male_values))}")
    print(f"Среднее девушек: {format_p(np.mean(female_values))}")
    print("Гипотезы: h0 - средние значения в группах равны (различий нет), h1 - средние значения различаются")

    _, p_male = stats.shapiro(male_values)
    _, p_female = stats.shapiro(female_values)
    print(f"Нормальность по шапиро-уилку: юноши p == {format_p(p_male)}, девушки p == {format_p(p_female)}")

    levens_stat, p_levene = stats.levene(male_values, female_values)
    print("Критерий Ливена о равенстве дисперсий: статистика =", format_p(levens_stat), ", p-значение =", format_p(p_levene))
    if p_levene < ALPHA:
        test_name = "t-критерий Уэлча (дисперсии различаются)"
        equal_var = False
    else:
        test_name = "t-критерий Стьюдента (дисперсии равны)"
        equal_var = True
    print(f"Выбран {test_name}")

    t_stat, p_value = stats.ttest_ind(male_values, female_values, equal_var=equal_var)

    n1, n2 = len(male_values), len(female_values)
    var1, var2 = np.var(male_values, ddof=1), np.var(female_values, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = abs(np.mean(male_values) - np.mean(female_values)) / pooled_std

    print(f"Статистика t-теста: {format_p(t_stat)}")
    print(f"p-значение: {format_p(p_value)}")
    print(f"Размер эффекта (d Коэна): {format_p(d)}")
    if p_value < ALPHA:
        print("Вывод: p-значение меньше 0.05, гипотеза h0 отвергается => гендерные различия статистически значимы")
    else:
        print("Вывод: p-значение больше или равно 0.05, гипотеза h0 не отвергается => значимых гендерных различий нет")

    results.append((col, abs(t_stat), p_value, d))
    print()

# ранжирование показателей по силе гендерных различий
print("Сортировка по модулю t-статистики по убыванию: ")
for column, abs_t, p_value, d in sorted(results, key=lambda item: item[1], reverse=True):
    significance = "значимо" if p_value < ALPHA else "не значимо"
   
    # чтобы по красоте было
    col = column + " " * (26 - len(column))
   
    print(f"{col}|t == {format_p(abs_t)}\t| p == {format_p(p_value)}\t| d == {format_p(d)}\t| {significance}")
print()

print(f"Сильнее всего гендерные различия проявились для показателя '{sorted(results, key=lambda item: item[1], reverse=True)[0][0]}'")
print(f"Слабее всего гендерные различия проявились для показателя '{sorted(results, key=lambda item: item[1])[0][0]}'")
print()
