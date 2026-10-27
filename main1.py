import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

ALPHA = 0.05 # для p-значения 

def format_p(p):
    if abs(p) < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"

indicators = ["vozrast", "gr-dni", "ctr", "nt", "rost", "ves", "ИМТ"]
names = {
    "vozrast": "Возраст беременной",
    "gr-dni": "Срок гестации в днях",
    "ctr": "КТР (копчико-теменной размер)",
    "nt": "ТВП (толщина воротникового пространства)",
    "rost": "Рост новорождённого",
    "ves": "Вес новорождённого",
    "ИМТ": "Индекс массы тела",
}

df = pd.read_excel("Пренатальная диагностика.xlsx", sheet_name="исходные данные")
print("Размер набора данных:", df.shape)
print("Первые строки таблицы:")
print(df.head())
print()

gender_col = "pol(1-м, 0-д)"

gender_counts = df[gender_col].value_counts().sort_index()
print("Девочек (пол = 0):", int(gender_counts.get(0, 0)))
print("Мальчиков (пол = 1):", int(gender_counts.get(1, 0)))
print()

# ИМТ = масса / рост в квадрате
df["ИМТ"] = (df["ves"] / 1000.0) / ((df["rost"] / 100.0) ** 2)
print("Добавлен столбец ИМТ")
print()

print("Числовые характеристики всех показателей")
print(df[indicators].describe().T.to_string())
print()
rows = []
for col in indicators:
    sample = df[col]
    rows.append({
        "Размах": sample.max() - sample.min(),
        "Асимметрия": format_p(stats.skew(sample, bias=False)),
        "Эксцесс": format_p(stats.kurtosis(sample, bias=False)),
    })
extra_stats = pd.DataFrame(rows, index=indicators)
print(extra_stats.to_string())
print()

print("Проверка показателей на нормальность")
print("Используются гистограмма с kde, асимметрия с эксцессом и критерий шапиро-уилка")
print("Гипотезы: h0 - данные распределены нормально, h1 - данные не распределены нормально")
print()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(indicators):
    sample = df[col].to_numpy()
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    skewness = stats.skew(sample, bias=False)
    kurtosis = stats.kurtosis(sample, bias=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    print("Показатель:", names[col])
    print("Асимметрия:", format_p(skewness))
    print("Эксцесс:", format_p(kurtosis))
    print(f"Критерий шапиро-уилка: статистика W == {format_p(shapiro_stat)}, p-значение == {format_p(shapiro_p)}")
    if shapiro_p < ALPHA:
        print("Вывод: p-значение меньше 0.05, гипотеза h0 отвергается => данные не распределены нормально")
    else:
        print("Вывод: p-значение больше или равно 0,05, гипотеза h0 не отвергается => данные распределены нормально")
    print()

    axis = axes[i]
    axis.hist(sample, bins="auto", density=True, alpha=0.6, color="steelblue", edgecolor="black")
    xs = np.linspace(sample.min(), sample.max(), 300)
    kde = stats.gaussian_kde(sample)
    axis.plot(xs, kde(xs), color="red", label="KDE")
    axis.plot(xs, stats.norm.pdf(xs, mean, std), color="green", label="нормальная кривая")
    axis.set_title(names[col])
    axis.legend()

for j in range(len(indicators), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
print()

# объёмы групп большие, поэтому по центральной предельной теореме t-тест применим
print("Гендерные различия параметров новорождённых")
print()

for col, description in [("rost", "рост"), ("ves", "вес"), ("ИМТ", "индекс массы тела")]:
    male_sample = df.loc[df[gender_col] == 1, col].to_numpy()
    female_sample = df.loc[df[gender_col] == 0, col].to_numpy()

    print("Показатель:", description)
    print("Гипотезы: h0 - средние значения мальчиков и девочек равны; h1 - средние значения различаются")
    print(f"Среднее мальчиков: {format_p(np.mean(male_sample))}")
    print(f"Среднее девочек: {format_p(np.mean(female_sample))}")

    _, p_male = stats.shapiro(male_sample)
    _, p_female = stats.shapiro(female_sample)
    print(f"Проверка нормальности по шапиро-уилку: мальчики p == {format_p(p_male)}, девочки p == {format_p(p_female)}")

    levene_stat, p_levene = stats.levene(male_sample, female_sample)
    print(f"Критерий Ливена о равенстве дисперсий: статистика == {format_p(levene_stat)}, p-значение == {format_p(p_levene)}")
    if p_levene < ALPHA:
        test_name = "t-критерий Уэлча (дисперсии различаются)"
        equal_var = False
    else:
        test_name = "t-критерий Стьюдента (дисперсии равны)"
        equal_var = True
    print(f"Выбран {test_name}")

    t_stat, t_p = stats.ttest_ind(male_sample, female_sample, equal_var=equal_var)
    print("Статистика Т-теста:", format_p(t_stat))
    print("p-значение:", format_p(t_p))
    if t_p < ALPHA:
        print("Вывод: p-значение меньше 0.05, гипотеза h0 отвергается => гендерные различия статистически значимы")
    else:
        print("Вывод: p-значение не меньше 0.05, гипотеза h0 не отвергается => значимых гендерных различий нет")
    print()
