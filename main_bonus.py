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


def gauss(sample, name):
    n = len(sample)
    print("Объём выборки:", n)
    print()

    mean = np.mean(sample)
    median = np.median(sample)
    print(f"Среднее значение {format_p(mean)}")
    print(f"Медиана {format_p(median)}")
    print()

    skewness = stats.skew(sample, bias=False)
    kurtosis = stats.kurtosis(sample, bias=False)
    print(f"Асимметрия {format_p(skewness)}")
    print(f"Эксцесс {format_p(kurtosis)}")
    print()

    print("Гипотезы:")
    print("h0 - данные распределены нормально")
    print("h1 - данные не подчиняются нормальному распределению")
    print()

    # группируем данные в равновероятные интервалы и сравниваем наблюдаемые частоты с ожидаемыми для нормального распределения
    std = np.std(sample, ddof=1)
    bin_count = int(np.sqrt(n))
    if bin_count < 5:
        bin_count = 5
    quantiles = np.linspace(1.0 / bin_count, 1.0 - 1.0 / bin_count, bin_count - 1)
    edges = np.concatenate(([np.min(sample)], stats.norm.ppf(quantiles, mean, std), [np.max(sample)]))
    edges = np.unique(np.sort(edges))
    observed, _ = np.histogram(sample, bins=edges)
    # ожидаемые частоты: для крайних интервалов учитываем хвосты нормального распределения
    cdf_values = stats.norm.cdf(edges, mean, std)
    expected = np.empty(len(observed))
    expected[0] = cdf_values[1]
    expected[1:-1] = cdf_values[2:-1] - cdf_values[1:-2]
    expected[-1] = 1.0 - cdf_values[-2]
    expected = n * expected
    chi2_stat, p_value = stats.chisquare(observed, expected)
    print("Критерий хи-квадрат пирсона:")
    print(f"Статистика хи-квадрат {format_p(chi2_stat)}")
    print(f"p-значение {format_p(p_value)}")
    if p_value < ALPHA:
        print("Вывод: p-значение меньше 0.05, гипотеза h0 отвергается - данные не распределены нормально")
    else:
        print("Вывод: p-значение не меньше 0.05, гипотеза h0 не отвергается - данные распределены нормально")
    print()

    plt.figure(figsize=(9, 5))
    plt.hist(sample, bins="auto", density=True, alpha=0.6, color="steelblue", edgecolor="black")
    xs = np.linspace(np.min(sample), np.max(sample), 300)
    kde = stats.gaussian_kde(sample)
    plt.plot(xs, kde(xs), color="red", label="KDE")
    plt.plot(xs, stats.norm.pdf(xs, mean, std), color="green", label="Нормальная кривая")
    plt.title(f"Проверка на нормальность: {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print()

    return chi2_stat, p_value


iris = pd.read_excel("Iris.xlsx", sheet_name="Data")
print(f"Размер набора данных: {iris.shape}")
print("Первые строки таблицы:")
print(iris.head())
print(f"Названия столбцов {list(iris.columns)}")
print()

features = iris.drop(columns=["Species_No", "Species_name"]).copy()
print("Удалены столбцы с номером сорта и названием сорта")
print("Оставшиеся столбцы: ")
print(list(features.columns))
print()

rename_map = {
    "Petal_width": "Ширина лепестка",
    "Petal_length": "Длина лепестка",
    "Sepal_width": "Ширина чашелистика",
    "Sepal_length": "Длина чашелистика",
}
features = features.rename(columns=rename_map)
print("Новые названия столбцов: ")
print(list(features.columns))
print()

features.index = range(1, 151)

print("Проверка всех показателей цветков на нормальность: ")
p_values = {}
for col in features.columns:
    print("=" * 70)
    print("Показатель:", col)
    chi2_stat, p_value = gauss(features[col].to_numpy(), col)
    p_values[col] = p_value
print()

worst_indicator = min(p_values, key=p_values.get)
print("Проверка самого ненормального показателя отдельно для каждого сорта")
print(f"Минимальное p-значение у показателя '{worst_indicator}' == {format_p(p_values[worst_indicator])}")
print("Проверка на нормальность отдельно для каждого сорта: ")
for start in range(0, 150, 50):
    variety_name = iris["Species_name"].iloc[start]
    variety_sample = features.iloc[start:start + 50][worst_indicator].to_numpy()
    print(f"Сорт: {variety_name}")
    gauss(variety_sample, f"сорт {str(variety_name)}")
print()

print("Проверка нормальности для мелких и крупных цветков: ")
small_sample = features.iloc[0:50][worst_indicator].to_numpy()
large_sample = pd.concat([features.iloc[50:100], features.iloc[100:150]])[worst_indicator].to_numpy()
print("Мелкие цветки: ")
gauss(small_sample, "мелкие цветки (setosa)")
print("Крупные цветки: ")
gauss(large_sample, "крупные цветки (два других сорта)")
print()

# по отдельности у каждого сорта ирисов показатели распределяются нормально 
# при объединении разных сортов в одну выборку нормальность нарушается, так как смешиваются несколько различных генеральных совокупностей с разными средними значениями
