import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

path = ("./Duomenų aibė nr. 2")

texts = []
labels = []

for file in os.listdir(path):
    file_path = os.path.join(path, file)

    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

            texts.append(text)
            labels.append(file[0])

# print("Tekstų failų kiekis: ", len(texts))
# print("Kategorijos: ", labels)
# print("\nPirmas tekstas:\n", texts[0])

# Filtravimas

texts_lower = [text.lower() for text in texts]
# print("Prieš:", texts[0])
# print("Po: ", texts_lower[0])

texts_no_punctuation = [re.sub(r'[^\w\s]', '', text) for text in texts_lower] # ^ - ne, \w - raides, \s - tarpai
# print("Prieš:", texts_lower[3])
# print("Po: ", texts_no_punctuation[3])

text_no_short = [' '.join(word for word in text.split() if len(word) > 2) for text in texts_no_punctuation]
# print("Prieš:", texts_no_punctuation[0])
# print("Po: ", text_no_short[0])

text_no_numbers = [re.sub(r'\d+', '', text) for text in text_no_short] # \d - bet koks skaitmuo, + - vienas ar daugiau is eiles
# print("Prieš:", text_no_short[3])
# print("Po: ", text_no_numbers[3])

stopwords = {
    'ir', 'yra', 'kad', 'su', 'į', 'i', 'iš', 'is', 'ant', 'po', 'bei',
    'bet', 'tad', 'tai', 'tas', 'to', 'tuo', 'jis', 'ji', 'jos', 'jų', 'ju',
    'mes', 'jus', 'jūs', 'man', 'mane', 'mano', 'mūsų', 'musu',
    'ar', 'o', 'ne', 'bei', 'kaip', 'kai', 'kur', 'kas', 'ką', 'ka',
    'čia', 'cia', 'ten', 'dar', 'tik', 'pat', 'per', 'apie', 'tarp',
    'dėl', 'del', 'nuo', 'iki', 'už', 'uz'
}

text_no_stopwords = [
    ' '.join(word for word in text.split() if word not in stopwords)
    for text in text_no_numbers
]

# print("Prieš:", text_no_numbers[0])
# print("Po: ", text_no_stopwords[0])

def lt_stem(word):
    suffixes = [
        # ilgesnės veiksmažodžių / būdvardžių / daiktavardžių galūnės
        'iausiais', 'iausiam', 'iausiais', 'iausioms', 'iausiose',
        'iausias', 'iausioje', 'iausiai', 'iausių', 'iausio', 'iausiam',
        'esniais', 'esniaus', 'esnėmis', 'esniems', 'esnėse', 'esnėmis',
        'uojamas', 'uojanti', 'uojanč', 'avimas', 'avimą', 'avime',
        'inimas', 'inimą', 'inime', 'ėjimas', 'ėjimą', 'ėjime',
        'ybėmis', 'ybėse', 'ybėms', 'ybėja', 'ystėmis', 'ystėse',
        'iškumas', 'iškumo', 'iškumą', 'iškume',

        # daiktavardžiai / būdvardžiai
        'imuose', 'imuosi', 'imuose', 'imuosi',
        'ymuose', 'ymuosi', 'imuoju', 'ymuoju',
        'iuose', 'iuosi', 'uose', 'uose',
        'iams', 'iems', 'omis', 'omis', 'uose', 'iams',
        'ėjais', 'ėjoms', 'ėjose', 'ėjams',
        'iniais', 'inėms', 'iniuose', 'iniams',
        'iomis', 'iosioms', 'iuosius', 'iuosiu',
        'iajam', 'iajai', 'iasis', 'iosios', 'iuosius',

        # dažnos lietuvių galūnės
        'imas', 'imas', 'ymas', 'ymas',
        'inis', 'inė', 'iniai', 'ines', 'iniu', 'inių',
        'iška', 'iški', 'išką', 'iškų', 'iškas',
        'iukas', 'iuką', 'iukai', 'iukų',
        'ukas', 'uką', 'ukai', 'ukų',
        'elis', 'elio', 'eliui', 'elį', 'elių',
        'ėlis', 'ėlio', 'ėliui', 'ėlį', 'ėlių',
        'aitė', 'aitės', 'aitę', 'aitėms',
        'utė', 'utės', 'utę', 'utėms',
        'ytojas', 'ytojo', 'ytojui', 'ytoją', 'ytojų',
        'ėtojas', 'ėtojo', 'ėtojui', 'ėtoją', 'ėtojų',

        # linksniai / daugiskaita
        'uose', 'omis', 'iams', 'iems', 'oms', 'ais',
        'ose', 'yse', 'ėje', 'ame', 'ėje', 'oje',
        'ąją', 'ajai', 'ajam', 'osios', 'asis', 'oji',
        'ios', 'ius', 'iems', 'iams', 'omis', 'ysis',
        'usis', 'ysis', 'ioji', 'oji', 'ąsias', 'uosius',

        # veiksmažodžiai
        'avosi', 'davosi', 'iuosi', 'uosi', 'osi',
        'uoja', 'uoju', 'uojai', 'uojam', 'uojat',
        'avau', 'avai', 'avome', 'avote',
        'ėjau', 'ėjai', 'ėjome', 'ėjote',
        'inau', 'inai', 'inome', 'inote',
        'davo', 'dama', 'damas', 'dami', 'damos',
        'tume', 'tumėte', 'čiau', 'tum', 'čiau',
        'site', 'siu', 'siu', 'čiau', 'tų',
        'ame', 'ate', 'avo', 'ėjo', 'ina', 'ino',
        'yti', 'oti', 'ėti', 'auti', 'uoti', 'inti',
        'ant', 'ent', 'iant', 'uojant', 'inant',

        # trumpesnės, bet dar prasmingos
        'imo', 'imu', 'imą', 'yme', 'ymo', 'ymą',
        'iai', 'iam', 'ias', 'ius', 'ių', 'oms',
        'ose', 'ais', 'ams', 'ems', 'oje', 'ėje',
        'oje', 'oje', 'ais', 'uos', 'ios', 'oje',
        'tas', 'tas', 'tis', 'tis', 'tis',
        'yti', 'oti', 'ėti', 'uti', 'ti',

        # vienos raidės / labai trumpos galūnės – paliekam pabaigai
        'as', 'is', 'ys', 'us',
        'a', 'ą', 'e', 'ę', 'ė', 'i', 'į', 'y', 'o', 'u', 'ų', 's'
    ]

    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

text_stemmed = [
    ' '.join(lt_stem(word) for word in text.split())
    for text in text_no_stopwords
]

preprocessed_text = text_stemmed

# Feature extraction

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_text)

print("Matricos dydis: ", X.shape)
print("Žodžiai: ", vectorizer.get_feature_names_out())
print("Pirmas tekstas: ", X[0].toarray())

# Hierarchinis klasterizavimas

X_dense = X.toarray()

hierarchy = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchy_labels = hierarchy.fit_predict(X_dense)

print("Hierarchinis klasterizavimas:", hierarchy_labels)

# for i, label in enumerate(hierarchy_labels):
#     print(f"Tekstas {i+1} -> klasteris {label}")

linked = linkage(X_dense, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    linked,
    labels=[str(i) for i in range(1, len(preprocessed_text) + 1)],
    leaf_rotation=0,
    leaf_font_size=12,
)
plt.title("Hierarchinio klasterizavimo dendrograma - ward metodas")
plt.xlabel("Tekstai")
plt.ylabel("Atstumas")
plt.show()

# Klasterizavimas taikant k-vidurkių metodą

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

print("Klasterizavimas taikant k-vidurkių metodą:", kmeans_labels)

# for i, label in enumerate(kmeans_labels):
#     print(f"Tekstas {i+1} -> klasteris {label}")

# Panašumo mato skaičiavimas

similarity = cosine_similarity(X[5], X[16])[0][0]

print("Panašumas tarp 6 ir 17 dokumento:", similarity)
