import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import norm
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
#%% ucitavanje baze u DataFrame
df = pd.read_csv("ChengduPM20100101_20151231.csv")
#%% provera kako izgleda prvih nekoliko vrsta u bazi
print(df.head())
#%% koliko ima obelezja (kolona) i uzoraka (vrsta)
print("shape: \n", df.shape)
# imamo 52584 uzoraka i 17 obelezja
#%% sta predstavlja jedan uzorak baze
# svaki uzorak predstavlja koncentraciju PM2.5 cestica i vrednosti
# drugih vremenskih obelezja za svaki sat
#%% izbaciti obelezja merenja koncentracije PM cestica osim US Post
df.drop(['No', 'PM_Caotangsi', 'PM_Shahepu'], inplace=True, axis=1)
# izbacena su trazena obelezja kao i No zato sto nije potrebno za analizu
#%% Nedostajuci podaci
NANs = df.isnull().sum()
print("\n Nedostajuće vrednosti obeležja: \n", NANs)
# postoje nedostajuce vrednosti za obelezja:
# PM US Post, DEWP, HUMI, PRES, TEMP, cbwd, Iws, precipitation, Iprec
udeo = df.isnull().sum()/len(df)*100
print("\n Procenat nedostajućih vrednosti obeležja: \n", udeo)
# procentualno obelezja koja imaju nedostajuce vrednosti od ukupno vrsta:
# PM US Post (45%), DEWP (1%), HUMI (1%), PRES (0.99%), TEMP (1%),
# cbwd (0.99%), Iws (1%), precipitation (5.6%), Iprec (5.6%)
#%% brisanje redova gde su vrednosti null <=1%
df_dewp=df.loc[df['DEWP'].isnull()].index
df.drop(df_dewp, inplace= True, axis = 0)
df_humi=df.loc[df['HUMI'].isnull()].index
df.drop(df_humi, inplace= True, axis = 0)
df_pres=df.loc[df['PRES'].isnull()].index
df.drop(df_pres, inplace= True, axis = 0) 
df_temp=df.loc[df['TEMP'].isnull()].index
df.drop(df_temp, inplace= True, axis = 0)
df_cbwd=df.loc[df['cbwd'].isnull()].index
df.drop(df_cbwd, inplace= True, axis = 0)
df_iws=df.loc[df['Iws'].isnull()].index
df.drop(df_iws, inplace= True, axis = 0)

# prvo trazila indeks pa brisala red sa tim indeksom, a sa drop na bi sam nasao indekse i obrisao
#%% popunjavanje sa prethodnom vrednoscu gde su vrednosti null = 5,6%
df['precipitation'].fillna(method='ffill', inplace=True)
df['Iprec'].fillna(method='ffill', inplace=True)

# fillna dodje do prve null vrednosti, skonta da je null i on se vrati jedan u nazad i prepise vrednost tog obelezja u taj nedostajuci
# popunjavala sa ffill jer su podaci iz sata u sat, jer se nece vreme menjati iz sata u sat sa velikom razlikom
#%% popunjavanje sa medijanom gde su vrednosti null = 45%
df['PM_US Post'].fillna(df['PM_US Post'].median(), inplace=True)

# srednjom vrednoscu popunjava sva obelezja, nedostaju za celu 2010 god, posto su iz god u god rez slicni pa ce zato biti slicni podaci kao od 2011
#%% Analiza nakon korekcije nedostajucih podataka
NANs = df.isnull().sum()
print("\n Nedostajuće vrednosti obeležja nakon korekcija: \n", NANs)
# primetimo da smo resili sve nedostajuce vrednosti
print("shape: \n", df.shape)
# imamo 52037 uzoraka i 14 obelezja
#%% kojim obelezjima raspolazemo, koja obelezja su numericka i kategoricka
tipovi_obelezja=df.dtypes
# kategoricka obelezja su: No, year,month, day, hour, season, cbwd
# numericka obelezja su: PM Caotangsi, PM Shahepu, PM US Post, DEWP,
# HUAMI, PRES, TEMP, Iws, precipitation, Iprec
#%% Dodela numerickih vrednosti za pravac vetra
# kako je 'cbwd' jedino kategoricko obelezje koje nema numericke
# vrednosti, moramo mu ih dodeliti radi lakse analize
print('Pravac vetra (cbwd): \n', df['cbwd'].unique())
# sada smo dobili sve vrednosti koje pravac vetra ima u data setu
# predstavicu vrednosti preko stepena ugla gde ce cv kao neodredjeni
# pravac biti 360 stepeni, severo-istok 45 stepeni, jugo-istok 45
# stepeni, severo-zapad 315 stepeni i jugo-zapad 225 stepeni

df['cbwd']=df['cbwd'].replace('cv',360)
df['cbwd']=df['cbwd'].replace('NE',45)
df['cbwd']=df['cbwd'].replace('SE',135)
df['cbwd']=df['cbwd'].replace('NW',315)
df['cbwd']=df['cbwd'].replace('SW',225)
df.head()

# ima metoda koja bi podeli na 5 kolona cv,ne... i 0 gde nema tu vrednost a 1 gde je ima
#%% Dodela kategorickih vrednosti za godisnja doba
# godisnja doba su predstavljena po vrednostima 1,2,3,4 te radi lakse
# predstave preimenovacemo ih po vrednostima prolece, leto, jesen, zima

zima=df[df["season"]==4]
jesen=df[df["season"]==3]
leto=df[df["season"]==2]
prolece=df[df["season"]==1]
#%% da li postoje nelogicne i nevalidne vrednosti
# nakon svih izmena data seta mozemo sada da prikazemo describe tabelu
describe_tabela=df.describe()
# maksimalna vrednost cestica se mnogo istice u onosu na medijanu te
# zakljucujemo da je doslo do greske pri unosu vrednosti u data set

# vidimo da se medijna i srednja vrednost kod koncentracije cestica
# razlikuju, zakljucujemo da postoje autlajeri sa visokim vrednostima
# koji uticu na srednju vrednost i povecavaju je

# primeticemo da obelezja year, month, day, hour i season nemaju
# nelogicne vrednosti jer su intervali opsega isti kao u realnom svetu

# posto je srednja vrednost (mean) veca od medijane (50%) vidi se da postoje autlajeri
# autlajeri su vrednosti koje puno odskacu od ostalih vrednosti
#%% Analiza koncentracije PM2.5 cestica
plt.boxplot([df['PM_US Post']]) 
plt.ylabel('PM2.5')
plt.grid()
# sa ove slike ne mogu se uociti konkretne vrednosti interkvartilnog
# opsega te cemo morati da ispisemo describe za koncentraciju PM2.5
kurtosis= df['PM_US Post'].kurtosis(axis = 0) 
skewnes=df['PM_US Post'].skew(axis = 0) 
print("Koeficijent spljostenosti:" ,kurtosis)
print("Koeficijent asimetrije:" ,skewnes)
describe_cestica=df['PM_US Post'].describe()
print(describe_cestica)
plt.grid()
# sada primetimo da je interkvartilni opseg izmedju 64ug/m3 i 73ug/m3

# takodje uocavamo da vecina autlajera nije mnogo udaljena ali vidimo
# da se pojavljuje i vrednost od 688g/m3

# kako imamo autlajere i za niske vrednosti zakljucujemo da ovo obelezje
# ima levu asimetricnu raspodelu

# 25% uzoraka ima vrednost <=64, isto za 75%
#%% Analiza koncentracije PM2.5 cestica za svako godisnje doba
plt.figure()
plt.boxplot([zima['PM_US Post'], prolece['PM_US Post'], leto['PM_US Post'], jesen['PM_US Post']])
plt.xlabel('Godisnje doba')
plt.ylabel('Koncentracija PM2.5')
plt.xticks([1,2,3,4], ["Zima", "Prolece", "Leto", "Jesen"])
plt.grid()
print("Describe PM2.5 tokom zime: \n", zima['PM_US Post'].describe())
print("\n Describe PM2.5 tokom proleca: \n", prolece['PM_US Post'].describe())
print("\n Describe PM2.5 tokom leta: \n", leto['PM_US Post'].describe())
print("\n Describe PM2.5 tokom jeseni: \n", jesen['PM_US Post'].describe())
plt.grid()
# zaključci koje mozemo izvesti iz ovog prikaza su:
# koncentracija cestica tokom zime ima interkvartilni opseg izmedju 69ug/m3 i 122ug/m3
# koncentracija cestica tokom proleca ima interkvartilni opseg izmedju 69ug/m3 i 69ug/m3
# koncentracija cestica tokom leta ima interkvartilni opseg izmedju 47ug/m3 i 69ug/m3
# koncentracija cestica tokom jeseni ima interkvartilni opseg izmedju 55ug/m3 i 74ug/m3

# primetimo da 50% uzoraka ima istu vrednost koncentracije cestica u prolece
# minimalne vrednosti kolicine cestica su tokom leta i jeseni a maksimalne vrednosti su tokom zime
# zakljucujemo da je vazduh najzagadjeniji tokom zime
#%% Oscilacija temperature tokom godisnjih doba
plt.figure()
plt.boxplot([zima['TEMP'], prolece['TEMP'], leto['TEMP'], jesen['TEMP']])
plt.xlabel('Godisnje doba')
plt.ylabel('Temperatura')
plt.xticks([1,2,3,4], ["Zima", "Prolece", "Leto", "Jesen"])
plt.grid()
print("Describe temperature tokom zime: \n", zima['TEMP'].describe())
print("\n Describe temperature tokom proleca: \n", prolece['TEMP'].describe())
print("\n Describe temperature tokom leta: \n", leto['TEMP'].describe())
print("\n Describe temperature tokom jeseni: \n", jesen['TEMP'].describe())
plt.grid()
# zaključci koje mozemo izvesti iz ovog prikaza su:
# temperatura tokom zime ima interakvartilni opseg izmedju 6ug/m3 i 10ug/m3
# temperatura tokom proleca ima interakvartilni opseg izmedju 1ug/m3 i 22ug/m3
# temperatura tokom leta ima interakvartilni opseg izmedju 23ug/m3 i 28ug/m3
# temperatura tokom jeseni ima interakvartilni opseg izmedju 15ug/m3 i 22ug/m3

# tokomm zime imamo najmanju temperaturu -3 sto je i normalno kao i maksimalna temperature 35 tokom leta
#%% Najcesci pravac vetra tokom svakog dana
pravac_vetra_day=df.groupby(["cbwd"])["day"].count().reset_index()
plt.figure()
plt.plot(pravac_vetra_day["cbwd"], pravac_vetra_day["day"])
plt.title("Pravac duvanja vetra po danima")
plt.xlabel("Pravac duvanja vetra")
plt.ylabel("Broj dana")
# tokom dana vetar najcesce duva u neodredjenom pravcu a najredje u pravcu jugo-istok (SE)
#%% Najcesci pravac vetra za svako godisnje doba
pravac_vetra_season=df.groupby(["season", "cbwd"])["day"].count().reset_index()
plt.figure()
plt.plot(pravac_vetra_season["cbwd"], pravac_vetra_season["season"])
plt.title("Pravac duvanja vetra za svako godisnje doba")
plt.xlabel("Pravac duvanja vetra")
plt.ylabel("Godisnja doba")
plt.yticks([1,2,3,4], ["Zima", "Prolece", "Leto", "Jesen"])

# greskom ostalo
#%%
print("Koeficijenti spljostenosti obelezja:")
for i in df[['PM_US Post','DEWP','HUMI','PRES','TEMP','Iws','precipitation']].columns:
    kurtosis=df[i].kurtosis(axis = 0) 
    print(i , ":" , kurtosis)
print("\n")

print("Koeficijenti asimetrije obelezja:")
for i in df[['PM_US Post','DEWP','HUMI','PRES','TEMP','Iws','precipitation',]].columns:
    skewnes=df[i].skew(axis = 0) 
    print(i , ":" , skewnes)
    
# da vidim koliko se svako obelezje razlikuje od normalne raspodele, da vidim da li je asimetricna
#%%
fig, ax = plt.subplots(1, 3, figsize=[20,5])
fig.subplots_adjust(hspace=0.5, wspace=0.15)
sb.distplot(df['PM_US Post'], fit=norm,ax=ax[0])
sb.distplot(df['Iws'], fit=norm,ax=ax[1])
sb.distplot(df['precipitation'], fit=norm,ax=ax[2])
# primetimo da od normalne raspodele najvise odstupaju obelezja PM_US 
# Post, Iws i precipitation
# ova tri obelezja imaju autlajere koji se razlikuju od najcesce pojavljivanih vrednosti
# kako je koenficijent spljostenosti pozitivan za obelezja koja smo
# prikazali na ova tri grafika te zakljucujemo da oni imaju puno uzoraka
# sa slicnim vrednostima u malom opsegu

# htela sam da potvrdim graficki to sto su mi koef pre pokazali da puno odstupa od normalne raspodele
#%% Analiza koncentracije cestica tokom godina
df['year'].unique()
df_2010=df[df['year']==2010]
df_2011=df[df['year']==2011]
df_2012=df[df['year']==2012]
df_2013=df[df['year']==2013]
df_2014=df[df['year']==2014]
df_2015=df[df['year']==2015]
fig, [(ax0, ax1), (ax2, ax3), (ax4, ax5)] = plt.subplots(3, 2, figsize=[10,7])
fig.subplots_adjust(hspace=0.2, wspace=0.15)

ax0.hist(df_2010['PM_US Post'],bins=20,density = True,histtype ='barstacked',stacked = True,color='red',label = "2010")
ax1.hist(df_2011['PM_US Post'],bins=20,density = True,histtype ='barstacked',stacked = True,color='orange', label = "2011")
ax2.hist(df_2012['PM_US Post'],bins=20,density = True,histtype ='barstacked',stacked = True,color='blue', label = "2012")
ax3.hist(df_2013['PM_US Post'],bins=20,density = True,histtype ='barstacked',stacked = True,color='skyblue', label = "2013")
ax4.hist(df_2014['PM_US Post'],bins=20,density = True,histtype ='barstacked',stacked = True,color='purple', label = "2014")
ax5.hist(df_2015['PM_US Post'],bins=20,density = True,histtype ='barstacked',stacked = True,color='pink', label = "2015")
fig.legend()
fig.suptitle("Raspodela koncentracije PM2.5 čestica po godinama")
# za 2010 i 2011 godinu koncentracija cestica je ista, nepromenjena
# uocavamo da je koncentracija cestica za 2012,2013,2014 i 2015
# veoma slicna, takodje uocavamo levu asimetricnu raspodelu

# pola 2012 god je falilo podataka te je medijaom popunjeno i to je ovaj najveci stubic a ovi manji su ustv podaci koji su bili u bazi
#%% Koncentracija PM2.5 cestica tokom godina
fig, [(ax0, ax1), (ax2, ax3), (ax4, ax5)] = plt.subplots(3, 2, figsize=[20,13])
fig.subplots_adjust(hspace=0.2, wspace=0.15)

ax0.boxplot(df_2010['PM_US Post'])
ax0.set_title('2010')
ax0.grid()

ax1.boxplot(df_2011['PM_US Post'])
ax1.set_title('2011')
ax1.grid()

ax2.boxplot(df_2012['PM_US Post'])
ax2.set_title('2012')
ax2.grid()

ax3.boxplot(df_2013['PM_US Post'])
ax3.set_title('2013')
ax3.grid()

ax4.boxplot(df_2014['PM_US Post'])
ax4.set_title('2014')
ax4.grid()

ax5.boxplot(df_2015['PM_US Post'])
ax5.set_title('2015')
ax5.grid()
#%%
IQR2010min=df_2010['PM_US Post'].quantile(0.25)
IQR2010max=df_2010['PM_US Post'].quantile(0.75)
IQR2011min=df_2011['PM_US Post'].quantile(0.25)
IQR2011max=df_2011['PM_US Post'].quantile(0.75)
IQR2012min=df_2012['PM_US Post'].quantile(0.25)
IQR2012max=df_2012['PM_US Post'].quantile(0.75)
IQR2013min=df_2013['PM_US Post'].quantile(0.25)
IQR2013max=df_2013['PM_US Post'].quantile(0.75)
IQR2014min=df_2014['PM_US Post'].quantile(0.25)
IQR2014max=df_2014['PM_US Post'].quantile(0.75)
IQR2015min=df_2015['PM_US Post'].quantile(0.25)
IQR2015max=df_2015['PM_US Post'].quantile(0.75)
print("IQR opseg za 2010.godinu:",IQR2010min,"-",IQR2010max,"ug/m3")
print("IQR opseg za 2011.godinu:",IQR2011min,"-",IQR2011max,"ug/m3")
print("IQR opseg za 2012.godinu:",IQR2012min,"-",IQR2012max,"ug/m3")
print("IQR opseg za 2013.godinu:",IQR2013min,"-",IQR2013max,"ug/m3")
print("IQR opseg za 2014.godinu:",IQR2014min,"-",IQR2014max,"ug/m3")
print("IQR opseg za 2015.godinu:",IQR2015min,"-",IQR2015max,"ug/m3")

# autlajeri se tokom 2012,2013,2014 i 2015 godina ne menjaju u velikoj meri
#%% Koncentracija PM2.5 cestica tokom godina za svaki mesec
df2010=df_2010[['PM_US Post','month']].groupby("month").median()
df2011=df_2011[['PM_US Post','month']].groupby("month").median()
df2012=df_2012[['PM_US Post','month']].groupby("month").median()
df2013=df_2013[['PM_US Post','month']].groupby("month").median()
df2014=df_2014[['PM_US Post','month']].groupby("month").median()
df2015=df_2015[['PM_US Post','month']].groupby("month").median()
plt.plot(df2010, 'red', label='2010')
plt.plot(df2011, 'orange', label='2011')
plt.plot(df2012, 'blue', label='2012')
plt.plot(df2013, 'green', label='2013')
plt.plot(df2014, 'purple', label='2014')
plt.plot(df2015, 'pink', label='2015')
plt.ylabel('Koncentracija PM2.5')
plt.xlabel('Meseci u godini')
plt.legend()
# primetimo da je 2010 i 2011 kao i u prethodnoj analizi koncentracija
# cestica ista
# primetimo takodje da je koncentracija cestica na pocetku i kraju godine
# (zimi) najveca, sto smo takodje vec pokazali
# uocavamo i da se tokom godina koncentracija cestica smanjuje
#%% Zavisnost promene PM2.5 u odnosu na ostala obelezja u bazi
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].scatter(df['DEWP'],df['PM_US Post'], color='purple')
axs[0, 0].set_title('Odnos DEWP-PM2.5')
axs[0, 1].scatter(df['HUMI'],df['PM_US Post'], color='pink')
axs[0, 1].set_title('Odnos HUMI-PM2.5')
axs[1, 0].scatter(df['PRES'],df['PM_US Post'], color='yellow')
axs[1, 0].set_title('Odnos PRES-PM2.5')
axs[1, 1].scatter(df['Iws'],df['PM_US Post'], color='skyblue')
axs[1, 1].set_title('Odnos Iws-PM2.5')

# uocavamo da se koncentracija cestica ponasa slicno u odnosu sa ova dva
# obelezja: DEWP i PRESS
# koncentracija cestica u odnosu sa HUMI nam govori da koncentracija 
# raste sa porastom vrednosti vlasnosti vazduha
# dok u odnosu sa Iws primetimo da je koncentracija cestica veca dok je brzina vetra mala

# korelacija cestica sa drugim obelezjima, jer se trazilo
# ako se poveca pritisak ne znamo da li ce se br cestica povecati ili smanjiti jer nije ni pozitivno ni negativno korelisano (ljubicasto)
# plavo je negativno korelisano tj povecava se jedno a drugo smanjuje
#%% Korelacija izmedju svih obelezja
corr = df[['PM_US Post','DEWP','HUMI','PRES','TEMP','Iws','precipitation','Iprec']].corr()
f = plt.figure(figsize=(12, 9))
sb.heatmap(corr.abs(), annot=True);
# vidimo da je temperatura najvise korelisana sa koncentracijom PM2.5 cestica
# (0.28) pa zatim DEWP (0.22)
#%% Medjukorelacija ostalih obelezja
sns.set()
sns.pairplot(df[['DEWP','HUMI','PRES','TEMP','cbwd','Iws','precipitation','Iprec']], height = 2.5)
plt.show();

# isto prikazano kao u tabeli samo preko tackica
#%%
fig, ax = plt.subplots(1, 3, figsize=[15,5])
fig.subplots_adjust(hspace=0.2, wspace=0.15)

ax[0].scatter(df['TEMP'], df['DEWP'], color='yellow', label="TEMP-DEWP")
ax[0].set_title('TEMP-DEWP')
ax[1].scatter(df['TEMP'], df['PRES'], color='orange', label="TEMP-PRES")
ax[1].set_title('TEMP-PRES')
ax[2].scatter(df['PRES'], df['DEWP'], color='pink', label="PRES-DEWP")
ax[2].set_title('PRES-DEWP')

# TEMP i DEWP su pozitivno korelisana obelezja (povecanjem jedne vrednosti
# povecava se i druga)
# TEMP i PRES su negativno korelisani (povecanjem temperature vazdusni
# pritisak se smanjuje)
# DEWP i PRES su isto negativno korelisani
# najmanju korelaciju sa koncentracijom cestica i ostalim obelezjima 
# ima obelezje percipitation

# f-ja subplot predstavlja vise slika u jednoj i 1,3 je u jednom redu tri kolone
#%%
df_rain=df[df['precipitation']!=0]
df_rainNo=df[df['precipitation']==0]
fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].hist(df_rain['PM_US Post'] , density=True, alpha=0.5, bins=30, label ='Kisan dan',color='green')
ax[0].hist(df_rainNo['PM_US Post'] , density=True, alpha=0.5, bins=30, label ='Nije kisan dan',color='orange')
ax[1].boxplot([df_rain['PM_US Post'],df_rainNo['PM_US Post']])
#plt.xticks([1, 2], ["Kisan dan", "Nije kisan dan"])
ax[0].set_xlabel('Koncentracija PM2.5')
ax[0].set_ylabel('Verovatnoca')
ax[0].legend()
print("Describe za dane kada ima kise: \n", df_rain['PM_US Post'].describe())
print("\n Describe za dane kada nema kise: \n", df_rainNo['PM_US Post'].describe())
# uocavamo da je veca verovatnoca da ce biti izmerena velika koncentracija
# cestica kada nema kise na sta nam ukazuje veliki broj i raspon autlajera

# alpha je providnost a bins je br stubica, xticks nije ni bilo potrebno jer je vec napisano u label
#%% Linearna regresija 
# izbacivanje obilježja PM2.5 iz skupa za obuku i ubacivanje u skup za testiranje
x=df.drop(['PM_US Post'], axis=1)
y=df['PM_US Post']

# iz skupa za treniranje izbacujem ono sto predvidjam i ubacujem u y
#%% funkcija koja racuna razlicite mere uspesnosti regresora
def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) 
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))
#%% podela skupa na trening i test podatke gdje je 10% uzoraka predvidjeno 
# za test skup a 90% za obuku modela
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=49)
# skaliranje podataka-standardizacija koje normalizuje obelezja tako da imaju
# strednju vrednost 0 i standardnu devijaciju 1
scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
x_train_std
# skaliranje se izvrsilo da bi se neka obelezja ne bi izdvajala po svojim 
# vrednostima od drugih i tako uticala na natprilagodjenje modela
#%% Hipoteza y=b0+b1x1+b2x2+...+bnxn
# osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)
#%% 
# Selekcija obelezja
import statsmodels.api as sm
X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()
model.summary()
print(model.summary())
# na osnovu p vrednosti koja je za TEMP 0.301 i za precipitation 0.187 
# sto je vece od 1%, ta obelezja se trebaju odbaciti, medjutim zakljucujemo
# da ova analiza nije potpuno tacna jer imamo visoku korelaciju koncentracije
# PM2.5 cestica sa temperaturom
#%% 
x.drop('TEMP',axis = 1,inplace = True)
x.drop('precipitation',axis = 1,inplace = True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#%% kako imamo novi skup podataka opet ih treba normalizovati i ponoviti postupak selekcije
scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
#%%
X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()
model.summary()
print(model.summary())
# kako p ima sve vrednosti nule, ne odbacujemo vise ni jedno obelezje
# i ponavljamo postupak obucavanja
#%%
first_regression_model = LinearRegression(fit_intercept=True, normalize=False)
first_regression_model.fit(x_train_std, y_train)
y_predicted = first_regression_model.predict(x_test_std)
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
print("koeficijenti: ", first_regression_model.coef_)
# zakljucak je da se greske nisu smanjile posle selekcije
# takodje primecujemo da ne postoji obelezje sa izrazenom tezinom
#%% Hipoteza y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...
# u hipotezu se ubacuju samo interakcije 
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)
regression_model_inter = LinearRegression(fit_intercept=True, normalize=True)
regression_model_inter.fit(x_inter_train, y_train)
y_predicted = regression_model_inter.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
print("koeficijenti: ", regression_model_inter.coef_)
#%%
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(x_inter_train, y_train)
y_predicted = lasso_model.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
print("koeficijenti: ", lasso_model.coef_)
#%%
ridge_model = Ridge(alpha=10)

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)
#%%
poly = PolynomialFeatures(degree=2,interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)
regression_model_inter = LinearRegression(fit_intercept=True, normalize=True)
regression_model_inter.fit(x_inter_train, y_train)
y_predicted = regression_model_inter.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
print("koeficijenti: ", regression_model_inter.coef_)
print(poly.get_feature_names())
#%%
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(x_inter_train, y_train)
y_predicted = lasso_model.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
print("koeficijenti: ", lasso_model.coef_)
#%%
poly = PolynomialFeatures(degree=3,interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)
regression_model_inter = LinearRegression(fit_intercept=True, normalize=True)
regression_model_inter.fit(x_inter_train, y_train)
y_predicted = regression_model_inter.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
print("koeficijenti: ", regression_model_inter.coef_)
#%%
lasso_model = Lasso(alpha=0.0001)
lasso_model.fit(x_inter_train, y_train)
y_predicted = lasso_model.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
print("koeficijenti: ", lasso_model.coef_)
#%%
poly = PolynomialFeatures(degree=4,interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)
regression_model_inter = LinearRegression(fit_intercept=True, normalize=True)
regression_model_inter.fit(x_inter_train, y_train)
y_predicted = regression_model_inter.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
print("koeficijenti: ", regression_model_inter.coef_)
#%%
lasso_model = Lasso(alpha=0.00001)
lasso_model.fit(x_inter_train, y_train)
y_predicted = lasso_model.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
print("koeficijenti: ", lasso_model.coef_)
#%%
poly = PolynomialFeatures(degree=5,interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)
regression_model_inter = LinearRegression(fit_intercept=True, normalize=True)
regression_model_inter.fit(x_inter_train, y_train)
y_predicted = regression_model_inter.predict(x_inter_test)
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
print("koeficijenti: ", regression_model_inter.coef_)