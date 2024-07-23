# %%
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import truncnorm
from numba import jit, njit
from shapely import MultiPoint, Point
from shapely.ops import nearest_points

# %% [markdown]
# Read dwelling types

# %%
dwellingTypes = pd.read_csv('./Data/DwellingTypesByZone2006.csv')

# %% [markdown]
# Read Zonal Profile data

# %%
profileData = pd.read_csv('./Data/GTAProfileData.csv')

# %% [markdown]
# Create ZoneSystem file which contains the zones to be used in the sim

# %%
zoneSystem = pd.DataFrame(data=profileData['DAUID'], columns=['DAUID'], dtype=str)
zoneSystem = zoneSystem[zoneSystem['DAUID'].str.len()==8]

# %% [markdown]
# Read Shapefile Zones

# %%
zoneShape:gpd.GeoDataFrame = gpd.read_file('./Data/TorontoZoneShape/TorontoZones_26917.shp')
zoneShape = zoneShape[zoneShape['DAUID'].astype(str).isin(zoneSystem['DAUID'])]

# %% [markdown]
# Create Zone Dataframe that holds all data

# %% [markdown]
# zoneData = pd.DataFrame(columns=['DAUID', 'Population', 'Employment rate', 'Area', "Total - Structural type of dwelling"])

# %% [markdown]
# for i, zone in zoneSystem.iterrows():
#     numDwellings=200
#     population = 400
#     try:numDwellings = dwellingTypes.loc[dwellingTypes['Geography'].astype(str)==zone['DAUID'], 'Total - Structural type of dwelling'].iloc[0]
#     except:pass
#     try:population = int(profileData.loc[profileData['DAUID'].astype(str)==zone['DAUID'], "Population"].iloc[0])
#     except:pass
# 
#     zoneData.loc[len(zoneData)] = [zone['DAUID'], population, profileData.loc[profileData['DAUID'].astype(str)==zone['DAUID'], "Employment rate"].iloc[0], float(zoneShape.loc[zoneShape['DAUID'].astype(str)==zone['DAUID'], 'geometry'].iloc[0].area), numDwellings]

# %% [markdown]
# zoneData.to_csv('./Data/combinedZoneData.csv', index=False)

# %% [markdown]
# <span style="font-size:5em;">CREATE PERSONS</span>

# %%
zoneData = pd.read_csv('./Data/combinedZoneData.csv')

# %% [markdown]
# Income generator

# %%
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# %%
salary_distribution_employed = get_truncated_normal(mean=70000, sd=40000, low=30000, upp=200000)
salary_distribution_retired = get_truncated_normal(mean=40000, sd=20000, low=15000, upp=70000)

# %%
totalPersons = zoneData['Population'].astype(float).sum()

# %%
employment_incomes = salary_distribution_employed.rvs(int(totalPersons*.8))
retired_incomes = salary_distribution_retired.rvs(int(totalPersons*.4))

# %% [markdown]
# <h3>synthesize persons<h3/>

# %% [markdown]
# initial person counts per zone

# %%
personZoneCounts = np.empty([0,2], dtype=np.int64)
for i, zone in zoneData.iterrows():
    personZoneCounts = np.r_[personZoneCounts, [[np.float64(zone['DAUID']), np.float64(zone['Population'])]]]

# %%
personZoneCounts

# %% [markdown]
# Create numba function to synthesize persons

# %%
@njit
def createPersons(personCounts):
    print("HEYYY")
    p = np.empty((0,10))
    for zoneNum in range(personCounts.shape[0]):
        print(zoneNum)
        numPeopleInZone = personCounts[zoneNum][1]
        for person in range(numPeopleInZone):
            print(person)
            age = np.random.randint(0,100)
            gender = np.random.randint(0,2)
            # 0:SINGLE 1:MARRIED 2:CHILD
            relationship = 2 if age < 18 else np.random.randint(0, 2)
            occupation = 0 if age < 4 \
                    else 4 if age > 65 \
                    else 3 if age < 22 \
                    else 1 if np.random.random() < .70 \
                    else 2
            occupation_type = 1
            workplace = -1
            income=-1
            if occupation==1:
                income = employment_incomes[np.random.randint(0,int(totalPersons*.9))]
            elif occupation==4:
                income = retired_incomes[np.random.randint(0,int(totalPersons*.1))]
            print('after income')
            
            schoolplace=-1
            
            p = np.append(p, np.array([[personZoneCounts[zoneNum][0], -1, age, gender, relationship, occupation, occupation_type, workplace, income, schoolplace]]), axis=0)
        print(zoneNum)
    return p
    

# %% [markdown]
# Create dataframe of persons using the c-compiled function above to create the data

# %%
persons = pd.DataFrame(data=createPersons(personZoneCounts), columns=['id', 'hhId', 'age', 'gender', 'relationship', 'occupation', 'occupation_type', 'workplace', 'income', 'schoolplace'])

# %% [markdown]
# Clean up varaibles

# %%
# del employment_incomes
# del retired_incomes
# del personZoneCounts
# del profileData
# del dwellingTypes


