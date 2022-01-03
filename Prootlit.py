# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:42:47 2022

@author: foulo
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings
from IPython.display import Markdown, display
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_theme({'legend.frameon':True})

#weatherAUS_csv = Path(__file__).parents[1]/'C:/Users/foulo/Desktop/weatherAUS.csv'


## MENU DE NAVIGATION ##

nav = st.sidebar.radio("Navigation", ["Accueil",
                                "Description du dataset",
                                "Les filouteries de Lise aka Kangooroo-Girl",
                                "🐰"])
if nav == "Accueil":
    "G'day mate ! ☀☔⚡🌡"

elif nav == "Description du dataset":

    st.title("DESCRIPTION DES VARIABLES 🦘")
    
    df = pd.read_csv("weatherAUS.csv")
    
    #Création d'un df de secours pour avoir une trace des données brutes.
    
    df_saved = pd.read_csv("weatherAUS.csv")
    
    pd.set_option("display.max_columns", None)
    
    st.markdown("<h4><u><font color = 'navy'>Affichage des premières lignes du dataframe.</h4></u>", unsafe_allow_html=True)
    st.write(df.head())
    st.write("\n------------------------------------------------------------------------")
    st.markdown("<h4><u><font color = 'navy'>Description statistique du dataframe.</h4></u>", unsafe_allow_html=True)
    st.write(round(df.describe(), 1))
    
    
    
    #Création de Year, Month & Day par slicing de la valeur de Date
    df["Year"] = df["Date"].apply(lambda x : int(x[:4]))
    df["Month"] = df["Date"].apply(lambda x : int(x[5:7]))
    df["Day"] = df["Date"].apply(lambda x : int(x[8:]))
    
    #Création du dictionnaire qui contient pour chaque état australien, les stations associées
    localites = {"SA" : ["Adelaide", "MountGambier", "Woomera", "Nuriootpa"],
                "WA" : ["Perth", "Albany", "PearceRAAF", "PerthAirport", "Walpole", "SalmonGums", "Witchcliffe"],
                "NSW" : ["Canberra", "Sydney", "Albury", "Wollongong", "MountGinini", "Tuggeranong", "Penrith", "Newcastle", "Cobar", "SydneyAirport", "BadgerysCreek", "WaggaWagga", "Moree", "Williamtown", "CoffsHarbour", "NorahHead", "Richmond"],
                "QLD" : ["Brisbane", "Townsville", "Cairns", "GoldCoast"],
                "TAS" : ["Hobart", "Launceston"],
                "VIC" : ["Melbourne", "Bendigo", "Ballarat", "Dartmoor", "Portland", "Mildura", "MelbourneAirport", "Sale", "Watsonia", "Nhil"],
                "NT" : ["Darwin", "AliceSprings", "Katherine", "Uluru"],
                "NI" : ["NorfolkIsland"]}
    
    #Création de la colonne State
    df["State"] = df.Location
    df["State"] = df["State"].replace(to_replace = ["Adelaide", "MountGambier", "Woomera", "Nuriootpa"], value = "SA")
    df["State"] = df["State"].replace(to_replace = ["Perth", "Albany", "PearceRAAF", "PerthAirport", "Walpole", "SalmonGums", "Witchcliffe"], value = "WA")
    df["State"] = df["State"].replace(to_replace = ["Canberra", "Sydney", "Albury", "Wollongong", "MountGinini", "Tuggeranong", "Penrith", "Newcastle", "Cobar", "SydneyAirport", "BadgerysCreek", "WaggaWagga", "Moree", "Williamtown", "CoffsHarbour", "NorahHead", "Richmond"], value = "NSW")
    df["State"] = df["State"].replace(to_replace = ["Brisbane", "Townsville", "Cairns", "GoldCoast"], value = "QLD")
    df["State"] = df["State"].replace(to_replace = ["Hobart", "Launceston"], value = "TAS")
    df["State"] = df["State"].replace(to_replace = ["Melbourne", "Bendigo", "Ballarat", "Dartmoor", "Portland", "Mildura", "MelbourneAirport", "Sale", "Watsonia", "Nhil"], value = "VIC")
    df["State"] = df["State"].replace(to_replace = ["Darwin", "AliceSprings", "Katherine", "Uluru"], value = "NT")
    df["State"] = df["State"].replace(to_replace = ["NorfolkIsland"], value = "NI")
    
    ##MENU DÉROULANT - CHOIX DES VARIABLES A AFFICHER
    
    st.markdown("<h4><u><font color = 'navy'>Exploration des variables.</h4></u>", unsafe_allow_html=True)
    
    var_a_afficher = st.selectbox(label = "Choisissez les variables à explorer dans le menu ci-dessous 🦘",
                                  options = ["Variables",
                                             "Date & lieu de relevé", 
                                             "Températures", 
                                             "Précipitations",
                                             "Ensoleillement, ennuagement",
                                             "Direction du vent",
                                             "Force du vent",
                                             "Pression atmosphérique", 
                                             "Pluie"])
    
    if var_a_afficher == "Date & lieu de relevé":
        st.markdown("<h2><u><center>Date, Location</center></u></h2>",unsafe_allow_html=True)
    
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<ul><li><b>Date</b> est définie comme <i>'The date of observation'</i>, c'est-à-dire la date de chaque relevé de mesure.</li><li><b>Location</b> est définie comme <i>'The common name of the location of the weather station'</i>, c'est-à-dire le nom de chaque lieu de relevé de mesure.</li></ul>",unsafe_allow_html=True)
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<h5><i>Date</i></h5>",unsafe_allow_html=True) 
        st.markdown("Les dates des relevés sont au format <b>yyyy-mm-dd</b>, les relevés vont du <b>{}</b> au <b>{}</b>. On constate qu'il n'y que <b>{}</b> relevés pour l'année <b>2007</b>, <b>{}</b> pour l'année <b>2008</b> et <b>{}</b> pour l'année <b>2017</b>. Pour les autres années, le nombre moyen de relevés est de <b>{}</b>.".format(
            df.Date.min(),
            df.Date.max(),
            df.Year[df["Year"] == 2007].count(),
            df.Year[df["Year"] == 2008].count(),
            df.Year[df["Year"] == 2017].count(),
            int(df.Year[(df["Year"] != 2007) & (df["Year"] != 2008) & (df["Year"] != 2017)].groupby(df.Year).count().mean())),unsafe_allow_html=True)
        
        st.markdown("<h5><i>Location</i></h5>",unsafe_allow_html=True)
        st.markdown("Le nombre médian de relevés par lieu est de <b>{}</b>. Il y'a au total <b>{}</b> lieux de relevé différents. Les lieux avec le plus de relevés sont <b>{}</b> ({} relevés) et <b>{}</b> ({} relevés). Trois lieux fournissent moins de relevés que les autres : <b>{}</b>, <b>{}</b> et <b>{}</b> ({} relevés chacun).".format(int(df.Location.value_counts().median()), len(set(df.Location)), df.Location.value_counts().keys()[0], df.Location.value_counts()[0], df.Location.value_counts().keys()[1], df.Location.value_counts()[1], df.Location.value_counts().keys()[-1], df.Location.value_counts().keys()[-2], df.Location.value_counts().keys()[-3], df.Location.value_counts()[-1]),unsafe_allow_html=True)
        
        #Valeurs manquantes
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>",unsafe_allow_html=True)
        st.markdown("Pas de valeurs manquantes pour ces deux variables.",unsafe_allow_html=True)
        
        #Création de variables (texte)
        st.markdown("<h4><u><font color = 'navy'>Création de variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<p>Création de la variable <b><i>State</i></b> qui représente la localisation des stations météorologiques par états.<ul><li>Les données de Canberra sont mutualisées avec la Nouvelle-Galle du Sud.</li><li>L'île de Norfolk est rattachée administrativement à l'état de Nouvelle-Galles du Sud, mais vu sa situation géographique (environ 1500km à l'est de l'Australie), les données seront traitées à part.</li><li>Si besoin, le classement par état est disponible dans le dictionnaire <i>localites</i>.</li></ul></p>",unsafe_allow_html=True)
        st.markdown("<p>Création des variables <b><i>Year</i></b>, <b><i>Month</i></b> et <b><i>Day</i></b>, des entiers représentant respectivement l'année, le mois et le jour.",unsafe_allow_html=True) 
            
        #Visualisation
        st.markdown("<h4><u><font color = 'navy'>Visualisation graphique :</u></h4>",unsafe_allow_html=True)
        
        fig1 = plt.figure(figsize = (6,3))
        sns.countplot("Year", data = df, palette = "cividis")
        plt.title("Nombre de relevés disponibles par année.", fontsize = 15, pad = 20)
        plt.xticks(size = 9)
        plt.yticks(size = 9)
        plt.xlabel('Années', fontsize=12)
        plt.ylabel('Nombre de relevés', fontsize=12)
        st.pyplot(fig1);
    
    ## TEMPERATURES 
        
    elif var_a_afficher == "Températures":
        st.markdown("<h2><u><center>MinTemp, MaxTemp, Temp9am, Temp3pm</center></u></h2>",unsafe_allow_html=True)
    
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("""<ul><li><b>MinTemp</b> est définie comme <i>'The minimum temperature in degrees celsius'</i>, soit la température minimale relevée sur la journée.</li>
        <li><b>MaxTemp</b> est définie comme <i>'The maximum temperature in degrees celsius'</i>, soit la température maximale relevée sur la journée.</li>
        <li><b>Temp9am</b> est définie comme <i>'Temperature (degrees C) at 9am'</i>, soit la température relevée à 9h.</li>
        <li><b>Temp3pm</b> est définie comme <i>'Temperature (degrees C) at 3pm'</i>, soit la température relevée à 15h.</li></ul>""",unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<h5><i>MinTemp et MaxTemp :</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">Ces deux variables varient de façon similaire au fil des mois : 
        les températures les plus hautes sont retrouvées en janvier et les plus basses en juillet. Au mois de juillet, 
        les températures minimales et maximales moyennes sont de <b>{} ± {} °C</b> et <b>{} ± {} °C</b> respectivement, 
        tandis qu'au mois de janvier elles sont de <b>{} ± {} °C</b> et <b>{} ± {} °C</b> respectivement.</div>
        """.format(
            df[df["Month"] == 7].MinTemp.describe()["mean"].round(1),
            df[df["Month"] == 7].MinTemp.describe()["std"].round(1),
            df[df["Month"] == 7].MaxTemp.describe()["mean"].round(1),
            df[df["Month"] == 7].MaxTemp.describe()["std"].round(1),
            df[df["Month"] == 1].MinTemp.describe()["mean"].round(1),
            df[df["Month"] == 1].MinTemp.describe()["std"].round(1),
            df[df["Month"] == 1].MaxTemp.describe()["mean"].round(1),
            df[df["Month"] == 1].MaxTemp.describe()["std"].round(1)),unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>Temp9am et Temp3pm:</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">La température moyenne à 9h est <b>plus basse les jours de pluie</b> (<b>{} ± {} °C</b>) que les jours sans pluie (<b>{} ± {} °C</b>).
        La variabilité est assez similaire qu'il pleuve ou non et il y'a de nombreux outliers, principalement dans les valeurs hautes.
        D'une localité à l'autre, la température matinale semble peu varier, mais plusieurs villes du nord se distinguent par leur températures
        plus élevées : Brisbane, Cairns, Gold Coast, Townsville, Alice Springs, Darwin, Katherine et Uluru. Les températures matinales 
        sont minimales au mois de <b>juillet</b>, qu'il pleuve ou non, et avec une variabilité par mois limitée et stable d'une année 
        sur l'autre. Les minimales moyennes en juillet sont de <b>{} ± {} °C</b> les jours de pluie, <b>{} ± {} °C</b> sinon. Les températures 
        matinales maximales sont observées au mois de <b>janvier</b> et sont de <b>{} ± {} °C</b> les jours de pluie, {} ± {} °C sinon.
        Les observations à 15h suivent les <b>mêmes tendances</b> qu'à 9h. La température moyenne à 15h est plus basse les jours de pluie
        <b>({} ± {} °C)</b> que les jours sans pluie <b>({} ± {} °C)</b>. Enfin, les températures l'après-midi sont maximales en <b>juillet</b>, 
        <b>{} ± {} °C</b> les jours de pluie, <b>{} ± {} °C</b> sinon. Au mois de <b>janvier</b>, elles sont de <b>{} ± {} °C</b> 
        es jours de pluie, <b>{} ± {} °C</b> sinon.</div>
        """.format(
            df[df["RainToday"] == "Yes"].Temp9am.describe()["mean"].round(1),
            df[df["RainToday"] == "Yes"].Temp9am.describe()["std"].round(1),
            df[df["RainToday"] == "No"].Temp9am.describe()["mean"].round(1),
            df[df["RainToday"] == "No"].Temp9am.describe()["std"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 7)].Temp9am.describe()["mean"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 7)].Temp9am.describe()["std"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 7)].Temp9am.describe()["mean"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 7)].Temp9am.describe()["std"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 1)].Temp9am.describe()["mean"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 1)].Temp9am.describe()["std"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 1)].Temp9am.describe()["mean"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 1)].Temp9am.describe()["std"].round(1),
            df[df["RainToday"] == "Yes"].Temp3pm.describe()["mean"].round(1),
            df[df["RainToday"] == "Yes"].Temp3pm.describe()["std"].round(1),
            df[df["RainToday"] == "No"].Temp3pm.describe()["mean"].round(1),
            df[df["RainToday"] == "No"].Temp3pm.describe()["std"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 7)].Temp3pm.describe()["mean"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 7)].Temp3pm.describe()["std"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 7)].Temp3pm.describe()["mean"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 7)].Temp3pm.describe()["std"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 1)].Temp3pm.describe()["mean"].round(1),
            df[(df["RainToday"] == "Yes") & (df["Month"] == 1)].Temp3pm.describe()["std"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 1)].Temp3pm.describe()["mean"].round(1),
            df[(df["RainToday"] == "No") & (df["Month"] == 1)].Temp3pm.describe()["std"].round(1)
        ),unsafe_allow_html=True)
        
        #Valeurs manquantes
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>",unsafe_allow_html=True)
        st.markdown("<ul><li>MinTemp : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>MaxTemp : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>Temp9am : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>Temp3pm : <i><b>{}</b> valeurs manquantes ({} %)</i>.</li></ul>".format(
            df_saved.MinTemp.isna().sum(),
            ((100*df_saved.MinTemp.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.MaxTemp.isna().sum(),
            ((100*df_saved.MaxTemp.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Temp9am.isna().sum(),
            ((100*df_saved.Temp9am.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Temp3pm.isna().sum(),
            ((100*df_saved.Temp3pm.isna().sum())/df_saved.shape[0]).round(1)),unsafe_allow_html=True)
        
        #Visualisation graphique
        st.markdown("<h4><u><font color = 'navy'>Visualisation graphique :</u></h4>",unsafe_allow_html=True)
        
        fig_temp = plt.figure(figsize = (10, 15))
        plt.suptitle("Variation mensuelle des températures", fontsize = 24, fontweight = "bold")
        plt.subplot(3,1,1)
        plt.title("Températures maximales et minimales.", fontsize = 18, pad = 20)
        sns.lineplot(x = "Month", y = "MaxTemp", data = df, label = "Température maximale", color = "orange", marker = "o")
        sns.lineplot(x = "Month", y = "MinTemp", data = df, label = "Température minimale", color = "teal", marker = "D")
        plt.ylabel("Température", fontsize = 15)
        plt.xlabel("Mois", fontsize = 15)
        plt.legend(facecolor = 'white')
        plt.grid(axis = "y")
        plt.ylim([0,30])
        plt.yticks(range(0,31,2), size = 12)
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["jan.", "fev.", "mar.", "avr.", "mai", "juin", "juil", "août", "sep.", "oct.", "nov.", "déc."], size = 12)
        
        plt.subplot(3,1,2)
        plt.title("Températures à 9h.", fontsize = 18, pad = 20)
        sns.lineplot(x = "Month", y = "Temp9am", data = df[df["RainToday"] == "Yes"], label = "Pluie", color = "navy", marker = "o")
        sns.lineplot(x = "Month", y = "Temp9am", data = df[df["RainToday"] == "No"], label = "Pas de pluie", color = "red", marker = "o")
        plt.xlabel("Mois", fontsize = 15)
        plt.ylabel("Température", size = 15)
        plt.legend(facecolor = 'white')
        plt.grid(axis = "y")
        plt.ylim([0,30])
        plt.yticks(range(0,31,2), size = 12)
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["jan.", "fev.", "mar.", "avr.", "mai", "juin", "juil", "août", "sep.", "oct.", "nov.", "déc."], size = 12)
        
        plt.subplot(3,1,3)
        plt.title("Températures à 15h.", fontsize = 18, pad = 20)
        sns.lineplot(x = "Month", y = "Temp3pm", data = df[df["RainToday"] == "Yes"], label = "Pluie", color = "navy", marker = "o")
        sns.lineplot(x = "Month", y = "Temp3pm", data = df[df["RainToday"] == "No"], label = "Pas de pluie", color = "red", marker = "o")
        plt.xlabel("Mois", fontsize = 15)
        plt.ylabel("Température", size = 15)
        plt.legend(facecolor = 'white')
        plt.grid(axis = "y")
        plt.ylim([0,30])
        plt.yticks(range(0,31,2), size = 12)
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["jan.", "fev.", "mar.", "avr.", "mai", "juin", "juil", "août", "sep.", "oct.", "nov.", "déc."], size = 12)
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_temp);        
    
    ##PRECIPITATIONS
        
    elif var_a_afficher == "Précipitations":
    
        #RAINFALL, EVAPORATION, HUMIDITY9AM, HUMIDITY3PM
        st.markdown("<h2><u><center>Rainfall, Evaporation, Humidity9am, Humidity3pm</center></u></h2>",unsafe_allow_html=True)
        
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("""<ul><li><b>Rainfall</b> est définie comme <i>'The amount of rainfall recorded for the day in mm'</i>, soit la hauteur des précipitations en mm.</li>
        <li><b>Evaporation</b> est définie comme <i>'The so-called Class A pan evaporation (mm) in the 24 hours to 9am'</i>, soit hauteur évaporée sur la journée, relevée à 9h.</li>
        <li><b>Humidity9am</b> est définie comme <i>'Humidity (percent) at 9am'</i>, soit le pourcentage d'humidité relevé à 9h.</li>
        <li><b>Humidity3pm</b> est définie comme <i>'Humidity (percent) at 3pm'</i>, soit le pourcentage d'humidité relevé à 15h.</li></ul>""",unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<h5><i>Rainfall :</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">Il y'a <b>{}</b> relevés qui concernent des jours de pluie (soit <b>{} %</b> des relevés), avec une grande variabilité 
        de la hauteur de précipitations. En effet, les jours de pluie la hauteur moyenne des précipitations est de <b>{} ± {} cm</b>, 
        avec une valeur maximale de <b>{} cm</b>. Attention, les moyennes sont biaisées par des valeurs hautes, la hauteur médiane 
        des précipitations étant de <b>{} cm</b>, IQR : (<b>{} - {}</b>). Enfin, cette variable est directement liée à RainToday : 
        RainToday = '<i>no</i>' quand Rainfall ≤ 1.0 et RainToday = '<i>yes</i>' quand Rainfall > 1.0 cm.</div>
        """.format(
            df.Rainfall[df["Rainfall"] > 1].count(),
            (100*(df.Rainfall[df["Rainfall"] > 1].count())/len(df.Rainfall)).round(2),
            df.groupby("RainToday")["Rainfall"].describe()["mean"][1].round(1),
            df.groupby("RainToday")["Rainfall"].describe()["std"][1].round(1),
            df.groupby("RainToday")["Rainfall"].describe()["max"][1],
            df.groupby("RainToday")["Rainfall"].describe()["50%"][1],
            df.groupby("RainToday")["Rainfall"].describe()["25%"][1],
            df.groupby("RainToday")["Rainfall"].describe()["75%"][1]),unsafe_allow_html=True)    
        
        st.markdown("<h5><i><br/>Evaporation :</i></h5>",unsafe_allow_html=True)
        st.markdown(("""<div style="text-align: justify">Evaporation présente beaucoup d'outliers et est visiblement correlée à RainToday. 
        Le phénomène d'évaporation est plus imporant les jours sans pluie, aussi cette variable semble se comporter 
        inversement à Rainfall en fonction de RainToday. La hauteur d'évaporation médiane est de <b>{} cm</b>, 
        IQR (<b>{} - {}</b>) les jours de pluie et de <b>{} cm</b>, IQR (<b>{} - {}</b>) les jours sans pluie.</div>
        """.format(
            df.groupby("RainToday")["Evaporation"].describe()["50%"][1],
            df.groupby("RainToday")["Evaporation"].describe()["25%"][1],
            df.groupby("RainToday")["Evaporation"].describe()["75%"][1],
            df.groupby("RainToday")["Evaporation"].describe()["50%"][0],
            df.groupby("RainToday")["Evaporation"].describe()["25%"][0],
            df.groupby("RainToday")["Evaporation"].describe()["75%"][0])),unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>Humidity9am :</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">Le pourcentage d'humidité à 9h moyen est de <b>{} ± {} %</b> les jours de pluie et de 
        <b>{} ± {} %</b> les jours sans pluie. L'humidité à 9h ne semble pas corrélée à l'humidité à 15h ; en
        revanche une corrélation avec la hauteur des précipitations pourrait exister
        (notamment pour les hautes précipitations), mais limitée par la forte variabilité existant pour le taux 
        d'humidité. Le taux d'humidité à 9h varie relativement peu d'une localité à l'autre, à l'exception de certaines 
        zones situées dans les terres (Alice Springs, Woomera, Cobar, Uluru...) qui présentent des taux plus bas. 
        On remarque dans l'ensemble de nombreux outliers pour les valeurs d'humidité plus basses, et il apprait 
        que l'humidité à 9h varie au fil des mois, de façon assez similaire qu'il pleuve ou non, et avec une variabilité 
        par mois limitée et stable d'une année sur l'autre.</div>
        """.format(
            df[df["RainToday"] == "Yes"].Humidity9am.describe()["mean"].round(1),
            df[df["RainToday"] == "Yes"].Humidity9am.describe()["std"].round(1),
            df[df["RainToday"] == "No"].Humidity9am.describe()["mean"].round(1),
            df[df["RainToday"] == "No"].Humidity9am.describe()["std"].round(1)),unsafe_allow_html=True)
                
        st.markdown("<h5><i><br/>Humidity3pm :</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">Le pourcentage d'humidité à 15h moyen est de <b>{} ± {} %</b> les jours de pluie 
        et de <b>{} ± {} %</b> les jours sans pluie. Il existe une variabilité notoire pour le taux d'humidité à 15h 
        d'une localité à l'autre. A la différence de l'humidité à 9h, il existe aussi des outliers pour les valeurs
        hautes ; les autres observations à 9h sont transposables à 15h. Enfin, le taux d'humidité à 15h est globalement
        plus faible que celui relevé à 9h.</div>
        """.format(
            df[df["RainToday"] == "Yes"].Humidity3pm.describe()["mean"].round(1),
            df[df["RainToday"] == "Yes"].Humidity3pm.describe()["std"].round(1),
            df[df["RainToday"] == "No"].Humidity3pm.describe()["mean"].round(1),
            df[df["RainToday"] == "No"].Humidity3pm.describe()["std"].round(1)),unsafe_allow_html=True)
        
        #Valeurs manquantes
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>",unsafe_allow_html=True)
        st.markdown("<ul><li>Rainfall : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>Evaporation : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>Humidity9am : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>Humidity3pm : <i><b>{}</b> valeurs manquantes ({} %)</i>.</li></ul>".format(
            df_saved.Rainfall.isna().sum(),
            ((100*df_saved.Rainfall.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Evaporation.isna().sum(),
            ((100*df_saved.Evaporation.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Humidity9am.isna().sum(),
            ((100*df_saved.Humidity9am.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Humidity3pm.isna().sum(),
            ((100*df_saved.Humidity3pm.isna().sum())/df_saved.shape[0]).round(1)),unsafe_allow_html=True)
        
        #Visualisation graphique
        st.markdown("<h4><u><font color = 'navy'>Visualisation graphique :</u></h4>",unsafe_allow_html=True)
        
        #histogrammes
        fig_precipitations_1 = plt.figure(figsize = (13,13))
        plt.suptitle("Distribution des variables Rainfall, Evaporation, Humidity9am et Humidity3pm", fontsize = 22, fontweight = "bold")
        
        plt.subplot(221)
        plt.title("Hauteur des précipitations\n (les jours de pluie)", fontsize = 18, pad = 20)
        sns.histplot(x = "Rainfall", data = df[df["Rainfall"] > 1], bins = 200)
        plt.xticks(range(0,51,5), size = 14)
        plt.xlim(0,50)
        plt.xlabel("Rainfall", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        
        plt.subplot(222)
        plt.title("Hauteur de l'évaporation", fontsize = 18, pad = 20)
        sns.histplot(x = "Evaporation", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 150)
        plt.xlim(0,25)
        plt.xticks(range(0,26,5), size = 14)
        plt.xticks()
        plt.xlabel("Evaporation", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(223)
        plt.title("Pourcentage d'humidité\n (relevé à 9h)", fontsize = 18, pad = 20)
        sns.histplot(x = "Humidity9am", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 15)
        plt.xticks(range(0,101,10), size = 14)
        plt.xlabel("Humidity9am", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(224)
        plt.title("Pourcentage d'humidité\n (relevé à 15h)", fontsize = 18, pad = 20)
        sns.histplot(x = "Humidity3pm", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 15)
        plt.xticks(range(0,101,10), size = 14)
        plt.xlabel("Humidity3pm", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.tight_layout(pad=2)
        st.pyplot(fig_precipitations_1);
        
        #boxplots
        fig_precipitations_2 = plt.figure(figsize = (11,11))
        plt.suptitle("Distribution comparée en fonction de la pluie", fontsize = 22, fontweight = "bold")
        
        plt.subplot(221)
        plt.title("Hauteur des précipitations", fontsize = 18, pad = 20)
        sns.boxplot(y = "Rainfall", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("Rainfall (mm)", size =16)
        plt.ylim([0,50])
        
        plt.subplot(222)
        plt.title("Hauteur de l'évaporation", fontsize = 18, pad = 20)
        sns.boxplot(y = "Evaporation", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("Evaporation (mm)", size =16)
        plt.ylim([0,25])
        
        plt.subplot(223)
        plt.title("Pourcentage d'humidité\n (relevé à 9h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "Humidity9am", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("Humidity9am (%)", size =16)
        
        plt.subplot(224)
        plt.title("Pourcentage d'humidité\n (relevé à 15h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "Humidity3pm", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("Humidity3pm (%)", size =16)
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_precipitations_2);
        
        st.markdown("<i>Pour une meilleure visualisation, les données de Rainfall sont représentées jusqu'à 50 mm, les outliers vont jusque {} mm. Les données pour Evaporation sont représentées jusqu'à 25 mm, les outliers vont jusque {} mm.</i>".format(
        df.Rainfall.describe()["max"],
        df.Evaporation.describe()["max"]),unsafe_allow_html=True)    
    
    ##ENSOLEILLEMENT, ENNUAGEMENT.
    
    elif var_a_afficher == "Ensoleillement, ennuagement":        
              
        #SUNSHINE, CLOUDS9AM, CLOUDS3PM
        st.markdown("<h2><u><center>Sunshine, Clouds9am, Clouds3pm</center></u></h2>",unsafe_allow_html=True)
        
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("""<ul><li><b>Sunshine</b> est définie comme <i>'The number of hours of bright sunshine in the day.'</i>, soit le nombre d'heures d'ensoleillement sur la journée.</li>
        <li><b>Cloud9am</b> est définie comme <i>'Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many'</i>, soit le degré de couverture du ciel, mesuré en octas à 9h.</li>
        <li><b>Cloud3pm</b> est définie comme <i>'Fraction of sky obscured by cloud at 3pm. This is measured in "oktas", which are a unit of eigths. It records how many'</i>, soit le degré de couverture du ciel, mesuré en octas à 15h.</li>
        </ul>""",unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<h5><i>Sunshine :</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">L'ensoleillement médian est de <b>{} heures</b> 
        les jours de pluie et de <b>{} heures</b> les jours sans pluie. Toute météo confondue, la durée d'ensoleillement est 
        <b>maximale</b> entre <b>décembre et février</b> (<b>{} heures</b>) et <b>minimale</b> entre <b>mai et juillet</b> 
        (<b>{} heures</b>). D'une année à l'autre, mois par mois, il existe une variabilité de l'ensoleillement plus importante 
        les jours de pluie que les jours sans pluie.</div>
        """.format(
            df[df["RainToday"] == "Yes"].Sunshine.median(),
            df[df["RainToday"] == "No"].Sunshine.median(),
            df[(df["Month"] == 12) | (df["Month"] == 1) | (df["Month"] == 2)].Sunshine.median(),
            df[(df["Month"] == 5) | (df["Month"] == 6) | (df["Month"] == 7)].Sunshine.median()),unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>Cloud9am :</i></h5>",unsafe_allow_html=True)       
        st.markdown("""<div style="text-align: justify">L'ennuagement médian à 9h est de <b>{}</b> les jours 
        de pluie et de <b>{}</b> les jours sans pluie.La variabilité est nettement plus importante les jours sans pluie. 
        L'ennuagement à 9h est très inconstant d'une localité à l'autre, toutefois il y a dans l'ensemble <b>peu d'outliers</b>. 
        L'ennuagement à 9h est totalement décorrélé de celui à 15h. <b>Douze localités</b> n'ont <b>aucun relevé</b>
        d'ennuagement à 9h : Badgery Creek, Norah Head, Penrith, Tuggeranong, Mount Ginini, Nhil, Dartmoor, Gold Coast, 
        Adelaide, Witchcliffe, Salmon Gums et Walpole. Enfin, l'ennuagement à 9h varie au fil des mois, de façon assez 
        similaire qu'il pleuve ou non, et avec une variabilité par mois limitée et stable d'une année sur l'autre.</div>
        """.format(
            round(df[df["RainToday"] == "Yes"].Cloud9am.median(),1),
            round(df[df["RainToday"] == "No"].Cloud9am.median(),1)),unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>Cloud3pm :</i></h5>",unsafe_allow_html=True)       
        st.markdown("""<div style="text-align: justify">L'ennuagement médian à 15 est de <b>{}</b> les jours de pluie 
        et de <b>{}</b> les jours sans pluie. En dépit de l'absence de corrélation avec l'ennuagement matinal, Cloud3pm se comporte similairement à Cloud9am.</div>
        """.format(
            round(df[df["RainToday"] == "Yes"].Cloud3pm.median(),1),
            round(df[df["RainToday"] == "No"].Cloud3pm.median(),1)),unsafe_allow_html=True)
        
        #Valeurs manquantes
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>",unsafe_allow_html=True)
        st.markdown("<ul><li>Sunshine : <i><b>{}</b> nans ({} %)</i></li> <li>Clouds9am : <i><b>{}</b> nans ({} %)</i></li> <li>Clouds3pm : <i><b>{}</b> nans ({} %)</i></li></ul>".format(
            df_saved.Sunshine.isna().sum(),
            ((100*df_saved.Sunshine.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Cloud9am.isna().sum(),
            ((100*df_saved.Cloud9am.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Cloud3pm.isna().sum(),
            ((100*df_saved.Cloud3pm.isna().sum())/df_saved.shape[0]).round(1)),unsafe_allow_html=True)      
        
        #Visualisation graphique
        st.markdown("<h4><u><font color = 'navy'>Visualisation graphique :</u></h4>",unsafe_allow_html=True)
        
        #histogrammes
        fig_soleil_1 = plt.figure(figsize = (12,12))
        plt.suptitle("Distribution des variables Sunshine, Cloud9am et Cloud3pm", fontsize = 22)
        
        plt.subplot(221)
        plt.title("Ensoleillement", fontsize = 18, pad = 20)
        sns.histplot(x = "Sunshine", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 15)
        plt.xticks(size = 14)
        #plt.xlim(0,50)
        plt.xlabel("Sunshine", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(223)
        plt.title("Ennuagement\n (relevé à 9h)", fontsize = 18, pad = 20)
        sns.histplot(x = "Cloud9am", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 9)
        #plt.xlim(0,25)
        plt.xticks(range(0,10,1), size = 14)
        plt.xticks()
        plt.xlabel("Cloud9am", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(224)
        plt.title("Ennuagement\n (relevé à 15h)", fontsize = 18, pad = 20)
        sns.histplot(x = "Cloud3pm", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 9)
        plt.xticks(range(0,10,1), size = 14)
        plt.xlabel("Cloud3pm", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_soleil_1);
        
        #boxplots
        fig_soleil_2 = plt.figure(figsize = (12,12))
        plt.suptitle("Distribution comparée en fonction de la pluie", fontsize = 22)
        
        plt.subplot(221)
        plt.title("Ensoleillement", fontsize = 18, pad = 20)
        sns.boxplot(y = "Sunshine", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(range(0,17,2),size = 14)
        plt.ylabel("Sunshine (h)", size =16)
        plt.ylim(0,16)
        
        plt.subplot(223)
        plt.title("Ennuagement\n (relevé à 9h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "Cloud9am", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(range(0,10,1), size = 14)
        plt.ylabel("Cloud9am (octa)", size =16)
        
        plt.subplot(224)
        plt.title("Ennuagement\n (relevé à 15h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "Cloud3pm", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(range(0,10,1), size = 14)
        plt.ylabel("Cloud3pm (octa)", size =16)
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_soleil_2);
        
        display(Markdown("""<i>L'obscurcissement du ciel par les nuages (variables Cloud9am et Cloud3pm) est mesurés en octa : 
        un ciel parfaitement clair est indiqué par la valeur de 0 octa, alors qu'un ciel complètement couvert est estimé à 8 octas. 
        La valeur spéciale de 9 octas est utilisée quand le ciel n'est pas observable en raison d'une obstruction à la visibilité 
        (par exemple en cas de brouillard).</i>"""))
    
    ##DIRECTION DU VENT.
    
    elif var_a_afficher == "Direction du vent":
        
        st.markdown("<h2><u><center>WindGustDir, WindDir9am, WindDir3pm</center></u></h2>", unsafe_allow_html=True)
        
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("""<ul><li><b>WindGustDir</b> est définie comme <i>'The direction of the strongest wind gust in the 24 hours to 
        midnight.'</i>, soit la direction de la plus forte rafale de vent sur 24h.</li>
        <li><b>WindDir9am</b> est définie comme <i>'Direction of the wind at 9am'</i>, soit la direction du vent à 9h.</li>
        <li><b>WindDir3pm</b> est définie comme <i>'Direction of the wind at 3pm'</i>, soit la direction du vent à 15h.</li>
        </ul>""", unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("<h5><i>WindGustDir :</i></h5>", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">Cette variable comporte <b>16 modalités</b> (Nord, Nord-Nord-Est, Nord-Est, Est-Nord-Est, Est...).
         Au sein d'un même état, les vents dominants ne sont jamais largement majoritaires : il existe une
         <b>importante variabilité</b> des directions qu'il pleuve ou non.
         Deux stations n'ont <b>aucun relevé</b> : <b>Albany</b> et <b>Newcastle</b>.</div>
        """, unsafe_allow_html=True)
        st.markdown("<h5><i><br/>WindDir9am et WindDir3pm :</i></h5>", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: justify">Ces deux variables sont comportent les 16 mêmes modalités que WindGustDir.
         La matrice de confusion entre les données mesurées à 9h et à 15h montre qu'à la fois,
          s'il est très fréquent que la direction du vent ne soit pas la même entre ces deux horaires,
           il reste assez rare que le vent change complètement d'orientation. 
           Par exemple, pour un vent venant du Sud à 9h, les principales directions à 15h sont Sud, 
           Sud-Sud-Est, Sud-Est, Sud-Sud-Ouest et Sud-Ouest.</div>
        """, unsafe_allow_html=True)
        
        #Valeurs manquantes
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>", unsafe_allow_html=True)
        st.markdown("<ul><li>WindGustDir : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>WindDir9am : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>WindDir3pm : <i><b>{}</b> valeurs manquantes ({} %)</i></li></ul>".format(
            df_saved.WindGustDir.isna().sum(),
            ((100*df_saved.WindGustDir.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.WindDir9am.isna().sum(),
            ((100*df_saved.WindDir9am.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.WindDir3pm.isna().sum(),
            ((100*df_saved.WindDir3pm.isna().sum())/df_saved.shape[0]).round(1)),unsafe_allow_html=True)   
        
    elif var_a_afficher == "Force du vent":
        #WINDGUSTSPEED, WINDSPEED9AM, WINDSPEED3PM,
        st.markdown("<h2><u><center>WindGustSpeed, WindSpeed9am, WindSpeed3pm</center></u></h2>", unsafe_allow_html=True)
        
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("""<ul><li><b>WindGustSpeed</b> est définie comme <i>'The speed (km/h) of the strongest wind gust in the 24 hours to midnight'</i>, soit la vitesse de la plus forte rafale de vent sur 24h.</li>
        <li><b>WindSpeed9am</b> est définie comme <i>'Wind speed (km/hr) averaged over 10 minutes prior to 9am'</i>, soit la vitesse moyenne du vent entre 8:50 et 9:00.</li>
        <li><b>WindSpeed3pm</b> est définie comme <i>'Wind speed (km/hr) averaged over 10 minutes prior to 3pm'</i>, soit la vitesse moyenne du vent entre 14:50 et 15:00.</li>
        </ul>""",unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("<h5><i>WindGustSpeed :</i></h5>",unsafe_allow_html=True)
        st.markdown("""<div style = 'text-align : justify'>Les relevés vont de <b>{}</b> à <b>{} km/h</b>, 
        avec une vitesse moyenne de <b>{} km/h</b>. Il existe de nombreux outliers dans les valeurs hautes, 
        mais leur impact reste limité (vitesse médiane : <b>{} km/h</b>). Similairement à la variable WindGustDir, 
        il n'y a <b>aucun relevé</b> pour les stations <b>Albany</b> et <b>Newcastle</b>. De façon intéressante, 
        les données suggèrent que le vent souffle plus fort les jours de pluie : <b>{} km/h</b> en moyenne 
         contre <b>{} km/h</b> les jours sans pluie. Enfin, la variabilité semble plus importante les jours de pluie.</div>
        """.format(df.WindGustSpeed.describe()["min"],
                   df.WindGustSpeed.describe()["max"],
                   df.WindGustSpeed.describe()["mean"].round(1),
                   df.WindGustSpeed.describe()["50%"],
                   df[df["RainToday"] == "Yes"].WindGustSpeed.describe()["mean"].round(1),
                   df[df["RainToday"] == "No"].WindGustSpeed.describe()["mean"].round(1)), unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>WindSpeed9am :</i></h5>", unsafe_allow_html=True)
        st.markdown("""<div style = 'text-align : justify'>La vitesse moyenne du vent à 9h est de <b>{} ± {} km/h</b> 
        les jours de pluie, et de <b>{} ± {} km/h</b> les jours sans pluie. Les données à 9h sont grossièrement corrélées 
        à celles mesurées à 15h, ainsi qu'avec le relevé de vitesse de la plus forte rafale,
        avec toutefois une importante variabilité.</div>
        """.format(df[df["RainToday"] == "Yes"].WindSpeed9am.describe()["mean"].round(1),
                   df[df["RainToday"] == "Yes"].WindSpeed9am.describe()["std"].round(1),
                   df[df["RainToday"] == "No"].WindSpeed9am.describe()["mean"].round(1),
                   df[df["RainToday"] == "No"].WindSpeed9am.describe()["std"].round(1)), unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>WindSpeed3pm :</i></h5>", unsafe_allow_html=True)
        st.markdown("""<div style = 'text-align : justify'>La vitesse moyenne du vent à 15h est de <b>{} ± {} km/h</b> 
        les jours de pluie, et de <b>{} ± {} km/h</b> les jours sans pluie. Les données à 15h sont grossièrement 
        corrélées avec le relevé de vitesse de la plus forte rafale, avec toutefois une importante variabilité.</div>
        """.format(df[df["RainToday"] == "Yes"].WindSpeed3pm.describe()["mean"].round(1),
                   df[df["RainToday"] == "Yes"].WindSpeed3pm.describe()["std"].round(1),
                   df[df["RainToday"] == "No"].WindSpeed3pm.describe()["mean"].round(1),
                   df[df["RainToday"] == "No"].WindSpeed3pm.describe()["std"].round(1)), unsafe_allow_html=True)
        
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>", unsafe_allow_html=True)
        st.markdown("<ul><li>WindGustSpeed : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>WindSpeed9am : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>WindSpeed3pm : <i><b>{}</b> valeurs manquantes ({} %)</i></li></ul>".format(
            df_saved.WindGustSpeed.isna().sum(),
            ((100*df_saved.WindGustSpeed.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.WindSpeed9am.isna().sum(),
            ((100*df_saved.WindSpeed9am.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.WindSpeed3pm.isna().sum(),
            ((100*df_saved.WindSpeed3pm.isna().sum())/df_saved.shape[0]).round(1)), unsafe_allow_html=True)      
        
        #Visualisation graphique
        st.markdown("<h4><u><font color = 'navy'>Visualisation graphique :</u></h4>", unsafe_allow_html=True)
        
        #histogrammes
        fig_vent_1 = plt.figure(figsize = (12,12))
        plt.suptitle("Distribution des variables WindGustSpeed, WindSpeed9am et WindSpeed3pm", fontsize = 22, fontweight = "bold")
        
        plt.subplot(221)
        plt.title("Vitesse du vent\n (plus forte rafale)", fontsize = 18, pad = 20)
        sns.histplot(x = "WindGustSpeed", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 24)
        plt.xticks(range(0,101,10), size = 14)
        plt.xlim(0,100)
        plt.xlabel("WindGustSpeed", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(223)
        plt.title("Vitesse du vent\n (relevée à 9h)", fontsize = 18, pad = 20)
        sns.histplot(x = "WindSpeed9am", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 30)
        plt.xlim(0,60)
        plt.xticks(size = 14)
        plt.xticks()
        plt.xlabel("WindSpeed9am", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(224)
        plt.title("Vitesse du vent\n (relevée à 15h)", fontsize = 18, pad = 20)
        sns.histplot(x = "WindSpeed3pm", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 20)
        plt.xticks(size = 14)
        plt.xlim(0,60)
        plt.xlabel("WindSpeed3pm", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_vent_1);
        
        st.markdown("""<i>Pour une meilleure visualisation, les données de WindGustSpeed sont représentées jusqu'à 100 km/h,
        les outliers vont jusque {} km/h. Les données pour WindSpeed 9am et 3am sont représentées jusqu'à 60 km/h, 
        les outliers vont jusque {} km/h (9am) et jusque {} km/h (3pm).</i>""".format(
        df.WindGustSpeed.describe()["max"],
        df.WindSpeed9am.describe()["max"],
        df.WindSpeed3pm.describe()["max"]), unsafe_allow_html=True)
        
        #boxplots
        fig_vent_2 = plt.figure(figsize = (12,12))
        plt.suptitle("Distribution comparée en fonction de la pluie", fontsize = 22, fontweight = "bold")
        
        plt.subplot(221)
        plt.title("Vitesse du vent\n (plus forte rafale)", fontsize = 18, pad = 20)
        sns.boxplot(y = "WindGustSpeed", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("WindGustSpeed (km/h)", size =16)
        #plt.ylim(0,16)
        
        plt.subplot(223)
        plt.title("Vitesse du vent\n (relevée à 9h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "WindSpeed9am", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("WindSpeed9am (km/h)", size =16)
        
        plt.subplot(224)
        plt.title("Vitesse du vent\n (relevée à 15h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "WindSpeed3pm", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("WindSpeed3pm (km/h)", size =16)
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_vent_2);
    
    ##FORCE DU VENT.
    
    elif var_a_afficher == "Pression atmosphérique":
        st.markdown("<h2><u><center>Pressure9am, Pressure3pm</center></u></h2>", unsafe_allow_html=True)
        
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("""<ul><li><b>Pressure9am</b> est définie comme <i>'Atmospheric pressure (hpa) reduced to mean sea level at 9am'</i>, soit la pression atmosphérique en fonction du niveau de la mer, à 9h.</li>
        <li><b>Pressure3pm</b> est définie comme <i>'Atmospheric pressure (hpa) reduced to mean sea level at 3pm'</i>, soit la pression atmosphérique en fonction du niveau de la mer, à 15h.</li>
        </ul>""", unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("<h5><i>Pressure9am :</i></h5>", unsafe_allow_html=True)
        st.markdown("""<div style = "text-align : justify">La pression atmosphérique moyenne à 9h est <b>plus basse</b> 
        les jours de pluie (<b>{} ± {} hpa</b>) que les jours sans pluie (<b>{} ± {} hpa</b>). Il y'a de nombreux outliers, 
        pour les valeurs hautes comme pour les valeurs basses. Cette variable est remarquablement constante d'une localité 
        à l'autre, avec une amplitude qui varie très peu. Exception notable pour quatre localités : Darwin (NT), Katherine (NT), 
        Cairns (QLD) et Townsville (QLD) qui ont des <b>pressions plus faibles</b> (toutes situées dans la partie 
        <b>Nord de l'Australie</b>). <b>Quatre localités</b> n'ont <b>aucun relevé</b> de pression atmosphérique à 9h : 
        Newcastle, Penrith, Mount Ginini et Salmon Gums.</div>
        """.format(df[df["RainToday"] == "Yes"].Pressure9am.describe()["mean"].round(1),
                  df[df["RainToday"] == "Yes"].Pressure9am.describe()["std"].round(1),
                  df[df["RainToday"] == "No"].Pressure9am.describe()["mean"].round(1),
                  df[df["RainToday"] == "No"].Pressure9am.describe()["std"].round(1)), unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>Pressure3pm :</i></h5>", unsafe_allow_html=True)
        st.markdown("""<div style = "text-align : justify">La pression atmosphérique moyenne à 15h est <b>plus basse</b> 
        les jours de pluie (<b>{} ± {} hpa</b>) que les jours sans pluie (<b>{} ± {} hpa</b>).
        La distribution de la pression atmosphérique est <b>très similaire à 15h et à 9h</b>. Toutefois à 15h, la différence 
        de pression entre les jours avec et sans pluie est moins marquée qu'à 9h. Il y'a également, à 15h, de nombreux outliers, 
        pour les valeurs hautes comme pour les valeurs basses.</div>
        """.format(df[df["RainToday"] == "Yes"].Pressure3pm.describe()["mean"].round(1),
                   df[df["RainToday"] == "Yes"].Pressure3pm.describe()["std"].round(1),
                   df[df["RainToday"] == "No"].Pressure3pm.describe()["mean"].round(1),
                   df[df["RainToday"] == "No"].Pressure3pm.describe()["std"].round(1)
                  ), unsafe_allow_html=True)
        
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>", unsafe_allow_html=True)
        st.markdown("<ul><li>Pressure9am : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>Pressure3pm : <i><b>{}</b> valeurs manquantes ({} %)</i></li></ul>".format(
            df_saved.Pressure9am.isna().sum(),
            ((100*df_saved.Pressure9am.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.Pressure3pm.isna().sum(),
            ((100*df_saved.Pressure3pm.isna().sum())/df_saved.shape[0]).round(1)), unsafe_allow_html=True)
        
        #Visualisation graphique
        st.markdown("<h4><u><font color = 'navy'>Visualisation graphique :</u></h4>", unsafe_allow_html=True)
        
        #histogrammes
        fig_pression_1 = plt.figure(figsize = (12,12))
        plt.suptitle("Distribution des variables Pressure9am et Pressure3pm", fontsize = 22, fontweight = 'bold')
        
        plt.subplot(221)
        plt.title("Pression atmosphérique\n (relevée à 9h)", fontsize = 18, pad = 20)
        sns.histplot(x = "Pressure9am", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 25)
        plt.xticks(size = 14)
        plt.xlabel("Pressure9am", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(222)
        plt.title("Pression atmosphérique\n (relevée à 15h)", fontsize = 18, pad = 20)
        sns.histplot(x = "Pressure3pm", data = df, hue = "RainToday", hue_order= ["Yes", "No"], bins = 25)
        plt.xticks(size = 14)
        plt.xticks()
        plt.xlabel("Pressure3pm", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Compte", size = 16)
        plt.legend(labels=["Pas de pluie","Pluie"], facecolor = 'white')
        
        plt.subplot(223)
        plt.title("Corrélation entre la pression atmosphérique\n à 9h et à 15h", fontsize = 18, pad = 20)
        sns.scatterplot("Pressure9am", "Pressure3pm", data = df)
        plt.xticks(size = 14)
        plt.xlabel("Pressure9am", size = 16)
        plt.yticks(size = 14)
        plt.ylabel("Pressure3pm", size = 16)
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_pression_1);
        
        #boxplots
        fig_pression_2 = plt.figure(figsize = (12,6))
        plt.suptitle("Distribution comparée en fonction de la pluie", fontsize = 22, fontweight = 'bold')
        
        plt.subplot(121)
        plt.title("Pression atmosphérique\n (relevée à 9h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "Pressure9am", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("Pressure9am (hpa)", size =16)
        
        plt.subplot(122)
        plt.title("Pression atmosphérique\n (relevée à 15h)", fontsize = 18, pad = 20)
        sns.boxplot(y = "Pressure3pm", data = df, x = "RainToday", palette = ["orange", "lightblue"], width=0.3)
        plt.xticks(ticks = [0,1], labels = ["Pas de pluie", "Pluie"], size = 16)
        plt.xlabel(None)
        plt.yticks(size = 14)
        plt.ylabel("Pressure3pm (km/h)", size =16)
        
        plt.tight_layout(pad=3)
        st.pyplot(fig_pression_2);
        
    ##PLUIE.    
        
    elif var_a_afficher == "Pluie":
    
        st.markdown("<h2><u><center>RainToday, RainTomorrow</center></u></h2>", unsafe_allow_html=True)
        
        #Définition des variables
        st.markdown("<h4><u><font color = 'navy'>Définition des variables :</u></h4>", unsafe_allow_html=True)
        st.markdown("""<ul><li><b>RainToday</b> est définie comme <i>'Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0'</i>, soit 1 pour un jour de pluie (défini pour une hauteur de précipitation > 1 mm, 0 sinon.</li>
        <li><b>RainTomorrow</b> est définie comme <i>'The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".'</i>, soit hauteur de précipitations le jour suivant.</li>
        </ul>""", unsafe_allow_html=True)
        st.markdown("Attention, les définitions fournies avec le dataset sont inexactes. Il s'agit pour les deux de variables catégorielles binaires, codées 'Yes' ou 'No' suivant s'il pleuvait ou non les jours considérés.", unsafe_allow_html=True)
        
        #Description des variables
        st.markdown("<h4><u><font color = 'navy'>Description des variables :</u></h4>",unsafe_allow_html=True)
        st.markdown("<h5><i>RainToday :</i></h5>", unsafe_allow_html=True)
        
        st.markdown("""<div style = 'text-align : justify'>Sur l'ensemble des données, il y a <b>{} jours de pluie</b> (<b>{} %</b>) et <b>{} jours sans pluie</b> (<b>{} %</b>).</div>
        """.format(df[df["RainToday"] == "Yes"]["RainToday"].count(),
                   (100*(df[df["RainToday"] == "Yes"]["RainToday"].count())/df.shape[0]).round(1),
                   df[df["RainToday"] == "No"]["RainToday"].count(),
                   (100*(df[df["RainToday"] == "No"]["RainToday"].count())/df.shape[0]).round(1)), unsafe_allow_html=True)
        
        st.markdown("<h5><i><br/>RainTomorrow :</i></h5>", unsafe_allow_html=True)   
        st.markdown("Il s'agit de la variable à prédire", unsafe_allow_html=True)
        
        #Valeurs manquantes
        st.markdown("<h4><u><font color = 'navy'>Valeurs manquantes :</u></h4>", unsafe_allow_html=True)
        st.markdown("<ul><li>RainToday : <i><b>{}</b> valeurs manquantes ({} %)</i></li> <li>RainTomorrow : <i><b>{}</b> valeurs manquantes ({} %)</i></li></ul>".format(
            df_saved.RainToday.isna().sum(),
            ((100*df_saved.RainToday.isna().sum())/df_saved.shape[0]).round(1),
            df_saved.RainTomorrow.isna().sum(),
            ((100*df_saved.RainTomorrow.isna().sum())/df_saved.shape[0]).round(1)),unsafe_allow_html=True)     
        
elif nav == "Les filouteries de Lise aka Kangooroo-Girl":
     st.image(
            "https://st.depositphotos.com/1033604/2008/i/600/depositphotos_20086857-stock-photo-kangaroo-posing-very-much-like.jpg",
            width=800, # Manually Adjust the width of the image as per requirement
        )
    

elif nav == "🐰":
    st.image(
            "https://m1.quebecormedia.com/emp/emp/matrixcbbb9deff-9126-47fc-b1fe-85c645ff9b6c_ORIGINAL.jpg?impolicy=crop-resize&x=0&y=0&w=0&h=0&width=925&height=925",
            width=800, # Manually Adjust the width of the image as per requirement
        )