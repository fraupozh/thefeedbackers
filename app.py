from dotenv import load_dotenv
load_dotenv()


import os

def create_keyfile_dict():
    variables_keys = {
        "type": os.environ.get("SHEET_TYPE"),
        "project_id": os.environ.get("SHEET_PROJECT_ID"),
        "private_key_id": os.environ.get("SHEET_PRIVATE_KEY_ID"),
        "private_key": os.environ.get("SHEET_PRIVATE_KEY"),
        "client_email": os.environ.get("SHEET_CLIENT_EMAIL"),
        "client_id": os.environ.get("SHEET_CLIENT_ID"),
        "auth_uri": os.environ.get("SHEET_AUTH_URI"),
        "token_uri": os.environ.get("SHEET_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.environ.get("SHEET_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.environ.get("SHEET_CLIENT_X509_CERT_URL")
    }
    return variables_keys


# libraries
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="The Feedbackers",
    layout="wide",
)

# dashboard title
st.title("The Feedbackers")


@st.cache
def get_data():
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(create_keyfile_dict(), scope)
    client = gspread.authorize(creds)
    spreadsheets = ['test_feedback_responces']
    
    #Open the Spreadsheet or several Spreadsheets
    for spreadsheet in spreadsheets:
        sh = client.open(spreadsheet)

    #Get all values in the first worksheet
    worksheet = sh.get_worksheet(0)
    data = worksheet.get_all_values()
    df = pd.DataFrame(columns = (data[0]))
    for i in range(1,len(data)):
        df.loc[len(df)] = data[i]
        
    return df
                      
df = get_data()

df = df.replace(r'^\s*$', np.nan, regex=True)


# metrics

df_wait = df.loc[:, ['Termin', 'Meine Wartezeit war angemessen', 'Ich habe … Minuten gewartet']].copy()
df_wait.rename(columns={'Ich habe … Minuten gewartet': 'Wartezeit (Minuten)'}, inplace=True)
df_wait = df_wait.dropna()
df_wait['Wartezeit (Minuten)'] = df_wait['Wartezeit (Minuten)'].astype(int)
time_mean = df_wait.groupby(['Termin']).mean('Wartezeit (Minuten)')

kpi1, kpi2 = st.columns(2)


kpi1.metric(
    label="Durchschnittliche Wartezeit mit Termin ⏳",
    value=f"{round(time_mean._get_value('Ich hatte einen Termin', 'Wartezeit (Minuten)'))} min"
)

kpi2.metric(
    label="Durchschnittliche Wartezeit ohne Termin ⏳",
    value=f"{round(time_mean._get_value('Ich hatte KEINEN Termin', 'Wartezeit (Minuten)'))} min"
)

#dataframe with likert scale questions only

dff = df.iloc[:, [0, 6, 7, 9, 10, 11, 17, 18, 19]].copy()

dff.set_index('Timestamp', inplace=True)
df_count = dff.apply(pd.Series.value_counts)
df_count = df_count.fillna(0).astype(int)

list_of_counts = df_count.transpose().values.tolist()

category_names = ['Trifft nicht zu', 'Trifft teilweise nicht zu', 'Neutral', 'Trifft teilweise zu', 'Trifft zu']

questions = list(df_count.columns)

results = dict(zip(questions, list_of_counts))

def likert(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    middle_index = data.shape[1]//2
    offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index]/2
    
    
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))
    

    fig, ax = plt.subplots()
    

    # Plot Bars
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths - offsets
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
        
        
    # Add Zero Reference Line
    #ax.axvline(0, linestyle=':', color='black', alpha=.25)
    
    # X Axis
    ax.set_xlim(-40, 40)
    ax.set_xticks(np.arange(-40, 41, 10))
    ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))
    ax.set_xlabel('Anzahl der Antworten')
    
    # Y Axis
    ax.invert_yaxis()
    
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Ledgend
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')


    return fig, ax


fig, ax = likert(results, category_names)



#Wartezeit

hue_order = ['Ja', 'Nein']

fig3 = sns.catplot(data=df_wait, x="Termin", y="Wartezeit (Minuten)", hue="Meine Wartezeit war angemessen", kind="swarm", hue_order=hue_order)


#layout

col1, col2 = st.columns([2, 1])

with col1:
   st.subheader('Zufriedenheitsgrad unserer Patienten')
   st.write(f"{len(dff.index)} Antworten")
   st.pyplot(fig)

with col2:
   st.subheader("Wartezeit")
   st.write(f"{len(df_wait.index)} Antworten")
   st.pyplot(fig3)

#st.metric(label="Antwortenanzahl", value=len(df_wait.index))
#st.subheader(f"Wartezeit auf Basis von {len(df_wait.index)} Antworten")    

#Kontaktaufnahme preprocessing

df_contact = df.iloc[:, [0, 3, 4]].dropna().copy()
df_contact["C"] = np.ones((len(df_contact),), dtype=int)

table = pd.pivot_table(df_contact, values='C', index=['Ich habe Kontakt wie folgt aufgenommen:'],
                       columns=['Meine Kontaktaufnahme war erfolgreich'], aggfunc=np.sum)
#table = table.fillna(0).astype(int)

#data assignment
N = 7
labels = table.index
yes_data = list(table.loc[:,'Ja'])
no_data = list(table.loc[:,'Nein'])
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
x = np.arange(len(labels))

fig2, ax2 = plt.subplots()

#006600, #cc0000

p1 = ax2.bar(ind, yes_data, width, color='#006600', label='Erfolgreicher Kontakt')
p2 = ax2.bar(ind, no_data, width,
            bottom=yes_data, color='#CC0000', label='Erfolgloser Kontakt')

ax2.set_ylabel('Antwortenanzahl')
#ax.set_title('Kontaktaufnahme')
ax2.set_xticks(x, labels, fontsize='large', rotation=30, ha="right")
ax2.legend(ncol=2, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

# Label with label_type 'center' instead of the default 'edge'
ax2.bar_label(p1, label_type='center')
ax2.bar_label(p2, label_type='center')
ax2.bar_label(p2)

fig2.tight_layout()

#Pie chart for "Ich habe Kontakt wie folgt aufgenommen"
df2 = df.groupby(['Ich habe Kontakt wie folgt aufgenommen:'])['Ich habe Kontakt wie folgt aufgenommen:'].count()
df2 = df2.sort_values(ascending=False)

y = df2.values
mylabels = df2.index
mycolors = ['#FFF1C9', '#F7B7A3', '#EA5F89', '#7F0000', '#9B3192', '#57167E', '#2B0B3F']
percents = y * 100 / y.sum()

fig1, ax1 = plt.subplots()
ax1.pie(y, labels=mylabels, colors = mycolors, shadow = True, startangle=90)
ax1.axis('equal')
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', labels=['%s, %1.f %%' % (l, s) for l, s in zip(mylabels, percents)])

col1, col2 = st.columns([1.5, 2.5])

with col1:
   st.subheader('Kontaktaufnahme')
   st.write(f"{len(df_contact.index)} Antworten")
   st.pyplot(fig2)

with col2:
   st.subheader('Bevorzugte Kontaktkanäle unserer Patienten')
   st.write(f"{df2.sum()} Antworten")
   st.pyplot(fig1)


