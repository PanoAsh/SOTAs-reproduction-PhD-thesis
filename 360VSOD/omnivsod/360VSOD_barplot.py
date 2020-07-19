import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set(style="whitegrid")

index = ['CanNotStopTheFeeling-Choir', 'UseSomebody', 'João-e-Maria', 'MrChildren', 'Improvising', 'Riptide',
             'CanNotStopTheFeeling-Solo', 'Crazy', 'Soprano', 'FrenchConversation-3', 'FrenchConversation-1',
             'ItalianConversation', 'Interview-2', 'Football', 'FrenchConversation-2', 'SpanishConversation',
             'Exhibition', 'BadmintonRackets', 'Racing', 'PianoTeaching-2', 'Grove-1', 'Lawn', 'PianoTeaching-1',
             'Debate', 'GameOverMan', 'VoiceComic', 'Grove-2', 'Skiing', 'Audition', 'Carré-de-Flûte', 'Prelude',
             'Cannon', 'Bluesaholic', 'Trumpet-Solo', 'Violin-Solo', 'Drums', 'PianoSaxophone', 'Badminton', 'Beach',
             'Engine', 'Tennis', 'Train', 'Bicycling', 'Carriage', 'Road', 'WaterFall', 'Dogs-2', 'Sliding', 'Dogs-1',
             'ChineseSpeaking', 'FrenchSpeaking', 'Soliloquizing', 'Military', 'Surfing', 'AudiIntro-1', 'Greeting',
             'AudiIntro-4', 'Coaching', 'Breakfast', 'TelephoneTech', 'Guiding', 'AudiIntro-2', 'AudiIntro-3',
             'SuperPower', 'Advertising', 'WaitingForAnInterview', 'EllenShow', 'Interview-1', 'HuPoTang']
keyFrames = [101, 130, 134, 145, 96, 201, 140, 165, 138, 220, 150, 102, 161, 150, 165, 137, 146, 136, 271, 103,
                 145, 156, 154, 150, 157, 147, 150, 140, 173, 105, 89, 125, 153, 130, 77, 129, 160, 150, 117, 125, 78,
                 140, 145, 125, 125, 130, 100, 53, 150, 130, 161, 125, 218, 85, 192, 107, 101, 149, 150, 138, 115, 164,
                 201, 145, 147, 305, 144, 347, 140]
objMasks = [387, 650, 402, 290, 428, 1005, 560, 330, 276, 220, 285, 199, 322, 300, 330, 265, 292, 402, 542, 206,
                280, 312, 308, 598, 418, 588, 314, 454, 692, 420, 445, 375, 459, 130, 77, 188, 320, 702, 186, 125, 297,
                280, 145, 238, 125, 130, 100, 63, 150, 130, 161, 125, 218, 85, 192, 214, 101, 149, 150, 138, 575, 164,
                201, 290, 147, 830, 144, 1041, 560]
subClasses = ['choir', 'vocal solo', 'dialogue', 'group chat', 'musical ensemble', 'solo', 'duet',
            'miscellaneous sources', 'male narrator', 'female narrator', 'f&m narrators']
numVids = [3, 6, 14, 5, 5, 2, 2, 12, 14, 4, 2]

def sns_official():
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    # Load the example car crash dataset
    crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

    # Plot the total crashes
    sns.set_color_codes('pastel')
    sns.barplot(x='total', y='abbrev', data=crashes, label='Total', color='b')

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="alcohol", y="abbrev", data=crashes, label="Alcohol-involved", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="", xlabel="Automobile collisions per billion miles")
    sns.despine(left=True, bottom=True)

    plt.show()

def self_plot():
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    speed = [0.1, 17.5, 40, 48, 52, 69, 88]
    lifespan = [2, 8, 70, 1.5, 25, 12, 28]
    index = ['snail', 'pig', 'elephant',
             'rabbit', 'giraffe', 'coyote', 'horse']
    df = pd.DataFrame({'speed': speed, 'lifespan': lifespan, 'index': index}, index=index)

    # Plot the total crashes
    sns.set_color_codes('pastel')
    sns.barplot(x='speed', y='index', data=df, label='speed', color='b')

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x='lifespan', y='index', data=df, label='lifespan', color='b')

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 100), ylabel="Hello again !", xlabel="Hello Seaborn !")
    sns.despine(left=True, bottom=True)

    plt.show()

def sod_plot():
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(18, 15))

    SODDataSet = pd.DataFrame({'index': index, 'keyFrames': keyFrames, 'objMasks': objMasks}, index=index).\
        sort_values('objMasks', ascending=False)

    sns.set_color_codes('pastel')
    sns.barplot(x='objMasks', y='index', data=SODDataSet, label='Instance-level Masks', color='b')

    sns.set_color_codes("muted")
    sns.barplot(x='keyFrames', y='index', data=SODDataSet, label='Key Frames', color='b')

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 1200), ylabel="", xlabel="360VSOD Statistics")
    sns.despine(left=True, bottom=True)

    plt.savefig('BarPlot_1.png')

def subClass_plot():
    sc_keyFrm = [np.sum(keyFrames[:3]), np.sum(keyFrames[3:9]), np.sum(keyFrames[9:23]), np.sum(keyFrames[23:28]),
                 np.sum(keyFrames[28:33]), np.sum(keyFrames[33:35]), np.sum(keyFrames[35:37]), np.sum(keyFrames[37:49]),
                 np.sum(keyFrames[49:63]), np.sum(keyFrames[63:67]), np.sum(keyFrames[67:69])]
    sc_objMsk = [np.sum(objMasks[:3]), np.sum(objMasks[3:9]), np.sum(objMasks[9:23]), np.sum(objMasks[23:28]),
                 np.sum(objMasks[28:33]), np.sum(objMasks[33:35]), np.sum(objMasks[35:37]), np.sum(objMasks[37:49]),
                 np.sum(objMasks[49:63]), np.sum(objMasks[63:67]), np.sum(objMasks[67:69])]

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15))

    sc = pd.DataFrame({'subClass': subClasses, 'numVids': numVids, 'sc_keyFrm': sc_keyFrm, 'sc_objMsk': sc_objMsk},
                      index=subClasses).sort_values('sc_objMsk', ascending=False)

    sns.set_color_codes('pastel')
    sns.barplot(x='sc_objMsk', y='subClass', data=sc, label='Instance-level Masks', color='m')

    sns.set_color_codes("muted")
    sns.barplot(x='sc_keyFrm', y='subClass', data=sc, label='Key Frames', color='m')

    #sns.set_color_codes("deep")
    #sns.barplot(x='numVids', y='subClass', data=sc, label='Videos', color='m')

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 4500), ylabel="", xlabel="360VSOD Sub-classes")
    sns.despine(left=True, bottom=True)

    plt.savefig('BarPlot_2.png')


if __name__ == '__main__':
    subClass_plot()
    sod_plot()
    #sns_official()
    #self_plot()