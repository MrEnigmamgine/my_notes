import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_wordcloud(input , figsize=[12,9]):
    
    from wordcloud import WordCloud
    cloud = WordCloud()
    input_type = type(input)
    if input_type == str:
        cloud.generate(input)
    
    if input_type == list:
        input = pd.Series(input)
        input_type = type(input)

    if input_type == pd.core.series.Series:
        input = input.value_counts().to_dict()
        input_type = type(input)
    
    if input_type == dict:
        output = {}
        for k, v in input.items():
            if type(k) != str:
                if type(k) == tuple:
                    output[' '.join(k)] = v
                else:
                    output[str(k)] = v
            else:
                output[k] = v
        cloud.generate_from_frequencies(output)
    
    plt.figure(figsize=figsize)
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()

