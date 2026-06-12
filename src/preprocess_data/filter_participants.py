import pandas as pd
import os

def run(layout='qwerty', wpm=80, country='US', fingers='9-10', subset_n=None):
    df = pd.read_csv('data/raw/metadata_participants.txt', sep='\t')
    
    if layout and str(layout).lower() != 'any': 
        df = df[df['LAYOUT'] == layout]
    if wpm and str(wpm).lower() != 'any': 
        df = df[df['AVG_WPM_15'] > float(wpm)]
    if country and str(country).lower() != 'any': 
        df = df[df['COUNTRY'] == country]
    if fingers and str(fingers).lower() != 'any': 
        df = df[df['FINGERS'] == fingers]
    if subset_n: 
        df = df.head(subset_n)

    name_parts = [str(x).lower() for x in [layout, wpm, country, fingers] if x]
    filename = f"{'_'.join(name_parts)}.txt"
    output_path = os.path.join('data/interim/participants', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    return output_path

if __name__ == "__main__":
    run(layout='qwerty', wpm=80, country='US', fingers='any')