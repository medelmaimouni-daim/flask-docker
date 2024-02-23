import os
import openai
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from api_key import api_key
from reportlab.lib import colors
from reportlab.platypus import Flowable
from reportlab.lib.units import mm, cm
import pandas as pd
from firebase_admin import db
import plotly.graph_objects as go
import random
import numpy as np



openai.api_key = api_key

def extract_audio_from_video(video_path, output_format='wav'):
    try:
        with VideoFileClip(video_path) as video:
            audio_path = video_path.rsplit('.', 1)[0] + '.' + output_format
            video.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def split_audio(audio_path, max_size_mb=25, format='wav'):
    try:
        audio = AudioSegment.from_file(audio_path, format=format)
    except FileNotFoundError as e:
        print(f"Error reading audio file: {audio_path}")
        raise e

    max_size_bytes = max_size_mb * 1024 * 1024
    # Reduce chunk length to 30 seconds
    chunk_length_ms = 1000 * 30  # 30 seconds

    parts = []
    start = 0
    end = chunk_length_ms
    while start < len(audio):
        chunk = audio[start:end]
        if len(chunk) > max_size_bytes:
            raise ValueError("A single chunk exceeds the maximum size limit. Consider using a lower bitrate audio format.")
        parts.append(chunk)
        start = end
        end += chunk_length_ms

    return parts


def transcribe_audio(file_path, language='fr'):
    try:
        with open(file_path, 'rb') as audio_file:
            transcript = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="text",
                language=language
            )
        if transcript:
                file_name = os.path.basename(file_path).rsplit('.', 1)[0]
                transcript_path = os.path.join('transcripts', f'{file_name}_transcript.txt')
                os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
                with open(transcript_path, 'w') as file:
                    file.write(transcript)
        return transcript
    except FileNotFoundError as e:
        print(f"Error opening audio chunk for transcription: {file_path}")
        raise e



class CustomLinearGauge(Flowable):
    def __init__(self, score, mean, width=50*mm, height=10*mm, gradient_steps=10):
        Flowable.__init__(self)
        self.score = score
        self.mean = mean
        self.width = width
        self.height = height
        self.gradient_steps = gradient_steps

    def draw(self):
        # Draw the background of the gauge
        self.canv.setStrokeColor(colors.black)
        self.canv.setFillColor(colors.whitesmoke)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=1)

        # Draw the pseudo-gradient score indicator
        score_indicator_width = (self.score / 10.0) * self.width  # Assuming the score is out of 10
        for step in range(self.gradient_steps):
            r = (1.0 / self.gradient_steps) * step
            if self.score >= self.mean:
                fill_color = colors.green
            elif self.score > self.mean - 2:
                fill_color = colors.orange
            else:
                fill_color = colors.red
            mix_color = colors.linearlyInterpolatedColor(colors.whitesmoke, fill_color, 0, self.gradient_steps, step)
            step_width = score_indicator_width / self.gradient_steps
            self.canv.setFillColor(mix_color)
            self.canv.rect(step * step_width, 0, step_width, self.height, fill=1, stroke=0)

        # Draw the mean indicator line
        mean_indicator_position = (self.mean / 10.0) * self.width  # Calculate the mean indicator position
        self.canv.setStrokeColor(colors.black)
        self.canv.line(mean_indicator_position, 0, mean_indicator_position, self.height)

        # Label for the mean
        self.canv.setFont("TomaSans-Regular", 6)
        self.canv.setFillColor(colors.black)
        mean_label = f"moyenne du cours = {self.mean}"
        self.canv.drawRightString(mean_indicator_position + 2*mm, self.height + 2*mm, mean_label)

        # Adding score label directly below the score indicator
        self.canv.setFont("TomaSans-Regular", 7)
        score_label = f"Score: {self.score}"
        score_label_width = self.canv.stringWidth(score_label, "TomaSans-Regular", 7)
        
        # Calculate x position for the score label to appear centered below the score indicator
        score_indicator_x_position = (self.score / 10.0) * self.width - score_label_width / 2

        # Ensure the score label is within the bounds of the gauge
        score_indicator_x_position = max(score_indicator_x_position, 0)
        score_indicator_x_position = min(score_indicator_x_position, self.width - score_label_width)

        # Set y position for the score label to appear just below the gauge
        score_label_y_position = -4*mm  # Adjust as needed

        self.canv.drawString(score_indicator_x_position - 2*mm, score_label_y_position, score_label)



def delete_all_files_in_folder(folder_path):
    # Define the video file extensions you want to delete
    
    
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Iterate through all files
    for file in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, file)        
        os.remove(file_path)
        print(f"Deleted {file_path}")


def sanitize_keys(data):
    if isinstance(data, dict):
        sanitized_data = {}
        for key, value in data.items():
            # Replace problematic characters in keys
            sanitized_key = key.replace('/', '_').replace('.', '_')
            # Recursively sanitize nested dictionaries
            sanitized_value = sanitize_keys(value)
            sanitized_data[sanitized_key] = sanitized_value
    elif isinstance(data, list):
        sanitized_data = [sanitize_keys(item) for item in data]
    else:
        return data
    return sanitized_data

def get_next_session_number(ref, id_session):
    # Get the current sessions for the id_session
    current_sessions_ref = ref.child(id_session)
    current_sessions = current_sessions_ref.get()
    if current_sessions is None:
        # If there are no sessions yet, start with 1
        return 1
    else:
        # Extract session numbers and find the max
        session_numbers = [int(key.split('_')[1]) for key in current_sessions.keys() if key.startswith('session_')]
        if session_numbers:  # Check if the list is not empty
            max_session = max(session_numbers)
            return max_session + 1
        else:
            return 1  # In case there are no keys with the 'session_' prefix
        

def get_current_session_number(ref, id_session):
    # Get the current sessions for the id_session
    current_sessions_ref = ref.child(id_session)
    current_sessions = current_sessions_ref.get()
    if current_sessions is None:
        # If there are no sessions yet, indicate as such (e.g., return 0 or None)
        return 0  # or None, depending on how you want to handle this case
    else:
        # Extract session numbers and find the max
        session_numbers = [int(key.split('_')[1]) for key in current_sessions.keys() if key.startswith('session_')]
        if session_numbers:  # Check if the list is not empty
            max_session = max(session_numbers)
            return max_session
        else:
            return 0  # or None, if no valid 'session_' keys exist

def append_new_data_to_df(existing_df, incoming_data):
    new_row = pd.DataFrame([incoming_data], columns=existing_df.columns)
    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    return updated_df

def store_new_data_in_firebase(incoming_data, updated_df):
    # Construct a dictionary from the incoming data for Firebase
    new_data_dict = dict(zip(updated_df.columns, incoming_data))
    
    # Reference to store the new data
    # Here, assuming structure: sessions/{id_session}/{unique_entry_id}
    # Consider generating a unique entry ID or using .push() to let Firebase generate it
    ref = db.reference(f"scores").push()
    ref.set(new_data_dict)

def store_new_global_data_in_firebase(incoming_data, updated_df):
    # Construct a dictionary from the incoming data for Firebase
    new_data_dict = dict(zip(updated_df.columns, incoming_data))
    
    # Reference to store the new data
    # Here, assuming structure: sessions/{id_session}/{unique_entry_id}
    # Consider generating a unique entry ID or using .push() to let Firebase generate it
    new_data_dict = {k: int(v) if isinstance(v, np.int64) else v for k, v in new_data_dict.items()}
    ref = db.reference(f"global_scores").push()
    ref.set(new_data_dict)


def fetch_score_data_from_firebase():
    ref = db.reference("scores")
    data = ref.get()
    
    columns = ['id_session', 'theme', 'session'] + [f'score{i}' for i in range(1, 19)]
    if data is None:
        return pd.DataFrame(columns=columns)
    
    all_rows = []
    # Directly iterating over the fetched data assuming it's properly structured
    for entry_id, entry_data in data.items():
        # Constructing each row by extracting values in the specified order
        # Assuming entry_data is correctly structured and directly corresponds to the DataFrame's columns
        row = [entry_data[col] for col in columns if col in entry_data]  # Safely extract values, ignoring missing keys
        all_rows.append(row)
    
    df = pd.DataFrame(all_rows, columns=columns)
    return df

def fetch_global_scores_data_from_firebase():
    ref = db.reference("global_scores")
    data = ref.get()
    columns = ['id_session', 'theme', 'session_label', 'session'] + [f'score{i}' for i in range(1, 6)]
    if data is None:
        return pd.DataFrame(columns=columns)
    all_rows = []
    for entry_id, entry_data in data.items():
        row = [entry_data[col] for col in columns if col in entry_data]  # Safely extract values, ignoring missing keys
        all_rows.append(row) 
    df = pd.DataFrame(all_rows, columns=columns)
    return df

def generate_evaluation_prompt_scores_only(transcripts):
    """
    Generates an evaluation prompt for ChatGPT that expects a structured response with scores only,
    based on a given text, including detailed definitions for each type of discourse.

    Parameters:
    transcripts (str): The combined transcripts to be evaluated.

    Returns:
    str: The evaluation prompt.
    """
    prompt = f"""Sur la base du texte suivant : {transcripts}, évaluez-le uniquement avec des scores de 1 à 10 sans fournir de commentaires détaillés, selon les critères et définitions suivants :

- **Informationnel** : Ce discours se concentre sur les faits et l'objectivité, avec une structure souvent chronologique. Il explore la tension entre le message et le messager, ainsi que la motivation derrière la transmission d'informations.

- **Effet (Motivationnel)** : Orienté vers l'avenir, ce discours vise à influencer les décisions ou les comportements, privilégiant l'impact des faits sur la vie quotidienne des destinataires.

- **Occasion** : Centré sur le cœur et la narration personnelle, ce discours utilise des anecdotes pour créer de l'authenticité et susciter l'émotion.

- **Formation** : Il s'agit d'un discours interactif qui privilégie l'explication, la reformulation et la comparaison, en utilisant à la fois des preuves et de l'empathie pour faciliter l'apprentissage.

Formattez votre réponse comme suit : {{
"Informationnel" : score,
"Effet" : score,
"Occasionnel" : score,
"Formation" : score
}}

Remplacez [score] par votre évaluation pour chaque critère."""

    return prompt



def create_radar_chart_from_df(df, color):
    # Extract scores and categories from the DataFrame
    scores = df['Score'].tolist() + [df['Score'].tolist()[0]]
    categories = df['Critere'].tolist() + [df['Critere'].tolist()[0]]
    
    # Create radar chart
    fig = go.Figure(
        go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='none',
            line=dict(color=color),
            fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)')
        )
    )

    # Set tickvals to mark every integer from 0 to 10 and other layout configurations
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                linecolor="#bcbcbc",
                gridcolor="#bcbcbc",
                range=[0, 10],
                tickvals=list(range(11))
            ),
            angularaxis=dict(
                linecolor="#bcbcbc",
                gridcolor="#bcbcbc"
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            itemsizing='constant'  # Ensure consistent legend marker size
        )
    
    )
    
    return fig


 # Sample categories
categories = ['SIMPLICITÉ DU DISCOURS', 'PRISE EN COMPTE DU PUBLIC', 'POSITIVITÉ DU DISCOURS', 'ASSERTIVITÉ', 'PÉDAGOGIE']

# Function to add radar chart from df
def add_radar_chart_from_df(df, categories, color, name):
    # Extract the scores for the given categories
    df['Critere'] = categories
    scores = [df[df['Critere'] == critere]['Score'].values[0] for critere in categories]
    # Append the first score at the end to close the loop
    scores.append(scores[0])
    
    # Append the first category at the end to close the loop
    categories_completed = list(categories) + [categories[0]]
    
    # Create the radar chart
    return go.Scatterpolar(
        r=scores,
        theta=categories_completed,
        fill='none',
        line=dict(color=color),
        fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)'),
        name=name  # Set the trace name
    )

# Function to add radar chart with fixed parameters
def add_fixed_radar_chart(scores, categories, color, name):
    # Ensure the scores list is complete by adding the first score to the end
    completed_scores = scores + [scores[0]]
    
    # Complete the categories by adding the first category to the end
    completed_categories = categories + [categories[0]]
    
    # Create the radar chart
    return go.Scatterpolar(
        r=completed_scores,
        theta=completed_categories,
        fill='none',
        line=dict(color=color),
        fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)'),
        name=name  # Set the trace name
    )


# Function to generate a random RGB color
def generate_random_color():
    return f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'

def generate_radar_charts(df):
    categories_score = ['score1', 'score2', 'score3', 'score4', 'score5']
    fig = go.Figure()
    
    for index, row in df.iterrows():
        scores = [row[score] for score in categories_score]
        scores.append(scores[0])  # Close the loop for radar chart
        
        # Determine color based on 'session'
        color = 'rgb(243,146,0)' if row['session'] == 1 else generate_random_color()
        
        # Add radar chart for each session
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories + [categories[0]],
            fill='none',
            name=row['session_label'],
            line=dict(color=color),
            fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)')
        ))

    # Set tickvals to mark every integer from 0 to 10 and other layout configurations
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                linecolor="#bcbcbc",
                gridcolor="#bcbcbc",
                range=[0, 10],
                tickvals=list(range(11))
            ),
            angularaxis=dict(
                visible=True,
                linecolor="#bcbcbc",
                gridcolor="#bcbcbc"
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            itemsizing='constant'
        )
    
    )
    
    return fig



