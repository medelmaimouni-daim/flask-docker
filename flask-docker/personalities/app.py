from flask import Flask, render_template, request, redirect, url_for, session, make_response, flash, Response
import os
import pandas as pd
import re
import plotly.express as px
from werkzeug.utils import secure_filename
from utilis import *
from api_key import api_key
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import Spacer, PageBreak
import textwrap
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import db
import json
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.platypus import Flowable
import panel as pn
from moviepy.editor import VideoFileClip
from datetime import datetime
from io import StringIO



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'  # Add this to set the upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the folder exists
openai.api_key = api_key


# Fetch your service account credentials and initialize the Firebase app
cred = credentials.Certificate('static/firebase/serviceAccountKey.json')  # Replace with the path to your Firebase admin SDK JSON file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://personalities-fr-default-rtdb.europe-west1.firebasedatabase.app'  # Replace with your Realtime Database URL
})

# Create a Firestore client
firestore_db = firestore.client()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Query Firestore for user
        users_ref = firestore_db.collection('users')
        query = users_ref.where('username', '==', username).where('password', '==', password).limit(1)
        results = query.stream()
        user_found = None
        for result in results:
            user_found = result.to_dict()
            break
        if user_found:
            session['username'] = user_found.get('username')
            session['role'] = user_found.get('role')  # Assuming 'role' is stored in each user document
            return redirect(url_for('upload'))
        else:
            msg = 'Invalid username or password'
    # Assume delete_all_files_in_folder is defined elsewhere
    delete_all_files_in_folder('static/analysis/radars')
    delete_all_files_in_folder('static/analysis/txt-json')

    return render_template('login.html', msg=msg)

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if 'username' not in session or session.get('role') != 'superadmin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))  # Redirect non-superadmins

    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role', 'user')  # Default role is 'user'

        # Insert the new account into Firestore
        try:
            users_ref = firestore_db.collection('users')
            # Check if username already exists
            query = users_ref.where('username', '==', username).limit(1)
            results = list(query.stream())
            if results:
                flash('Username already exists.', 'danger')
            else:
                users_ref.add({
                    'full_name': full_name,
                    'email': email,
                    'username': username,
                    'password': password,
                    'role': role
                })
                flash('New account created successfully.', 'success')
        except Exception as e:
            flash(f'Error creating account: {e}', 'danger')

    return render_template('create_account.html')


ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}  # Allowed video formats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    # Retrieve session entries from Firebase
    ref = db.reference('sessions')  # Adjust if your Firebase path is different
    sessions_data = ref.get() or {}

    if request.method == 'POST':
        # Retrieve form data
        id_session = request.form.get('id_session')
        theme = request.form.get('theme')
        name = request.form.get('name')
        company = request.form.get('company')
        today = datetime.now().date()  # This gets today's date
        date = today.strftime('%Y-%m-%d')        # Check if the id_session is unique
        


        # Handle file upload
        file = request.files.get('video')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Store form data in session for use in other routes
            session['form_data'] = {'id_session': id_session, 'theme':  theme, 'name': name, 'company': company, 'date': date}

            return redirect(url_for('analyze', filename=filename))

        else:
            print('Invalid file type or no file selected')
            return redirect(request.url)
        
        
    # Provide an empty default if there's no data in session
    form_data = session.get('form_data', {'id_session': '', 'theme': '', 'name': '', 'company': '', 'date': ''})

    return render_template('upload.html', username=session['username'], form_data=form_data)

@app.route('/sessions')
def sessions():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Retrieve session entries from Firebase
    ref = db.reference('sessions')
    sessions_data = ref.get() or {}

    # Assuming each session entry is structured under 'sessions/{session_id}/session_{number}'
    # Adjust the data extraction logic as per your actual data structure
    data = []
    for session_id, sessions in sessions_data.items():
        for session_key, details in sessions.items():
            # Ensure 'details' is a dictionary and contains necessary keys
            if isinstance(details, dict) and all(key in details for key in ['session', 'theme', 'date']):
                data.append({
                    'ID' : "{cat}_{client}_{id_session}".format(cat = details['theme'], client = session_id, 
                                                                id_session = details['session']),
                    'ID Client': session_id,
                    'Session': details['session'],
                    'Catégorie': details['theme'],
                    'Date': details['date']
                })
    base_url = '/download-hist-pdf/'  # Your Flask route for PDF downloads
    # Create the DataFrame
    df = pd.DataFrame(data)
    df['ID'] = df['ID'].apply(lambda x: f'<a href="{base_url}{x}">{x}</a>')
    html_table_session = df.to_html(classes='data-table', index=False, escape=False)  # Convert DataFrame to HTML table

    # Pass the HTML table to the sessions template
    return render_template('sessions.html', html_table_session=html_table_session)



@app.route('/analyze/<filename>')
def analyze(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    session['file_name'] = filename.rsplit('.', 1)[0]  # Store the base file name without extension

    # Extract audio from the video
    audio_path = extract_audio_from_video(video_path)
    if audio_path is None:
        print('Error extracting audio from video')
        return redirect(url_for('upload'))

    # Check file size and either transcribe directly or split and transcribe
    file_size = os.path.getsize(audio_path)
    max_size_bytes = 25 * 1024 * 1024
    if file_size <= max_size_bytes:
        try:
            transcript = transcribe_audio(audio_path)
            
        except Exception as e:
            print(f'Error during transcription: {e}')
            return redirect(url_for('upload'))
        # Proceed with semantic analysis
    else:
        try:
            audio_chunks = split_audio(audio_path)
        except Exception as e:
            print(f'Error splitting audio file: {e}')
            return redirect(url_for('upload'))

        all_transcripts = []
        for i, chunk in enumerate(audio_chunks):
            chunk_path = f'temp_chunk_{i}.wav'
            chunk.export(chunk_path, format='wav')
            try:
                transcript = transcribe_audio(chunk_path)
            except Exception as e:
                os.remove(chunk_path)
                print(f'Error transcribing chunk: {e}')
                continue
            os.remove(chunk_path)
            all_transcripts.append(transcript)
        
        combined_transcript = '\n'.join(all_transcripts)
        transcript = combined_transcript
        file_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        transcript_path = os.path.join('transcripts', f'{file_name}_transcript.txt')
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        with open(transcript_path, 'w') as f:
            f.write(combined_transcript)
    file_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    transcript_path = os.path.join('transcripts', f'{file_name}_transcript.txt')

    def get_video_length(video_path):
        with VideoFileClip(video_path) as video:
            return video.duration

    video_length = get_video_length(video_path)

    with open(transcript_path, 'r') as file:
        combined_transcripts = file.read()
    form_data = session.get('form_data', {})
    theme = form_data['theme']
    # Perform Semantic Analysis using GPT-4
    ppp_axes =  generate_evaluation_prompt_scores_only(combined_transcripts)
    # Define the system message
    system_msg = f"""Tu es un expert en linguistique avec une expertise en analyse sémantique et en calcul de KPIs pour différents niveaux d\'audience comme TEDx, politique, et chef d\'entreprise. Le thème du pitch est: {theme}"""
    
    response_axes = openai.ChatCompletion.create(model="gpt-4-1106-preview",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": ppp_axes}])
    response_axes_text = response_axes.choices[0]['message']['content'].strip().replace("\n", '')
    response_axes_json = json.loads(response_axes_text)

    user_msg = f"""Salut, le texte suivant : "[ {combined_transcripts} ]" est la transcription d'un pitch présenté devant un comité d'experts. Effectue une analyse sémantique par session, en prenant en compte que le public est spécialisé, basée sur les critères suivants avec une échelle de 1 à 10, où 1 représente le niveau le plus bas et 10 le plus élevé. Les critères incluent:
1. SIMPLICITÉ DU DISCOURS: Moyenne du nombre de mots par phrase, structuration simple des phrases, organisation des idées, reformulation et itération, silences, vitesse moyenne globale (nb de mots par minute), et l'usage des mots clés.
2. PRISE EN COMPTE DU PUBLIC: Ratio du nombre de 'Je' / ('vous' + 'nous'), nombre d'occurrences impersonnelles, nombre de mots avant le premier 'vous' sans compter le 'Je' de la première phrase, personification.
3. POSITIVITÉ DU DISCOURS: Ratio nb de phrases positives / négatives (total, premier 1/3, second 1/3, et dernier 1/3).
4. ASSERTIVITÉ: Niveau d'agressivité, niveau de manipulation, non affirmation.
5. PÉDAGOGIE: Clarté et compréhension, engagement de l'audience, structure logique, innovation et originalité, usage de dispositifs rhétoriques.
Réponds sous forme d'un tableau (délimiteur ; ) 'Critère;score (une notation stricte);Score détaillé' avec des scores de 1 à 10, directement sans phrase d'introduction. Respecte la manière d'écriture des headers (Critère;Score;Score détaillé) et des critères en MAJ et la notation de score doit etre stricte."""

    response = openai.ChatCompletion.create(model="gpt-4-1106-preview",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": user_msg}])
    response_text = response.choices[0]['message']['content']
    rows = response_text.split('\n')

    # Remove the first row
    if rows:
        rows = rows[1:]

    # Define the standard header
    standard_header = "Critere;Score;Score detaillee"

    # Prepend the standard header to the remaining text
    response_text = standard_header + "\n" + '\n'.join(rows)
    processed_rows = []
    for row in response_text.split('\n'):
        # Find the positions of the first two semicolons
        semicolon_positions = [m.start() for m in re.finditer(';', row)]

        if len(semicolon_positions) > 2:
            # Get the position after the second semicolon
            position = semicolon_positions[1] + 1

            # Keep the part before the position as it is, replace semicolons with commas in the part after
            row = row[:position] + row[position:].replace(';', ',')

        processed_rows.append(row)
        # Recombine the processed rows back into a single string
        response_text = '\n'.join(processed_rows)
    prompt_text = f"""
Analyse le texte suivant en fournissant des scores et des commentaires pour chaque critère mentionné. Le texte, la transcription d'une video de {video_length} secondes, est: "{combined_transcripts}". Évalue chaque aspect sur une échelle de 1 à 10, où 1 représente le niveau le plus bas et 10 le plus élevé, et fournis un commentaire explicatif pour chaque score donné. Les réponses doivent être structurées pour permettre une extraction facile des scores et des commentaires.

Critères d'évaluation et questions :

- Simplicité/Clarté: 
  - Moyenne du nombre de mots par phrase: {{score}} ; {{commentaire}}
  - Structuration simple des phrases: {{score}} ; {{commentaire}}
  - Organisation des idées: {{score}} ; {{commentaire}}
  - Reformulation et itération: {{score}} ; {{commentaire}}
  - Vitesse moyenne globale (nombre de mots par minute): {{score}} ; {{commentaire}}
  - Utilisation des mots clés: {{score}} ; {{commentaire}}

- Vers le public:
  - Ratio du nombre de "Je" / ("vous" + "nous"): {{score}} ; {{commentaire}}
  - Nombre d'occurrences impersonnelles: {{score}} ; {{commentaire}}
  - Nombre de mots avant le premier "vous": {{score}} ; {{commentaire}}
  - Personnification: {{score}} ; {{commentaire}}

- Être positif:
  - Ratio nombre de phrases positives / négatives: {{score}} ; {{commentaire}}

- Assertif:
  - Niveau d'agressivité: {{score}} ; {{commentaire}}
  - Non affirmation: {{score}} ; {{commentaire}}

- Pédagogie:
  - Clarté et compréhension: {{score}} ; {{commentaire}}
  - Engagement de l'audience: {{score}} ; {{commentaire}}
  - Structure logique: {{score}} ; {{commentaire}}
  - Innovation et originalité: {{score}} ; {{commentaire}}
  - Usage de dispositifs rhétoriques: {{score}} ; {{commentaire}}

Fournis les scores et commentaires dans un format qui peut être facilement interprété comme une structure de données, par exemple: {{\"Simplicité/Clarté\": {{\"Moyenne du nombre de mots par phrase\": {{\"score\": 7, \"commentaire\": \"Le texte présente une structure claire mais pourrait être simplifié.\"}}, ...}}, ...}}.
""".strip()


    # Envoie la requête à OpenAI
    # Create a dataset using GPT
    response_detaille = openai.ChatCompletion.create(model="gpt-4-1106-preview",
                                                messages=[{"role": "system", "content": system_msg},
                                                {"role": "user", "content": prompt_text}])

    # Extraction du texte de la réponse
    response_detaille_text = response_detaille.choices[0]['message']['content'].strip()

    # Supposant que la réponse contient une structure JSON dans un bloc de code (```json ... ```)
    # Nous devons extraire ce JSON du texte de la réponse
    json_start = response_detaille_text.find('{')
    json_end = response_detaille_text.rfind('}') + 1
    response_json_str = response_detaille_text[json_start:json_end].replace('\n', '').replace('N/A', 'null')

    # Convertir la chaîne JSON en dictionnaire Python
    analyzed_data = json.loads(response_json_str)
    file_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_analysis.txt')
    axes_analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_analysis_axes.txt')
    detailled_analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_detailled.txt')
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, 'w') as f:
        f.write(response_text)
    with open(detailled_analysis_path, 'w', encoding='utf-8') as file:
        json.dump(analyzed_data, file, ensure_ascii=False)
    with open(axes_analysis_path, 'w', encoding='utf-8') as filee:
        json.dump(response_axes_json, filee, ensure_ascii=False)

    df = pd.read_csv(StringIO(response_text), sep=";")

    sanitized_detailled_data = sanitize_keys(analyzed_data)


    # Assuming 'session' is a dictionary-like object that contains 'form_data'
    form_data = session.get('form_data', {})

    # Extract data from the form_data
    id_session = form_data['id_session']
    theme = form_data['theme']
    date = form_data['date']
    # Assuming 'df' is your DataFrame and has been populated with your data
    # Convert the DataFrame to a dictionary format
    df_dict = df.to_dict(orient='records')

    # Reference to the sessions in the database
    sessions_ref = db.reference('sessions')

    # Get the next session number for the given id_session
    next_session_number = get_next_session_number(sessions_ref, id_session)

    # Combine all data into a single dictionary to store in Firebase
    data_to_store = {
        'id_client': id_session,
        'theme': theme,
        'date': date,
        'ppp_axes' : response_axes_json,
        'df_elements': df_dict,
        'detailled_data': sanitized_detailled_data,
        'session': next_session_number  # Add the session number here
    }

    # getting all scores
    scores_list = [id_session, theme, next_session_number]
    global_scores_list = [id_session, theme, "session_{i}".format(i = next_session_number), next_session_number]

    for elt in sanitized_detailled_data.keys():
        for element in sanitized_detailled_data[elt]:
            scores_list.append(sanitized_detailled_data[elt][element]['score'])

    for critere in df['Critere']:
        global_scores_list.append(df[df['Critere'] ==critere]['Score'].values[0])

    # Fetch existing DataFrame from Firebase
    existing_score_df = fetch_score_data_from_firebase()   

    existing_global_score_df = fetch_global_scores_data_from_firebase() 

    # Append new data to the existing DataFrame
    updated_df = append_new_data_to_df(existing_score_df, scores_list)
    # Append new data to the existing DataFrame
    updated_global_df = append_new_data_to_df(existing_global_score_df, global_scores_list)

    # Store the data in Firebase
    store_new_data_in_firebase(scores_list, updated_df)
    store_new_global_data_in_firebase(global_scores_list, updated_global_df)

    # Save the combined data in Firebase Realtime Database using id_session and session number as the key
    session_ref = sessions_ref.child(f'{id_session}/session_{next_session_number}')
    session_ref.set(data_to_store)

    print(f'Data for session {id_session} stored successfully in Firebase.') 
 
    delete_all_files_in_folder('uploaded_videos')
    delete_all_files_in_folder('transcripts')

    return redirect(url_for('results'))


@app.route('/results')
def results():
    if 'username' not in session:
        return redirect(url_for('login'))
    # Fetch results and pass to template

    file_name = session.get('file_name', 'default_file_name')  # Retrieve the file name or use a default
    analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_analysis.txt')
    detailled_analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_detailled.txt')
    axes_analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_analysis_axes.txt')
    df = pd.read_csv(analysis_path, sep=';', encoding='latin1', index_col=False)
    html_table = df.to_html(classes='data-table', index=False)  # Convert DataFrame to HTML table
    with open(axes_analysis_path,  'r', encoding='utf-8') as file1:
        axes_json = json.load(file1)
    axes_df = pd.DataFrame(list(axes_json.items()), columns=['Critere', 'Score'])
    # Create the radar plot
 
    axes_fig = create_radar_chart_from_df(axes_df, 'rgb(243,146,0)')
    # Initialize figure
    fig = go.Figure()

    # Add the radar chart based on df with a name
    fig.add_trace(add_radar_chart_from_df(df, categories, 'rgb(243,146,0)',  "Apprenti"))

    # Names for the additional radar charts
    names = ['Etalon TED', 'Etalon Politique', 'Etalon Entreprise']

    # Fixed scores for the additional radar charts
    fixed_scores_list = [
        [5.2, 5.6, 6.4, 6.6, 8, 5.2],
        [4.8, 5.6, 4.4, 6.2, 5.8, 4.8],
        [5.4, 6, 6.8, 7, 6.8, 5.4]
    ]

    # Colors for the additional radar charts
    colors = ['rgb(128, 128, 0)', 'rgb(72, 209, 204)', 'rgb(0, 78, 146)']

    # Add three additional radar charts with fixed parameters and names
    for scores, color, name in zip(fixed_scores_list, colors, names):
        fig.add_trace(add_fixed_radar_chart(scores, categories, color, name))

    # Update the layout with a range from 0 to 10 for the radial axis
    # And set tickvals to mark every integer from 0 to 10
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
    radar_plot_directory = os.path.join('static','analysis', 'radars')
    # Create the directory and any necessary intermediate directories
    os.makedirs(radar_plot_directory, exist_ok=True)

    # Now you can safely save files in this directory
    # Correctly set the radar plot path
    radar_plot_path = os.path.join('analysis', 'radars', f'{file_name}.png').replace('\\', '/')
    axes_radar_plot_path = os.path.join('analysis', 'radars', f'{file_name}_axes.png').replace('\\', '/')

    fig.write_image(os.path.join('static', radar_plot_path.replace('/', os.sep)))
    axes_fig.write_image(os.path.join('static', axes_radar_plot_path.replace('/', os.sep)))

    # Read the response text
    with open(analysis_path, 'r') as file:
        response_text = file.read()



    # Assuming 'session' is a dictionary-like object that contains 'form_data'
    form_data = session.get('form_data', {})
    # Reference to the sessions in the database
    sessions_ref = db.reference('sessions')

    # Get the next session number for the given id_session
    current_session_number = get_current_session_number(sessions_ref, form_data['id_session'])
    return render_template('results.html', response_text=response_text, radar_plot_path=radar_plot_path,
                           axes_radar_plot_path = axes_radar_plot_path, html_table = html_table,  
                           form_data=form_data,  current_session_number = current_session_number)

@app.route('/download-pdf')
def download_pdf():
    if 'username' not in session:
        return redirect(url_for('login'))

    file_name = session.get('file_name', 'default_file_name')
    form_data = session.get('form_data', {'name': 'N/A', 'company': 'N/A', 'date': 'N/A'})
    axes_radar_plot_path = os.path.join('static', 'analysis', 'radars', f'{file_name}_axes.png').replace('\\', '/')
    global_radar_plot_path = os.path.join('static', 'analysis', 'radars', f'{file_name}_global.png').replace('\\', '/')
    id_session = form_data['id_session']
    theme = form_data['theme']
    # Reference to the sessions in the database
    ref = db.reference("sessions")
    data = ref.get()

    # Get the next session number for the given id_session
    current_session_number = get_current_session_number(ref, id_session)

    data = data[id_session]["session_{i}".format(i = current_session_number)]

    df = pd.DataFrame(data['df_elements'])  
    
    detailled_analysis_path = os.path.join('static', 'analysis', 'txt-json', f'{file_name}_detailled.txt')
    score_df = fetch_score_data_from_firebase()
    global_score_df = fetch_global_scores_data_from_firebase()

    mean_score_df = score_df.groupby(['id_session', 'theme']).mean()
    mean_global_score_df = global_score_df.groupby(['id_session', 'theme']).mean()
   

    global_score_df = fetch_global_scores_data_from_firebase()
    global_score_df = global_score_df[global_score_df['id_session'] == id_session]
    if global_score_df.shape[0] > 1:
        global_fig = generate_radar_charts(global_score_df)
        global_fig.write_image(os.path.join( global_radar_plot_path.replace('/', os.sep)))


    # Load the detailed analysis data from JSON
    with open(detailled_analysis_path, 'r', encoding='utf-8') as json_file:
        detailed_data = json.load(json_file)

    # Wrap text
    last_col_index = len(df.columns) - 1
    for i in range(len(df)):
        df.iloc[i, last_col_index] = textwrap.fill(str(df.iloc[i, last_col_index]), width=50)

    pdf_buffer = io.BytesIO()
    
    # Register font variants
    pdfmetrics.registerFont(TTFont('WorkSans-Regular', 'static/fonts/WorkSans-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('WorkSans-Bold', 'static/fonts/WorkSans-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('WorkSans-Italic', 'static/fonts/WorkSans-Italic.ttf'))
    pdfmetrics.registerFont(TTFont('WorkSans-BoldItalic', 'static/fonts/WorkSans-BoldItalic.ttf'))

    # Assuming TomaSans is corrected to TTF; if OTF, ensure compatibility or convert to TTF
    pdfmetrics.registerFont(TTFont('TomaSans-Regular', 'static/fonts/TomaSans-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('TomaSans-Bold', 'static/fonts/TomaSans-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('TomaSans-Italic', 'static/fonts/TomaSans-Italic.ttf'))
    pdfmetrics.registerFont(TTFont('TomaSans-BoldItalic', 'static/fonts/TomaSans-BoldItalic.ttf'))

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontName = 'WorkSans-Regular'  # Corrected font name
    normal_style.textColor = colors.HexColor('#000000')
    # Creating a centered and bold style for section titles
    title_style = ParagraphStyle(name='CenteredBold', parent=styles['Heading1'], alignment=1)

    # Adjusting the bold style to be used for sub-keys
    bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'], fontName='WorkSans-Bold')

    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm,
                            title="Analysis Results")

    logo_path = os.path.join('static', 'img/logo.png')
    logo = Image(logo_path, 4*cm, 2*cm)
    logo.hAlign = 'RIGHT'




    # Start with the static parts of your paragraph
    text_paragraphs_content = (f"<b>ID Client: </b> {id_session}<br/>"
                               f"<b>Session: </b> {current_session_number}<br/>"
                                f"<b>Catégorie de la session: </b> {theme}<br/>"
                                f"<b>Date: </b> {form_data['date']}<br/>")

    # Conditionally add 'Nom' and 'Entreprise' if they have non-empty values
    if form_data.get('name'):  # Checks if 'name' is not None or empty
        text_paragraphs_content += f"<b>Nom: </b> {form_data['name']}<br/>"
    if form_data.get('company'):  # Checks if 'company' is not None or empty
        text_paragraphs_content += f"<b>Entreprise: </b> {form_data['company']}<br/>"

    text_paragraphs = Paragraph(text_paragraphs_content)

    # The table data is a list containing one row with two cells:
    # The first cell contains the combined text, and the second cell contains the logo.
    
    table_data = [[text_paragraphs, logo]]

    # Create the table from the data
    header_table = Table(table_data, colWidths=[None, 4*cm])

    # Set the style for the table to align the logo to the right
    header_table.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN', (1,0), (1,0), 'RIGHT'),
    # Set row height, adjust the value as needed
    ('ROWHEIGHTS', (0,0), (-1,-1), 60),
    ]))

    # Add the table to the story
    story = [header_table]
    #story.append(Spacer(1, -2*cm))  # Adjust -1*cm to move the logo up by the desired amount
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Utilisation des 4 grands types de discours</b>", title_style))
    story.append(Spacer(1, 12))
    story.append(Spacer(1, 12))

    story.append(Image(axes_radar_plot_path, 18*cm, 12*cm))

    story.append(PageBreak())
    # Add a spacer after the header for spacing before the next content
    story.append(Spacer(1, 12))

    story.append(Spacer(1, -1*cm))  # Adjust to move the logo up by the desired amount
    story.append(logo)
    i = 1
    for critere in df['Critere']:
            # Inserting gauge visualization here
            mean_score = round(mean_global_score_df["score{i}".format(i=i)].values[0],2)
            i += 1
            score = df[df['Critere'] == critere]['Score'].values[0]
            gauge = CustomLinearGauge(score = score, mean = mean_score )

            # Combine sub_key description and gauge in a Table for inline appearance
            content_paragraph = Paragraph(f"<b>{critere}:</b>")
            table_data = [[content_paragraph, gauge]]
            content_table = Table(table_data, colWidths=[None, 200])
            story.append(content_table)

            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))
            # Adding commentaire below the gauge
            commentaire_paragraph = Paragraph(f"{df[df['Critere'] == critere]['Score detaillee'].values[0]}", styles['Normal'])
            story.append(commentaire_paragraph)
            story.append(Spacer(1, 12))  # Space after commentaire
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

   

    story.append(PageBreak())

    radar_plot_path = os.path.join('static', 'analysis', 'radars', f'{file_name}.png')
    #story.append(Spacer(1, -2*cm))  # Adjust -1*cm to move the logo up by the desired amount
    story.append(logo)
    story.append(Image(radar_plot_path, 18*cm, 12*cm))



    # Function to create gauge omitted for brevity; assume LinearGauge class is defined
    i = 1
    for key, value in detailed_data.items():
 
        story.append(PageBreak()) if story else None  # Add a PageBreak if not the first page
        story.append(Paragraph(key, title_style))

        story.append(Spacer(1, -2*cm))  # Adjust to move the logo up by the desired amount
        story.append(logo)
        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))
    
        for sub_key, sub_value in value.items():
            # Inserting gauge visualization here
            mean_score = round(mean_score_df["score{i}".format(i=i)].values[0],2)
            i += 1
            gauge = CustomLinearGauge(score = sub_value['score'], mean = mean_score )

            # Combine sub_key description and gauge in a Table for inline appearance
            content_paragraph = Paragraph(f"<b>{sub_key}:</b>")
            table_data = [[content_paragraph, gauge]]
            content_table = Table(table_data, colWidths=[None, 200])
            story.append(content_table)

            story.append(Spacer(1, 12))

            # Adding commentaire below the gauge
            commentaire_paragraph = Paragraph(f"{sub_value['commentaire']}", styles['Normal'])
            story.append(commentaire_paragraph)
            story.append(Spacer(1, 12))  # Space after commentaire
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

 

    if global_score_df.shape[0] > 1:
        story.append(PageBreak())
        story.append(Paragraph(f"<b>Titre à définir</b>", title_style))
        story.append(Spacer(1, -2*cm))  # Adjust to move the logo up by the desired amount
        story.append(logo)

        story.append(Image(global_radar_plot_path, 18*cm, 12*cm))
 
    doc.build(story)

    pdf = pdf_buffer.getvalue()
    pdf_buffer.close()

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={id_session}_{current_session_number}_{theme}_results.pdf'
    return response

@app.route('/download-hist-pdf/<session_id>')
def download_hist_pdf(session_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    # Example PDF generation using ReportLab
    theme = session_id.split("_")[0]
    id_session = session_id.split("_")[1]
    current_session_number = session_id.split("_")[2]

    ref = db.reference("sessions")
    data = ref.get()
    data = data[id_session]["session_{i}".format(i = current_session_number)]

    date = data['date']

    detailed_data = data['detailled_data']
    df = pd.DataFrame(data['df_elements'])
    axes_json = data['ppp_axes']
    radar_plot_directory = os.path.join('static','analysis', 'radars')
    # Create the directory and any necessary intermediate directories
    os.makedirs(radar_plot_directory, exist_ok=True)
    radar_plot_path = os.path.join('static', 'analysis', 'radars', f'{id_session}_{current_session_number}_{theme}_radar.png').replace('\\', '/')
    axes_radar_plot_path = os.path.join('static', 'analysis', 'radars', f'{id_session}_{current_session_number}_{theme}_axes.png').replace('\\', '/')
    global_radar_plot_path = os.path.join('static', 'analysis', 'radars', f'{id_session}_{current_session_number}_{theme}_global.png').replace('\\', '/')

    axes_df = pd.DataFrame(list(axes_json.items()), columns=['Critere', 'Score'])
    # Create the radar plot
 
    axes_fig = create_radar_chart_from_df(axes_df, 'rgb(243,146,0)')
    # Initialize figure
    fig = go.Figure()

    # Add the radar chart based on df with a name
    fig.add_trace(add_radar_chart_from_df(df, categories, 'rgb(243,146,0)',  "Apprenti"))

    # Names for the additional radar charts
    names = ['Etalon TED', 'Etalon Politique', 'Etalon Entreprise']

    # Fixed scores for the additional radar charts
    fixed_scores_list = [
        [5.2, 5.6, 6.4, 6.6, 8, 5.2],
        [4.8, 5.6, 4.4, 6.2, 5.8, 4.8],
        [5.4, 6, 6.8, 7, 6.8, 5.4]
    ]

    # Colors for the additional radar charts
    colors = ['rgb(128, 128, 0)', 'rgb(72, 209, 204)', 'rgb(0, 78, 146)']

    # Add three additional radar charts with fixed parameters and names
    for scores, color, name in zip(fixed_scores_list, colors, names):
        fig.add_trace(add_fixed_radar_chart(scores, categories, color, name))

    # Update the layout with a range from 0 to 10 for the radial axis
    # And set tickvals to mark every integer from 0 to 10
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

    fig.write_image(os.path.join( radar_plot_path.replace('/', os.sep)))
    axes_fig.write_image(os.path.join( axes_radar_plot_path.replace('/', os.sep)))

    score_df = fetch_score_data_from_firebase()
    global_score_df = fetch_global_scores_data_from_firebase()

    mean_score_df = score_df.groupby(['id_session', 'theme']).mean()
    mean_global_score_df = global_score_df.groupby(['id_session', 'theme']).mean()
   
 
    global_score_df = fetch_global_scores_data_from_firebase()
    global_score_df = global_score_df[global_score_df['id_session'] == id_session]
    if global_score_df.shape[0] > 1:
        global_fig = generate_radar_charts(global_score_df)
        global_fig.write_image(os.path.join( global_radar_plot_path.replace('/', os.sep)))

    # Reference to the sessions in the database
    sessions_ref = db.reference('sessions')


    # Wrap text
    last_col_index = len(df.columns) - 1
    for i in range(len(df)):
        df.iloc[i, last_col_index] = textwrap.fill(str(df.iloc[i, last_col_index]), width=50)

    pdf_buffer = io.BytesIO()
    
    # Register font variants
    pdfmetrics.registerFont(TTFont('WorkSans-Regular', 'static/fonts/WorkSans-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('WorkSans-Bold', 'static/fonts/WorkSans-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('WorkSans-Italic', 'static/fonts/WorkSans-Italic.ttf'))
    pdfmetrics.registerFont(TTFont('WorkSans-BoldItalic', 'static/fonts/WorkSans-BoldItalic.ttf'))

    # Assuming TomaSans is corrected to TTF; if OTF, ensure compatibility or convert to TTF
    pdfmetrics.registerFont(TTFont('TomaSans-Regular', 'static/fonts/TomaSans-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('TomaSans-Bold', 'static/fonts/TomaSans-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('TomaSans-Italic', 'static/fonts/TomaSans-Italic.ttf'))
    pdfmetrics.registerFont(TTFont('TomaSans-BoldItalic', 'static/fonts/TomaSans-BoldItalic.ttf'))

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontName = 'WorkSans-Regular'  # Corrected font name
    #normal_style.textColor = colors.HexColor('#000000')
    # Creating a centered and bold style for section titles
    title_style = ParagraphStyle(name='CenteredBold', parent=styles['Heading1'], alignment=1)

    # Adjusting the bold style to be used for sub-keys
    bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'], fontName='WorkSans-Bold')

    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm,
                            title="Analysis Results")

    logo_path = os.path.join('static', 'img/logo.png')
    logo = Image(logo_path, 4*cm, 2*cm)
    logo.hAlign = 'RIGHT'




    # Start with the static parts of your paragraph
    text_paragraphs_content = (f"<b>ID Client: </b> {id_session}<br/>"
                               f"<b>Session: </b> {current_session_number}<br/>"
                                f"<b>Catégorie de la session: </b> {theme}<br/>"
                                f"<b>Date: </b> {date}<br/>")



    text_paragraphs = Paragraph(text_paragraphs_content)

    # The table data is a list containing one row with two cells:
    # The first cell contains the combined text, and the second cell contains the logo.
    
    table_data = [[text_paragraphs, logo]]

    # Create the table from the data
    header_table = Table(table_data, colWidths=[None, 4*cm])

    # Set the style for the table to align the logo to the right
    header_table.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN', (1,0), (1,0), 'RIGHT'),
    # Set row height, adjust the value as needed
    ('ROWHEIGHTS', (0,0), (-1,-1), 60),
    ]))

    # Add the table to the story
    story = [header_table]
    #story.append(Spacer(1, -2*cm))  # Adjust -1*cm to move the logo up by the desired amount
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Utilisation des 4 grands types de discours</b>", title_style))
    story.append(Spacer(1, 12))
    story.append(Spacer(1, 12))

    story.append(Image(axes_radar_plot_path, 18*cm, 12*cm))

    story.append(PageBreak())
    # Add a spacer after the header for spacing before the next content
    story.append(Spacer(1, 12))

    story.append(Spacer(1, -1*cm))  # Adjust to move the logo up by the desired amount
    story.append(logo)
    i = 1
    for critere in df['Critere']:
            # Inserting gauge visualization here
            mean_score = round(mean_global_score_df["score{i}".format(i=i)].values[0],2)
            i += 1
            score = df[df['Critere'] == critere]['Score'].values[0]
            gauge = CustomLinearGauge(score = score, mean = mean_score )

            # Combine sub_key description and gauge in a Table for inline appearance
            content_paragraph = Paragraph(f"<b>{critere}:</b>")
            table_data = [[content_paragraph, gauge]]
            content_table = Table(table_data, colWidths=[None, 200])
            story.append(content_table)

            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))
            # Adding commentaire below the gauge
            commentaire_paragraph = Paragraph(f"{df[df['Critere'] == critere]['Score detaillee'].values[0]}", styles['Normal'])
            story.append(commentaire_paragraph)
            story.append(Spacer(1, 12))  # Space after commentaire
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

   

    story.append(PageBreak())

    #story.append(Spacer(1, -2*cm))  # Adjust -1*cm to move the logo up by the desired amount
    story.append(logo)
    story.append(Image(radar_plot_path, 18*cm, 12*cm))



    # Function to create gauge omitted for brevity; assume LinearGauge class is defined
    i = 1
    for key, value in detailed_data.items():
 
        story.append(PageBreak()) if story else None  # Add a PageBreak if not the first page
        story.append(Paragraph(key, title_style))

        story.append(Spacer(1, -2*cm))  # Adjust to move the logo up by the desired amount
        story.append(logo)
        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))
    
        for sub_key, sub_value in value.items():
            # Inserting gauge visualization here
            mean_score = round(mean_score_df["score{i}".format(i=i)].values[0],2)
            i += 1
            gauge = CustomLinearGauge(score = sub_value['score'], mean = mean_score )

            # Combine sub_key description and gauge in a Table for inline appearance
            content_paragraph = Paragraph(f"<b>{sub_key}:</b>")
            table_data = [[content_paragraph, gauge]]
            content_table = Table(table_data, colWidths=[None, 200])
            story.append(content_table)

            story.append(Spacer(1, 12))

            # Adding commentaire below the gauge
            commentaire_paragraph = Paragraph(f"{sub_value['commentaire']}", styles['Normal'])
            story.append(commentaire_paragraph)
            story.append(Spacer(1, 12))  # Space after commentaire
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

 

    if global_score_df.shape[0] > 1:
        story.append(PageBreak())
        story.append(Paragraph(f"<b>Titre à définir</b>", title_style))
        story.append(Spacer(1, -2*cm))  # Adjust to move the logo up by the desired amount
        story.append(logo)

        story.append(Image(global_radar_plot_path, 18*cm, 12*cm))
 
    doc.build(story)

    pdf = pdf_buffer.getvalue()
    pdf_buffer.close()

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={id_session}_{current_session_number}_{theme}_results.pdf'
    
    return response


@app.route('/download-csv')
def download_csv():
     # Retrieve session entries from Firebase
    ref = db.reference('sessions')
    sessions_data = ref.get() or {}

    # Assuming each session entry is structured under 'sessions/{session_id}/session_{number}'
    # Adjust the data extraction logic as per your actual data structure
    data = []
    for session_id, sessions in sessions_data.items():
        for session_key, details in sessions.items():
            # Ensure 'details' is a dictionary and contains necessary keys
            if isinstance(details, dict) and all(key in details for key in ['session', 'theme', 'date']):
                data.append({
                    'ID' : "{cat}_{client}_{id_session}".format(cat = details['theme'], client = session_id, 
                                                                id_session = details['session']),
                    'ID Client': session_id,
                    'Session': details['session'],
                    'Catégorie': details['theme'],
                    'Date': details['date']
                })
    base_url = '/download-hist-pdf/'  # Your Flask route for PDF downloads
    # Create the DataFrame
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=Historique de soumissions.csv"}
)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')#, port=int(os.environ.get('PORT', 5000)))
