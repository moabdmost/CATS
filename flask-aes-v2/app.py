import os
from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import subprocess
import sys
sys.path.append('/workspace')
from fake_detector.detector import fake_real
import json
from datetime import datetime  # Add this line to import datetime

#sys.path.append("/workspace/unikl_project/expats")

# from task import predict

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users2.db'
app.config['SECRET_KEY'] = 'your-secret-key'
db = SQLAlchemy(app)
migrate = Migrate(app, db)


# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    id_number = db.Column(db.String(20), nullable=False)
    usertype = db.Column(db.String(20), nullable=False)
    approved = db.Column(db.Boolean, default=False)

# submit model
class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, nullable=False)
    answer_text = db.Column(db.Text, nullable=False)
    real_false = db.Column(db.String(10))
    likelihood = db.Column(db.String(20))
    predicted_score = db.Column(db.Float)
    submission_datetime = db.Column(db.DateTime, default=datetime.utcnow)  # Use datetime.utcnow as the default value

    # Specify the foreign key for the relationship with ON DELETE CASCADE
    student_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    student = db.relationship('User', backref='submissions')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            # if user.approved:  # Check if the user is approved
                session['username'] = username
                if user.usertype == 'student':
                    return redirect('/dashboard')   # Redirect student users to the dashboard page
                elif user.usertype == 'assessor':
                    return redirect('/assessor')
                else:
                    return redirect('/admin')  # Redirect admin and assessor users to the admin panel
        #     else:
        #         return 'User not approved yet. Please wait for admin approval.'
        # else:
        #     return 'Invalid username or password.'
    else:
        return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

# Signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        id_number = request.form['id_number']
        usertype = request.form['usertype']
        #user = User(username=username, password=password, id_number=id_number, usertype=usertype)
        user = User(username=username, password=password, id_number=id_number, usertype=usertype, approved=False)
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    
    else:
        return render_template('signup.html')
    #return render_template('signup.html')

# Approval page
@app.route('/approval', methods=['GET', 'POST'])
def approval():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'admin':
        return 'Access denied'

    if request.method == 'POST':
        username = request.form['username']
        user = User.query.filter_by(username=username).first()
        if user:
            user.approved = True
            db.session.commit()
            return f'{username} has been approved'
        else:
            return 'User not found'
    
    users = User.query.filter_by(approved=False).all()
    return render_template('approval.html', users=users)

@app.route('/reject', methods=['POST'])
def reject_user():
    username = request.form['username']
    user_to_reject = User.query.filter_by(username=username).first()

    if not user_to_reject:
        return 'User not found'
    
    db.session.delete(user_to_reject)
    db.session.commit()
    
    return redirect('/approval')


# Admin panel page (GET request)
@app.route('/admin', methods=['GET'])
def admin_panel():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'admin':
        return 'Access denied'

    users = User.query.all()
    return render_template('admin_panel.html', users=users)

# Edit user page
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'admin':
        return 'Access denied'

    user = User.query.get(user_id)
    if not user:
        return 'User not found'

    if request.method == 'POST':
        user.username = request.form['username']
        user.password = request.form['password']
        user.id_number = request.form['id_number']
        user.usertype = request.form['usertype']
        db.session.commit()
        return redirect('/admin')

    return render_template('edit_user.html', user=user)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'admin':
        return 'Access denied'

    user = User.query.get(user_id)
    if not user:
        return 'User not found'

    # Delete the related submissions first
    for submission in user.submissions:
        db.session.delete(submission)

    # Then delete the user
    db.session.delete(user)
    db.session.commit()

    return redirect('/admin')

@app.route('/view_database')
#@login_required
def view_database():
    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'admin':
        return 'Access denied'

    users = User.query.all()
    return render_template('view_database.html', users=users)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'student':
        return 'Access denied'

    return render_template('dashboard.html', username=username)

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'student':
        return 'Access denied'

    num_features = 300
    keyword = "computer"
    question_id = int(request.form['question_number']) # get the selected question id  
    text = request.form['answer']
    text = text.replace('\r', '').replace('\n', ' ')

    ### implement detector.py to detect fake or real essay
    real_false , rf_score = fake_real(text, model_keyword= keyword + '_detector')
    print( real_false , rf_score)

    if os.path.exists("/workspace/expats/prediction_process") != True:
        os.makedirs("/workspace/expats/prediction_process")
    file_path = "/workspace/expats/prediction_process/predict.txt"
    text_to_write = text + "\n\n" #append new line otherwise return error float object is not iterable
                                  #to display score need to omit score result from the 2nd line

    print(text_to_write)                           
    write_text_to_file(file_path, text_to_write)
    
    run_command("poetry run expats predict /workspace/expats/config/predict.yaml /workspace/models/"+ keyword +" /workspace/expats/prediction_process/result.txt", working_directory='/workspace/expats')

    
    if os.path.exists("/workspace/expats/prediction_process/result.txt") == True:
        # Read predicted score from result.txt
        predicted_score = read_predicted_score_from_file("/workspace/expats/prediction_process/result.txt")
        print_text_file("/workspace/expats/prediction_process/result.txt")
        os.remove("/workspace/expats/prediction_process/result.txt")
    else: print("Result was not found")

    # Get the current date and time
    submission_datetime = datetime.utcnow()

    # Process the submitted text as needed
    # Save the submission data to the database
    submission = Submission(
            
        question_id=question_id,
        answer_text=text,
        real_false=real_false,
        likelihood=rf_score,  # Save the real label as likelihood
        predicted_score=predicted_score,  # Save the predicted score
        submission_datetime=submission_datetime,
        student=user
    )
    db.session.add(submission)
    db.session.commit()

    return redirect('/dashboard')
    
@app.route('/delete_submission/<int:submission_id>', methods=['POST'])
def delete_submission(submission_id):
    submission = Submission.query.get(submission_id)
    if not submission:
        return 'Submission not found'

    db.session.delete(submission)
    db.session.commit()

    return redirect('/view_result')


@app.route('/view_result')
def view_result():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'admin':
        return 'Access denied'

    # Get all submissions from the database
    submissions = Submission.query.all()

    return render_template('view_results.html', submissions=submissions)

@app.route('/assessor')
def assessor():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    user = User.query.filter_by(username=username).first()
    if user.usertype != 'assessor':
        return 'Access denied'

    return render_template('assessor.html', username=username)

# get score from first line first column
def read_predicted_score_from_file(file_path): 
    try:
        # Open the file in read mode ('r')
        with open(file_path, 'r') as file:
            # Read the first line from the file
            first_line = file.readline()
            # Split the line by whitespace to get the values in each column
            columns = first_line.split()
            # Get the predicted score from the first column
            predicted_score = float(columns[0]) if columns else None
            return predicted_score
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")
        return None

#creates a new file with input text
def write_text_to_file(file_path, text):
    try:
        # Open the file in write mode ('w')
        with open(file_path, 'w') as file:
            # Write the specified text to the file
            file.write(text)
        
    except Exception as e:
        print(f"Error occurred while writing to the file: {e}")

#runs command in shell
def run_command(command, working_directory=None):
    try:
        # Execute the command in the shell
        subprocess.run(command, shell=True, check=True, cwd=working_directory)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the command: {e}")

#opens text file and prints it
def print_text_file(file_path):
    try:
        # Open the file in read mode ('r')
        with open(file_path, 'r') as file:
            # Read and print each line in the file
            for line in file:
                print(line, end="")
        print("\nFile printing completed.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
