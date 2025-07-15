
from flask import Flask, render_template, redirect, url_for, request, session
import os
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['user'] = username
            return redirect(url_for('select_tool'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/select_tool', methods=['GET', 'POST'])
def select_tool():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        tool = request.form.get('tool')
        if tool == 'ilf':
            return redirect(url_for('ilf_scoring'))
        elif tool == 'csd':
            return redirect(url_for('csd_analyzer'))
        else:
            return "Tool not recognized", 400

    return render_template('select_tool.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))
    
    @app.route('/ilf_scoring')
def ilf_scoring():
    return "<h2>ILF Scoring Page Coming Soon</h2>"

@app.route('/csd_analyzer')
def csd_analyzer():
    return "<h2>CSD Analyzer Page Coming Soon</h2>"




