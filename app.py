from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/enroll', methods=['POST'])
def enroll():
    name = request.form['name']
    password = request.form['password']
    with open('bank_details.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, password])  # Save name and password

    # For now, just redirecting to home
    return redirect(url_for('home'))


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    data = pd.read_csv('bank_details.csv')
    if data[(data['name'] == username) & (data['password'] == password)].empty:
        return "Invalid credentials", 401  # Unauthorized

    # For now, just redirecting to home
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
