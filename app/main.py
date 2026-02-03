from flask import Flask, session, redirect, url_for, request, render_template
from routes_Evap_new import register_evaporation_routes
from routes_Water_level_new import register_water_level_routes
# from routes_C2 import register_water_level_routes
from routes_Bromine_concentration_new import register_bromine_concentration_routes

def create_app():
    app = Flask(__name__)
    app.secret_key = 'YOUR_SECRET_KEY_HERE'
    app.config['UPLOAD_FOLDER'] = 'app/uploads'  # or just 'uploads' depending on your structure

    # Register Blueprints
    register_evaporation_routes(app)
    register_water_level_routes(app)
    register_bromine_concentration_routes(app)

    @app.route('/')
    @app.route('/Login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if username == 'admin' and password == 'password':
                session['user'] = username
                return redirect(url_for('home'))  # make sure this matches your actual route function name
            else:
                return render_template('login.html', error="Invalid username or password")
        return render_template('login.html')

    @app.route('/Home', methods=['GET', 'POST'])
    def home():
        if 'user' not in session:
            return redirect(url_for('login'))

        if request.method == 'POST':
            choice = request.form.get('choice')
            if choice == 'evaporation':
                return redirect(url_for('Evaporation'))  # must match the function name in routes_evaporation
            elif choice == 'water':
                return redirect(url_for('Waterlevel'))  # must match the function name in routes_water_level
            elif choice == 'bromine_concentration':
                return redirect(url_for('BromineConcentration'))

        return render_template('Home.html')

    @app.route('/Logout', methods=['GET', 'POST'])
    def logout():
        session.pop('user', None)
        return redirect(url_for('login'))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)