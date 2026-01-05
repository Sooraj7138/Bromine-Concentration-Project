---

## ⚙️ Setup (Without Docker)

1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run app locally**
    ```bash
    cd app
    python main.py
    ```

3. Open your browser: [http://localhost:5000](http://localhost:5000)

---

## 🐳 Setup Using Docker (Recommended for deployment)

1. **Build the image**
    ```bash
    docker build -t bromine-flask-app .
    ```

2. **Run the container**
    ```bash
    docker run -p 5000:5000 bromine-flask-app
    ```

3. Open browser: [http://localhost:5000](http://localhost:5000)

---

## 📌 Notes

- Ensure uploaded CSVs match the expected formats (see modal previews).
- Evaporation must be predicted and downloaded before uploading to the bromine section.
- Logs and outputs are stored in `app/outputs`.

---

## 🧠 Tech Stack

- **Flask** (Web Framework)
- **TensorFlow/Keras** (GRU ML models)
- **Plotly** (Interactive graphs)
- **Docker** (Deployment-ready containerization)

---

## 🙋 Support

If you encounter issues or need help customizing this app, contact the developer or raise an issue.