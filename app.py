from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from cvanalyzer import get_results

app = Flask(__name__)
BASE_UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = BASE_UPLOAD_FOLDER
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc = request.form.get("job_description", "").strip()
        uploaded_files = request.files.getlist("resumes")

        # ✅ Basic validation: make sure both inputs are present
        if not job_desc or not uploaded_files:
            return render_template("index.html", error="Please provide job description and upload at least one resume.")

        # 🔹 Create unique session folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"session_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)

        saved_files = []
        for file in uploaded_files:
            if file and file.filename.endswith(".pdf"):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_folder, filename)
                file.save(filepath)
                saved_files.append(filepath)

        # ✅ Check if any files were saved
        if not saved_files:
            return render_template("index.html", error="No valid PDF files were uploaded.")

        try:
            results = get_results(session_folder, job_desc)
        except Exception as e:
            return render_template("index.html", error=f"Failed to process resumes: {str(e)}")

        return render_template("results.html", results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
