from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths for screenshots and CSV file
SCREENSHOT_DIR = './screenshots-2'  # Folder containing screenshots
RATINGS_FILE = './ratings-set2-pooja.csv'  # CSV file to save ratings

# Ensure CSV file has headers
def initialize_csv():
    """Create the CSV file with headers if it does not exist."""
    if not os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Screenshot"])  # Start with just the Screenshot column

initialize_csv()

def get_screenshots():
    """
    Retrieve and sort screenshot filenames.
    """
    screenshots = [f for f in os.listdir(SCREENSHOT_DIR) if f.endswith('.png')]
    return sorted(screenshots, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sorting numerically

screenshots = get_screenshots()

def read_csv():
    """Read the CSV file into a dictionary."""
    if not os.path.exists(RATINGS_FILE):
        return {}

    with open(RATINGS_FILE, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    if not data:
        return {}

    headers = data[0]  # Column headers
    rows = {row[0]: row[1:] for row in data[1:]}  # {screenshot: [ratings per user]}

    return headers, rows

def write_csv(headers, rows):
    """Write data back to the CSV file."""
    with open(RATINGS_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for screenshot, ratings in rows.items():
            writer.writerow([screenshot] + ratings)

@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index2.html')

@app.route('/screenshots', methods=['GET'])
def get_screenshot_list():
    """Get the list of available screenshots."""
    screenshot_urls = [request.host_url + "screenshot/" + f for f in screenshots]
    return jsonify({"screenshots": screenshot_urls})

@app.route('/screenshot/<filename>', methods=['GET'])
def serve_screenshot(filename):
    """Serve a specific screenshot file."""
    return send_from_directory(SCREENSHOT_DIR, filename)

@app.route('/rate', methods=['POST'])
def rate_screenshot():
    """Save the rating for the screenshot."""
    data = request.json

    # Debugging: log incoming request data
    print(f"Received data: {data}")

    if not data or 'screenshot' not in data or 'rating' not in data or 'user' not in data:
        return jsonify({"error": "Invalid data.", "data_received": data}), 400

    screenshot = data['screenshot']
    rating = data['rating']
    user = data['user']

    # Ensure only User1 is considered
    if user != 'User1':
        return jsonify({"error": "Only User1 can submit ratings."}), 400

    # Check if the CSV file exists, if not, create it with a header row
    if not os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Screenshot', 'User1'])  # Only User1 is included in the header

    # Read the CSV file to get the current ratings
    rows = []
    updated = False
    with open(RATINGS_FILE, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Find the row for the given screenshot
    for row in rows:
        if row[0] == screenshot:
            # Update the rating for User1
            row[1] = rating  # User1's rating is updated
            updated = True
            break

    if not updated:
        # If the screenshot doesn't exist, add a new row with the rating for User1
        new_row = [screenshot, rating]  # Screenshot and rating for User1
        rows.append(new_row)

    # Write the updated rows back to the CSV
    with open(RATINGS_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return jsonify({"message": "Rating submitted successfully."})


if __name__ == '__main__':
    app.run(debug=True)
