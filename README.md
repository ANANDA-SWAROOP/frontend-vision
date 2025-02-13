

---

# Databyte Vision ChatBot 🖼️🤖

A conversational image chatbot built with **HTML**, **Tailwind CSS**, and **JavaScript**. This project allows users to send text and images, processes them (via a Python backend), and displays the response in a visually appealing chat interface.

---

## Features ✨

- **Text and Image Input**: Users can send text messages and upload images.
- **Dynamic Chat Interface**: Messages are displayed in a chat bubble format with smooth animations.
- **Image Previews**: Uploaded images are displayed as thumbnails before sending.
- **Backend Integration**: Ready to connect with a Python backend for processing text and images.
- **Responsive Design**: Works seamlessly on desktop and mobile devices.
- **Modern UI**: Built with Tailwind CSS for a sleek and modern look.

---

## How It Works 🛠️

1. **Frontend**:
   - Built with **HTML**, **Tailwind CSS**, and **JavaScript**.
   - Handles user input, image uploads, and displays chat messages dynamically.
   - Communicates with the backend via API calls.

2. **Backend**:
   - A Python backend processes the user's text and images.
   - Sends back a response (text and optional image) to the frontend.

---

## Installation and Setup 🚀

### Prerequisites
- A modern web browser (Chrome, Firefox, Edge, etc.)
- A Python backend (optional for now, as the frontend includes a simulated API response)

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ANANDA-SWAROOP/frontend-vision.git
   cd frontend-vision
   ```

2. Open the `index.html` file in your browser:
   ```bash
   open index.html  # On macOS
   start index.html # On Windows
   ```

3. (Optional) Connect to a Python backend:
   - Replace the `simulateApiCall` function in the `index.html` file with actual API calls to your backend.
   - Ensure your backend accepts `multipart/form-data` for text and image uploads.

---

## Code Structure 📂

```
vision-chatbot/
│
├── index.html          # Main HTML file with Tailwind CSS and JS
├── README.md           # This file
```

---

## Backend Integration 🔗

To connect the frontend to your Python backend:

1. Modify the `simulateApiCall` function in the `index.html` file:
   ```javascript
   async function simulateApiCall(data) {
       const formData = new FormData();
       formData.append('prompt', data.text);
       data.images.forEach((img, i) => formData.append(`image_${i}`, img));

       const response = await fetch('/api/endpoint', {
           method: 'POST',
           body: formData
       });

       return await response.json();
   }
   ```

2. Ensure your backend:
   - Accepts `multipart/form-data` for text and image uploads.
   - Returns a JSON response with `text` and an optional `image` URL.

---
