# InvisCrack – React Frontend

Glass defect detection UI built with React + plain CSS.

---

## Setup

### 1. Create the React app

```bash
npx create-react-app inviscrack-frontend
cd inviscrack-frontend
```

### 2. Replace source files

Copy these files into the `src/` folder, replacing what's already there:

```
src/
├── App.js        ← main component (upload, detection, canvas overlay)
├── App.css       ← all styles
├── index.js      ← entry point
└── index.css     ← global reset
```

### 3. Install & start

```bash
npm install
npm start
```

The app opens at **http://localhost:3000**

---

## Make sure the backend is running

The FastAPI backend must be running on port 8000 before you use the frontend:

```bash
# In your InvisCrack backend folder
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## How it works

| Step | What happens |
|------|-------------|
| Select image | Preview shown immediately |
| Click "Run Detection" | Image POSTed to `http://localhost:8000/detect` |
| Response received | Bounding boxes drawn on canvas overlay |
| Results panel | Lists each defect with type, confidence, severity |
| Summary panel | Total count, severe count, per-type breakdown |

---

## Severity colours

| Severity | Box colour |
|----------|-----------|
| Severe   | 🔴 Red    |
| Minor    | 🟢 Green  |

---

## Changing the API URL

Edit this line at the top of `src/App.js`:

```js
const API_URL = "http://localhost:8000/detect";
```
