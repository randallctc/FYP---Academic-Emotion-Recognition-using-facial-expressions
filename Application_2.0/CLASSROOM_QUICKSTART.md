# Classroom Emotion Monitoring - Quick Start Guide

## 🎯 What This Does

- **Teacher** can start a recording session
- **Students** join via web browser and turn on webcams
- **Real-time** emotion monitoring (Boredom, Confusion, Frustration, Engagement)
- **Analytics** saved to JSON after lesson

## 📦 Setup (One-Time)

### 1. Install Additional Requirements

```bash
pip install -r classroom_requirements.txt
```

This installs:
- Flask (web server)
- Flask-SocketIO (real-time communication)

### 2. Make Sure You Have

- ✅ `all_emotions_final.h5` (your model) in the project folder
- ✅ `emotion_recognition_app.py` in the project folder
- ✅ Python 3.10 or 3.11
- ✅ All students on the same network as teacher

## 🚀 Running the Application

### Step 1: Start the Server (Teacher's Computer)

```bash
python classroom_server.py
```

You should see:
```
CLASSROOM EMOTION MONITORING SERVER
Server starting on http://localhost:5000

URLs:
  Teacher Dashboard: http://localhost:5000/teacher
  Student Page:      http://localhost:5000/student
```

### Step 2: Teacher Opens Dashboard

On teacher's computer, open browser and go to:
```
http://localhost:5000/teacher
```

### Step 3: Students Join

Students open browser and go to:
```
http://<TEACHER_IP>:5000/student
```

**To find teacher's IP:**
- Windows: Open CMD, type `ipconfig`, look for IPv4 Address
- Mac/Linux: Terminal, type `ifconfig` or `ip addr`
- Example: `http://192.168.1.100:5000/student`

Students will:
1. Enter their name
2. Click "Join Class"
3. Allow webcam access

### Step 4: Start Recording

Teacher clicks **"Start Recording"** button

### Step 5: Teach Your Lesson

- Students' emotions are tracked in real-time
- Teacher sees live emotion bars
- Alerts shown when >50% students bored/confused

### Step 6: Stop Recording

Teacher clicks **"Stop Recording"** button

### Step 7: View Analytics

Results saved in `classroom_sessions/session_YYYYMMDD_HHMMSS.json`

## 📊 What You'll See

### Teacher Dashboard:

```
┌─────────────────────────────────────┐
│ 📊 Current Emotions                 │
├─────────────────────────────────────┤
│ Engagement:   15 students (50%) ████│
│ Boredom:       8 students (27%) ██  │
│ Confusion:     5 students (17%) █   │
│ Frustration:   2 students (6%)  ░   │
└─────────────────────────────────────┘

⚠️ [00:05:20] High boredom: 66.7% of students
```

### JSON Output:

```json
{
  "session_id": "20260223_143022",
  "duration": "00:15:30",
  "total_students": 30,
  "timeline": [
    {
      "timestamp": "00:00.000",
      "total_students": 30,
      "emotions": {
        "Engagement": 24,
        "Boredom": 3,
        "Confusion": 2,
        "Frustration": 1
      },
      "percentages": {
        "Engagement": 80.0,
        "Boredom": 10.0,
        "Confusion": 6.7,
        "Frustration": 3.3
      }
    },
    {
      "timestamp": "00:05:20",
      "total_students": 30,
      "emotions": {
        "Engagement": 5,
        "Boredom": 20,
        "Confusion": 3,
        "Frustration": 2
      },
      "percentages": {
        "Engagement": 16.7,
        "Boredom": 66.7,
        "Confusion": 10.0,
        "Frustration": 6.7
      },
      "alert": "⚠️ High boredom: 66.7% of students"
    }
  ]
}
```

## 🎓 Usage Tips

### For Teachers:

1. **Start recording BEFORE lesson begins** - to capture initial engagement
2. **Watch the alerts** - red warnings indicate problem areas
3. **Review JSON afterwards** - identify exact timestamps where engagement dropped
4. **Test with 2-3 students first** - before full class

### For Students:

1. **Good lighting** - face should be clearly visible
2. **Stay in frame** - don't move too far from camera
3. **Natural expressions** - system works best with genuine reactions
4. **Stable internet** - for smooth transmission

## 🔧 Troubleshooting

### "Model not found"
```bash
# Make sure model is in the same directory
ls all_emotions_final.h5
```

### "Cannot access webcam"
- Check browser permissions (allow camera access)
- Only one app can use webcam at a time
- Try different browser (Chrome/Firefox recommended)

### "Students can't connect"
- Make sure students use teacher's IP address, not `localhost`
- Check firewall - port 5000 must be open
- Both must be on same WiFi network

### "Slow/Laggy"
- Fewer students = better performance
- Close other applications
- Use GPU if available (see PERFORMANCE.md)

### "Port already in use"
```bash
# Kill existing process
# Windows: taskkill /F /IM python.exe
# Mac/Linux: pkill -9 python
```

## 📈 Analyzing Results

### Read the JSON:

```python
import json

with open('classroom_sessions/session_20260223_143022.json', 'r') as f:
    data = json.load(f)

# Find low engagement moments
for entry in data['timeline']:
    engagement_pct = entry['percentages'].get('Engagement', 0)
    if engagement_pct < 30:
        print(f"Low engagement at {entry['timestamp']}: {engagement_pct:.1f}%")
```

### Export to Excel (Optional):

```python
import pandas as pd
import json

with open('classroom_sessions/session_20260223_143022.json', 'r') as f:
    data = json.load(f)

# Convert timeline to DataFrame
df = pd.DataFrame(data['timeline'])
df.to_excel('lesson_analytics.xlsx', index=False)
```

## 🎯 Example Workflow

```bash
# 1. Start server
python classroom_server.py

# 2. Teacher opens: http://localhost:5000/teacher
# 3. Students open: http://192.168.1.100:5000/student

# 4. Wait for all students to join
# 5. Click "Start Recording"
# 6. Teach lesson (45 minutes)
# 7. Click "Stop Recording"

# 8. Check results:
ls classroom_sessions/
cat classroom_sessions/session_*.json
```

## 💡 Pro Tips

1. **Do a dry run** with 2-3 volunteers first
2. **Prepare students** - explain what emotion tracking means
3. **Internet stability** - wired connection better than WiFi
4. **Review alerts immediately after** class for quick improvements
5. **Compare sessions** over time to track teaching improvement

## 🚨 Known Limitations

- **Maximum ~30 students** (CPU limitation)
- **Requires good lighting** for face detection
- **2-second delay** between emotion updates (can adjust)
- **Model accuracy** - depends on your training data

## 📞 Need Help?

Check these files:
- `MODEL_TROUBLESHOOTING.md` - Model issues
- `PERFORMANCE.md` - Speed improvements
- `README.md` - General documentation

## ✅ Quick Checklist

Before class:
- [ ] Server running
- [ ] Teacher dashboard opened
- [ ] Test with 1 student
- [ ] All students know the URL
- [ ] Webcams working

During class:
- [ ] Click "Start Recording"
- [ ] Monitor real-time emotions
- [ ] Note alert timestamps

After class:
- [ ] Click "Stop Recording"
- [ ] Check JSON file created
- [ ] Review problem areas
- [ ] Plan improvements

---

**That's it! You're ready to monitor classroom emotions!** 🎓📊
