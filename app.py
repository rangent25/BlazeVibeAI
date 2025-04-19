import os
import logging
import json
import nltk
from flask import Flask, render_template, request, jsonify
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data (run once)
try:
    nltk.download('vader_lexicon')
    logger.info("NLTK vader_lexicon downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK vader_lexicon: {e}")

app = Flask(__name__)

# Mock context storage (in-memory for simplicity)
context_store = {}

# Dictionary of predefined questions and answers
question_responses = {
    "how are you": "I'm doing great, thanks for asking! How can I assist you today?",
    "what is beaconmind ai": "BeaconMind AI is a tool designed to support students' mental well-being and academic leadership. It provides AI-driven insights, interactive stories, and customizable avatars inspired by social media trends.",
    "how to reduce stress": "To reduce stress, try deep breathing exercises, take short breaks, or talk to someone you trust. Would you like more detailed tips?",
    "what to do if i feel sad": "I'm sorry to hear that. Try talking to a friend, engaging in a hobby, or seeking help from a counselor. Would you like resources?",
    "help with homework": "I can’t do your homework, but I can guide you! Tell me what you’re working on, and I’ll suggest some steps or resources.",
    "friends offered weed": "That sounds like a tough situation. It’s important to say no and talk to a trusted adult or counselor for support. I can help you with a plan if you’d like.",
    "i’m stressed": "It sounds like you’re feeling stressed. Take a deep breath and try some relaxation techniques. Would you like some tips?",
    "how to make friends": "Making friends can be fun! Try joining a club, being kind, or starting with a smile. Want more ideas?",
    "how to stay positive": "To stay positive, practice gratitude by noting 3 good things daily, exercise regularly, and surround yourself with supportive people. Want more ideas?",
    "what are healthy habits": "Healthy habits include getting 7-9 hours of sleep, eating balanced meals, exercising, and taking time to relax. These boost your mental health!",
    "how to manage anxiety": "To manage anxiety, try deep breathing, mindfulness, or journaling your thoughts. If it persists, consider talking to a professional. Need guidance?",
    "how to build self-confidence": "Building self-confidence involves setting small goals, celebrating successes, and practicing self-kindness. Want tips to get started?",
    "what to do if i feel overwhelmed": "Feeling overwhelmed is okay. Break tasks into smaller steps, take breaks, and talk to someone. I can help you create a plan if you’d like.",
    "how to avoid peer pressure": "To avoid peer pressure, practice saying 'no' firmly, hang out with positive friends, and have an exit plan. Want role-play scenarios?",
    "what are the risks of drugs": "Drugs can harm your brain, body, and future. They can lead to addiction, health issues, and legal trouble. Always seek help if offered. Want more info?",
    "how to say no to drugs": "You can say no by being assertive ('No, thanks, I’m good'), walking away, or suggesting alternatives like a game. Practice with me if you’d like!",
    "what if i know someone using drugs": "If someone you know is using drugs, encourage them to seek help and inform a trusted adult. Your support can make a difference. Need advice?",
    "how to improve my mood": "To improve your mood, try listening to music, going for a walk, or doing something you enjoy. Small steps can help a lot!",
    "how to deal with loneliness": "Loneliness can be tough. Try connecting online, joining a group, or reaching out to someone. Want more suggestions?",
    "what is mindfulness": "Mindfulness is focusing on the present moment without judgment. Try a 5-minute breathing exercise to start. Interested?",
    "how to set goals": "Set SMART goals—Specific, Measurable, Achievable, Relevant, Time-bound. Want help creating one?",
    "what is a confidence booster": "A quick confidence booster is standing tall, smiling, or recalling a past success. Want more techniques?",
    "how to handle bullying": "If you’re facing bullying, tell a trusted adult, stay calm, and avoid reacting. Want a safety plan?",
    "what are the signs of depression": "Signs include persistent sadness, loss of interest, or sleep changes. If you notice these, talk to someone. Need help?",
    "how to prepare for exams": "Plan a study schedule, review notes, and take practice tests. Want a detailed plan for your subject?",
    "what is regression analysis": "Regression analysis predicts relationships between variables, like sales and advertising. Want an example?",
    "how to calculate probability": "Probability is the chance of an event, calculated as favorable outcomes divided by total outcomes. Want a die-rolling example?",
    "what is multicollinearity": "Multicollinearity happens when independent variables in a regression are highly correlated, affecting results. Want a fix?",
    "how to use pandas in python": "Pandas helps analyze data with DataFrames. Try `import pandas as pd` and load a CSV with `pd.read_csv()`. Want code?",
    "what is climate change": "Climate change is long-term shifts in weather due to human activities like emissions. Want action tips?",
    "how to save energy": "Turn off lights, use energy-efficient appliances, and unplug devices. Want a home plan?",
    "what should i eat today": "Try a balanced meal with veggies, protein, and grains. Want a recipe based on your preferences?",
    "how to plan my day": "List tasks, prioritize, and set time blocks. Want a sample schedule?",
    "what is a t-test": "A t-test compares means to see if they differ significantly. Want an example with data?",
    "how to visualize data": "Use Matplotlib or Seaborn in Python to plot graphs. Want a line chart code?",
    "what are sustainable practices": "Sustainable practices include recycling, reducing waste, and using renewables. Want ideas for school?",
    "how to improve focus": "Try the Pomodoro technique or minimize distractions. Want a timer setup?",
    "what to do if i’m anxious about school": "Take deep breaths, break tasks into steps, and talk to a teacher. Need a plan?",
    "how to learn statistics": "Start with basics like mean and median, then explore probability. Want a study guide?",
    "what is autocorrelation": "Autocorrelation is when data points are correlated over time, common in time series. Want a test method?",
    "how to say no to alcohol": "Be firm with 'No, I don’t drink,' and suggest a non-alcoholic activity. Want practice?",
    "what if i feel peer pressure to drink": "Stay with supportive friends, have an excuse ready, and leave if needed. Want strategies?",
    "how to create a study group": "Invite classmates, set a schedule, and assign topics. Want a template?",
    "what is a hypothesis test": "A hypothesis test checks if a claim about data is true using statistics. Want an example?",
    "how to reduce carbon footprint": "Use public transport, eat less meat, and conserve energy. Want a personal plan?",
    "what should i do this weekend": "Plan a walk, read a book, or relax. Want ideas based on your location?",
    "how to deal with exam anxiety": "Practice relaxation, prepare early, and get rest. Want a pre-exam routine?",
    "what is econometrics": "Econometrics uses stats to test economic theories. Want a real-world example?",
    "how to interpret r-squared": "R-squared shows how much variance in data is explained by a model (0 to 1). Want a deeper explanation?",
    "what are sampling methods": "Methods include random, stratified, and cluster sampling. Want details on one?",
    "how to stay motivated": "Set rewards, track progress, and remind yourself of goals. Want a motivation plan?",
    "what to do if i fail a test": "Review mistakes, seek help, and try again. Want a recovery strategy?",
    "how to manage time": "Use a calendar, prioritize tasks, and avoid multitasking. Want a daily schedule?",
    "what is heteroskedasticity": "Heteroskedasticity is uneven variance in errors in regression. Want a detection method?",
    "how to learn coding": "Start with Python, practice daily, and use online tutorials. Want a beginner project?",
    "what are the benefits of exercise": "Exercise boosts mood, energy, and focus. Want a workout plan?",
    "how to handle rejection": "Reflect, learn from it, and try again. Want coping tips?",
    "what is a confidence interval": "A confidence interval estimates a parameter with a range and confidence level. Want a calculation?",
    "how to reduce screen time": "Set limits, take breaks, and replace with hobbies. Want a schedule?",
    "what to do if i’m bored": "Try a new hobby, exercise, or learn something new. Want suggestions?",
    "how to support a friend in need": "Listen, offer help, and suggest resources. Want advice for a specific situation?",
    "what is a p-value": "A p-value measures evidence against a null hypothesis (typically < 0.05 is significant). Want an example?",
    "how to prepare for a job interview": "Practice answers, research the company, and dress well. Want mock questions?",
    "what are renewable energy sources": "Sources like solar and wind are renewable. Want pros and cons?",
    "how to improve sleep": "Maintain a routine, avoid screens before bed, and relax. Want a bedtime plan?",
    "default": "Thanks for sharing! I’m here to help. Please provide more details or ask a specific question so I can assist you better.",
    "how are you": "I'm doing great, thanks for asking! How can I assist you today?",
    "what is beaconmind ai": "BeaconMind AI is a tool designed to support students' mental well-being...",
    # ... (keep all your existing questions) ...

    # --- New: Exam & Academic Stress ---
    "i'm scared of failing": "It's normal to feel this way. Focus on preparation—break material into small chunks and practice regularly. Want a study plan?",
    "can't focus on studies": "Try the Pomodoro technique (25 mins study + 5 mins break), eliminate distractions, and create a quiet workspace. Need focus tips?",
    "what if i fail my exams": "Exams don't define you. If things don't go well, you can retake or explore alternative paths. Want to discuss options?",

    # --- New: Substance Abuse ---
    "how to quit vaping": "Reduce gradually, chew gum as a substitute, and seek support from a counselor. You’ve got this! Need a step-by-step plan?",
    "is weed dangerous": "Yes, it can harm memory, motivation, and mental health, especially for teens. Want science-backed facts?",
    "i regret trying drugs": "It takes courage to admit this. Talk to a trusted adult or call [local helpline]. You’re not alone.",

    # --- New: Sexual Harassment/Safety ---
    "someone touched me inappropriately": "This is NOT your fault. Tell a trusted adult immediately or contact [helpline]. You deserve safety and support.",
    "i'm being sextorted": "Don’t panic. Stop communication, save evidence, and report to [cybercrime unit]. This is illegal—help is available.",
    "how to report harassment": "Document incidents (screenshots/texts), tell a teacher/parent, and contact school authorities. Want help drafting a report?",

    # --- New: Mental Health Crises ---
    "i want to self-harm": "You’re important. Please call [crisis hotline] or talk to a trusted adult NOW. Your pain matters, and help is available.",
    "i think about suicide": "You are not alone. Call [suicide hotline] immediately. Your life is precious, and people care deeply about you.",
    "how to help a suicidal friend": "Stay with them, listen without judgment, and get help from an adult/counselor immediately. Want conversation tips?",

    # --- New: Relationships ---
    "my partner is abusive": "No one should hurt you. Reach out to [domestic violence hotline] or a counselor. You deserve respect and safety.",
    "how to break up safely": "Do it in public/with a friend nearby, be clear but kind, and block if needed. Want a safety plan?",
    "i feel used in my relationship": "Healthy relationships are equal. If you feel exploited, it’s okay to step back. Want to talk through signs?",

    # --- New: Family Issues ---
    "my parents hit me": "This is never okay. Tell a teacher, counselor, or call [child abuse hotline]. You have the right to be safe.",
    "family fights all the time": "Try setting boundaries (e.g., 'I need space when you yell'). If it’s unsafe, talk to a counselor. Need coping strategies?",
    "parents don’t accept my identity": "You are valid. Seek LGBTQ+ support groups or a school counselor. Want affirming resources?",

    # --- New: Future Anxiety ---
    "i have no career direction": "Start by exploring free online courses or career quizzes. Many people change paths—it’s okay not to know yet!",
    "is college worth it": "It depends! Consider alternatives like trade schools or certifications. Want a pros/cons list for your situation?",
    "how to choose a major": "Think about what excites you, job prospects, and try introductory classes. Need a decision-making framework?",

    # --- New: Digital Safety ---
    "nudes leaked online": "First, document everything. Report to the platform and [cybercrime unit]. You have legal options—want guidance?",
    "i’m addicted to social media": "Set app timers, turn off notifications, and replace scrolling with hobbies. Want a digital detox plan?",
    "someone is cyberbullying me": "Don’t respond. Block, report to the platform, and tell an adult. You don’t deserve this—want help reporting?",

    # --- New: Physical Health ---
    "how to lose weight safely": "Focus on balanced meals, portion control, and exercise—not extreme diets. Want healthy habit tips?",
    "i think i have an eating disorder": "Talk to a doctor or counselor ASAP. Recovery is possible with support. Need help finding resources?",
    "can’t sleep at night": "Avoid screens before bed, try calming tea or reading, and keep a consistent schedule. Want a bedtime routine?",

    # --- New: Financial Worries ---
    "how to pay for college": "Explore scholarships, grants, and work-study programs. FAFSA can help—want application tips?",
    "i’m in debt": "Contact a financial counselor (many schools offer free help). Prioritize high-interest debts first. Need a plan template?",
    "how to budget as a student": "Track income/expenses, limit eating out, and use student discounts. Want a simple spreadsheet?",

    # --- Keep your existing default ---
    "default": "Thanks for sharing! I’m here to help. Please provide more details or ask a specific question..."
}
# Mock AI response function with intent detection
def get_mock_ai_response(input_text):
    input_lower = input_text.lower().strip()
    
    # Check for keywords in the input
    matched_question = None
    highest_score = 0
    
    for question in question_responses.keys():
        # Split question into keywords
        keywords = question.split()
        score = 0
        
        # Count how many keywords from this question appear in the input
        for keyword in keywords:
            if keyword in input_lower:
                score += 1
        
        # Normalize score by question length (to avoid favoring short questions)
        normalized_score = score / len(keywords)
        
        # Track the best matching question
        if normalized_score > highest_score:
            highest_score = normalized_score
            matched_question = question
    
    # If we found a good match (at least 50% of keywords present)
    if highest_score >= 0.5 and matched_question:
        stress_level = "Low"
        teacher_alert = None
        
        if any(keyword in input_lower for keyword in ["stressed", "sad", "overwhelmed", "anxiety", "weed", "drugs", "depress"]):
            stress_level = "High" if "weed" in input_lower or "drugs" in input_lower else "Moderate"
            teacher_alert = {"message": "Student needs attention due to stress or drug-related concern."} if stress_level != "Low" else None
        
        return {
            "response": question_responses[matched_question],
            "stress_level": stress_level,
            "teacher_alert": teacher_alert
        }
    
    # Default response if no good match
    return {
        "response": question_responses["default"],
        "stress_level": "Low",
        "teacher_alert": None
    }

# Store context function
def store_context(student_id, input_text, response):
    if student_id not in context_store:
        context_store[student_id] = []
    context_store[student_id].append({
        "timestamp": datetime.now().isoformat(),
        "input": input_text,
        "response": response
    })
    logger.info(f"Stored context for student {student_id}: {input_text}")

# Route for the landing page
@app.route('/')
def indexpage():
    template_path = os.path.join(app.root_path, 'templates', 'index1.html')
    if not os.path.exists(template_path):
        logger.error(f"Template file not found at: {template_path}")
        return "Error: Template 'index1.html' not found.", 500
    try:
        return render_template('index1.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return f"Error rendering template: {e}", 500

# Route for the inner page
@app.route('/inner')
def inner():
    template_path = os.path.join(app.root_path, 'templates', 'index.html')
    if not os.path.exists(template_path):
        logger.error(f"Template file not found at: {template_path}")
        return "Error: Template 'index.html' not found.", 500
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return f"Error rendering template: {e}", 500

# Route for student interaction
@app.route('/student', methods=['POST'])
def student_interaction():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        input_text = data.get('input_text')

        if not student_id or not input_text:
            return jsonify({"error": "Missing student_id or input_text"}), 400

        # Get mock AI response
        result = get_mock_ai_response(input_text)

        # Store context
        store_context(student_id, input_text, result["response"])

        # Prepare response
        response_data = {
            "response": result["response"],
            "stress_level": result["stress_level"],
            "teacher_alert": result["teacher_alert"]
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in student_interaction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')