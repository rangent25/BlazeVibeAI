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
    "how to deal with exam stress": "Take deep breaths, plan your study time, and believe in yourself. You've got this!",
    "how to focus while studying": "Minimize distractions, set small goals, and take short breaks. Would you like a focus plan?",
    "how to prepare for exams effectively": "Create a study schedule, revise regularly, and practice past papers. Want a study planner?",
    "i feel overwhelmed by exams": "It's okay to feel overwhelmed. Break tasks into smaller parts and tackle one thing at a time.",
    "how to manage time during exams": "Prioritize subjects, use a timetable, and avoid cramming. Time management boosts confidence!",
    "what if i fail my exam": "One exam doesn't define you. Learn from the experience and keep moving forward. You can do it!",
    "i can't remember anything i study": "Try active recall and spaced repetition techniques. They really help memory!",
    "how to stay motivated during exams": "Visualize your goals, reward yourself after study sessions, and stay positive.",
    "what to do one day before the exam": "Light revision, rest well, and prepare your materials. You've already done the hard work!",
    "should i pull an all-nighter before exam": "No, sleep is crucial! A fresh mind performs way better than a tired one.",
    "how to reduce anxiety before exams": "Practice breathing exercises, focus on what you know, and believe in yourself!",
    "how to set realistic study goals": "Break big goals into smaller steps and celebrate each achievement. Progress matters!",
    "i feel like giving up on studies": "Take a short break, remember your 'why', and come back stronger. You’re closer than you think!",
    "how to balance multiple subjects": "Create a subject rotation schedule and stick to it. Variety keeps your mind fresh!",
    "how to avoid distractions while studying": "Turn off notifications, use study apps, and create a quiet space. Need app suggestions?",
    "i panic during exams": "Breathe deeply, pause for a moment, and focus on one question at a time. You can train your brain to stay calm!",
    "how to deal with exam fear": "Understand that fear is normal. Preparation and positive self-talk reduce fear a lot.",
    "i forgot everything during the exam": "It happens! Take deep breaths, skip the question, and come back to it. Your brain often recalls later.",
    "how to prioritize subjects for study": "Focus first on subjects you find hardest or have exams earliest. Smart prioritization saves energy!",
    "how to make studying fun": "Use colorful notes, quizzes, group study, or reward systems. Learning can be exciting!",
    "what to eat during exams for better focus": "Foods like nuts, fruits, and water keep your brain sharp. Avoid heavy junk food!",
    "how to stay consistent with studying": "Create a daily habit, even small ones. Consistency beats intensity!",
    "how to deal with failure in academics": "Failure is feedback. Reflect, learn, and come back stronger. Many successful people failed first!",
    "how to stay positive after a bad exam": "Remind yourself of your efforts, not just the outcome. There's always another chance!",
    "how to boost memory for exams": "Try mnemonics, mind maps, and teaching what you learned to someone else!",
    "how to deal with peer pressure about grades": "Focus on your personal journey, not others'. Everyone has a different path!",
    "how to relax after studying": "Listen to music, walk outside, or watch something uplifting. You deserve breaks too!",
    "how to avoid procrastination while studying": "Break your tasks into tiny parts and start with just 5 minutes. Getting started is the hardest!",
    "how to manage study breaks": "Use the 50/10 rule: study for 50 minutes, break for 10 minutes. Keeps you fresh!",
    "how to overcome fear of failure": "See failure as a step towards growth. Every try builds your resilience!",
    "how to stay confident before exams": "Recall your preparation, trust yourself, and imagine success. Confidence is a choice!",
    "how to study when feeling demotivated": "Start with easy topics to build momentum. Progress brings motivation!",
    "how to stop overthinking before exams": "Focus on actions, not outcomes. One step at a time brings results!",
    "how to revise quickly before exams": "Focus on key concepts, summaries, and practice questions. Smart revision saves time!",
    "how to create an exam revision plan": "List topics, allocate days, and stick to short, focused sessions. Need a template?",
    "how to stop comparing myself to others": "Your journey is unique. Focus on being better than yesterday, not better than others!",
    "how to prepare for practical exams": "Practice hands-on tasks, understand key concepts, and stay calm. You've practiced this!",
    "how to deal with low grades": "Low grades don't mean low potential. Analyze, adjust strategies, and aim higher next time!",
    "how to handle pressure from parents about grades": "Communicate openly, show your effort, and set realistic expectations. Need tips for tough conversations?",
    "how to study smarter, not harder": "Use active recall, past papers, and focus on understanding, not just memorizing!",
    "how to stay energized during exam week": "Sleep well, eat balanced meals, stay hydrated, and move your body!",
    "how to avoid burnout during exams": "Mix study with relaxation. Listen to your body's signals and recharge when needed.",
    "how to get back on track after failure": "Pause, reflect, learn, and start fresh. Every comeback story starts with a setback!",
    "how to use past papers effectively": "Time yourself, review mistakes, and simulate exam conditions for best practice!",
    "how to study when feeling sleepy": "Splash water on your face, walk around, or switch topics. Short active breaks help!",
    "how to study for long hours without getting tired": "Take regular breaks, stretch, hydrate, and use the Pomodoro technique!",
    "how to overcome fear of tough subjects": "Break them into small parts and celebrate small wins. Progress builds courage!",
    "how to stay calm during exams": "Pause, breathe deeply, and tackle questions one at a time. Stay focused, not rushed.",
    "how to remember formulas easily": "Use visualization, write them out repeatedly, and create funny mnemonics!",
    "how to prepare for multiple exams in a week": "Create a daily subject plan and balance between topics smartly!",
    "how to make a study timetable": "Divide your day into study blocks with specific goals. Want a sample template?",
    "what to do if i blank out during an exam": "Stay calm, skip the question, and return later. Your memory usually unlocks!",
    "how to overcome self-doubt before exams": "Focus on your preparation, repeat positive affirmations, and trust your growth!",
    "how to avoid last-minute cramming": "Start early, revise regularly, and stay consistent. Need a weekly study plan?",
    "how to plan a study break": "Short breaks after 50 minutes help your brain absorb better. Make them fun but short!",
    "how to increase exam writing speed": "Practice timed writing, use keywords, and avoid overthinking. Speed grows with practice!",
    "how to deal with tough exam questions": "Stay calm, attempt what you can, and don't get stuck. Move and return later if needed!",
    "how to avoid getting distracted by social media": "Use apps like Forest or simply turn off notifications. You control your time!",
    "how to handle negative thoughts before exams": "Challenge them with evidence: You've prepared and are capable. Positivity is powerful!",
    "how to maximize exam revision time": "Focus on past mistakes, key concepts, and practice papers. Work smart!",
    "how to stay disciplined during exams": "Set daily goals, track them, and reward yourself for completing tasks!",
    "how to revise subjects you dislike": "Mix them with subjects you like or turn them into mini-challenges!",
    "how to study effectively at night": "Keep lighting bright, take short movement breaks, and avoid heavy meals!",
    "how to avoid burnout while studying": "Balance intense sessions with relaxation, hydration, and sleep!",
    "how to manage group study sessions": "Set clear goals, stick to schedules, and help each other stay on track!",
    "how to reduce pressure from competitive exams": "Focus on your own journey, not comparisons. Progress over perfection!",
    "how to stay happy during exam time": "Celebrate small wins, take nature breaks, and stay connected with positive friends!",
    "how to deal with academic expectations": "Set your own goals, communicate clearly, and define success on your terms!",
    "how to finish studying syllabus on time": "Break it into daily chunks, track your progress, and adjust flexibly!",
    "how to be consistent with exam preparation": "Daily small steps create big results. Habit beats mood every time!",
    "how to prepare for oral exams": "Practice speaking aloud, record yourself, and simulate real scenarios!",
    "how to stay confident if exam preparation is incomplete": "Focus on strengths, cover key topics, and stay calm. Partial preparation still matters!",
    "how to deal with competition stress": "See competition as motivation, not pressure. Your only competition is your past self!",
    "how to manage sudden exam stress": "Pause, breathe, and reframe the situation. Stress shrinks when you act calmly!",
    "how to prepare mentally for exams": "Visualize success, affirm your abilities, and start with belief!",
    "how to study boring topics effectively": "Gamify it, teach it to a friend, or create silly memory tricks!",
    "how to relax mind after long study hours": "Listen to calming music, stretch, or do a short meditation!",
    "how to make exams feel less scary": "Preparation reduces fear. Every topic you study lowers the fear meter!",
    "how to manage academic competition pressure": "Everyone has a unique timeline. Focus on being better, not the best!",
    "how to avoid exam day panic": "Prepare your bag early, revise lightly, and trust your preparation!",
    "how to tackle difficult exam papers": "Start with easy questions, gain confidence, then attempt harder ones!",
    "how to feel proud after exams": "No matter the outcome, completing an exam is an achievement itself!",
    "how to stay healthy during exams": "Good sleep, balanced meals, and a little exercise keep you sharp!",
    "how to improve exam performance": "Consistent study, practice tests, and relaxation techniques make a huge difference!",
    "how to build exam writing stamina": "Practice full-length papers at home under timed conditions!",
    "how to overcome perfectionism in exams": "Aim for progress, not perfection. Done is better than perfect!",
    "how to find motivation when feeling drained": "Reconnect with your goals, reward small efforts, and take micro-breaks!",
    "how to avoid exam regrets": "Prepare steadily, stay calm, and give your best shot. Regret nothing if you tried your best!",

    # --- New: Substance Abuse ---
    "how to quit vaping": "Reduce gradually, chew gum as a substitute, and seek support from a counselor. You’ve got this! Need a step-by-step plan?",
    "is weed dangerous": "Yes, it can harm memory, motivation, and mental health, especially for teens. Want science-backed facts?",
    "i regret trying drugs": "It takes courage to admit this. Talk to a trusted adult or call [local helpline]. You’re not alone.",
    "what is substance abuse": "Substance abuse is using drugs, alcohol, or other substances in a way that harms your health or daily life.",
    "how to know if i have a substance problem": "If substances are affecting your health, emotions, studies, or relationships, it's worth reaching out for support.",
    "why do people start using drugs or alcohol": "Sometimes it's curiosity, peer pressure, stress, or wanting to escape. But there are always healthier ways to cope!",
    "can stress lead to substance abuse": "Yes, stress can push people toward unhealthy coping. Talking about it and finding better outlets can help.",
    "how to say no to drugs without feeling awkward": "A simple, firm 'no thanks' is enough. Real friends respect your choices.",
    "what are healthy ways to handle stress instead of substances": "Exercise, talking to someone you trust, creative hobbies, or mindfulness are great options!",
    "how does substance abuse affect mental health": "It can worsen anxiety, depression, and mood swings. Healing starts with seeking support.",
    "i feel pressure to try substances, what should i do": "Trust your gut. You don't owe anyone an explanation. Your health matters most!",
    "what are early signs of substance abuse": "Changes in behavior, secrecy, declining grades, mood swings, and loss of interest in activities.",
    "can substance abuse be treated": "Absolutely. With the right support, counseling, and care, recovery is 100% possible!",
    "how to help a friend struggling with substances": "Listen without judgment, encourage them to seek help, and remind them they are not alone.",
    "is occasional drug use harmless": "Even occasional use can lead to risks. It's better to stay safe and substance-free!",
    "how to recover from substance addiction": "Recovery starts with admitting the problem, getting professional help, and building a strong support system.",
    "what is detox": "Detox is safely removing substances from the body, usually with medical support. It's the first step to healing.",
    "how can alcohol affect academic performance": "It can lower concentration, memory, and motivation. Staying clear keeps you sharp and focused!",
    "what to do if i relapse after quitting": "Don't be hard on yourself. Relapse can be part of recovery. Reflect, reach out, and keep moving forward.",
    "how to build new habits after quitting substances": "Fill your schedule with positive activities that you enjoy and celebrate small wins!",
    "how to deal with cravings": "Cravings pass. Distract yourself, call a friend, exercise, or drink water to ride it out!",
    "why is peer pressure so powerful": "We all want to fit in. But real strength is choosing what’s right for you, not what’s popular.",
    "is vaping safer than smoking": "Vaping still carries serious health risks. Don't let marketing fool you!",
    "how to avoid environments that encourage substance use": "Spend time with positive people and places that uplift you, not pressure you.",
    "how to rebuild trust after substance issues": "Be patient, stay consistent, and show your commitment to change through actions, not just words.",
    "what is withdrawal": "Withdrawal is your body's reaction when you stop using a substance it got used to. Medical help makes it safer.",
    "how to stay strong when friends offer substances": "Remember your goals, your health, and your dreams. Saying no is saying yes to your future!",
    "what are long-term effects of drug abuse": "Memory loss, health issues, damaged relationships, and lost opportunities. Prevention is the best protection!",
    "can i still have fun without drinking or drugs": "Absolutely! Genuine fun doesn’t need substances. Adventures, friendships, creativity—those are real highs!",
    "how to find support for quitting substances": "Start with a counselor, trusted adult, or local support groups. You're not alone.",
    "what is substance dependency": "When your body or mind feels like it 'needs' a substance to function. But you can regain control!",
    "what to do if i feel judged for not using substances": "Stay proud. Your choice shows strength, maturity, and courage.",
    "how to replace substance habits with positive ones": "Try sports, art, music, volunteering, or anything that builds you up instead of breaking you down.",
    "can i recover on my own without help": "Recovery is hard alone. Getting support makes it easier, faster, and less lonely.",
    "how to deal with guilt after substance abuse": "Forgive yourself. Growth starts with self-compassion and looking forward, not back.",
    "how to talk to parents about substance problems": "Be honest, stay calm, and remember they care about you. It's a brave step!",
    "how can substances affect my future goals": "Substance use can delay or derail dreams. Staying clean keeps you on track!",
    "is it too late to quit if i've been using for a long time": "It's never too late to choose a better path. Healing starts the moment you decide!",
    "how to avoid relapse triggers": "Know your triggers, plan ahead, and build a strong, positive routine!",
    "what if i lose friends after quitting substances": "Real friends support your growth. You'll make new, healthier friendships too!",
    "why do substances seem fun at first but harmful later": "Substances trick your brain at first, but over time they hurt more than help.",
    "how to handle parties without using substances": "Have an exit plan, stick with sober friends, and focus on the music, dancing, and conversations!",
    "how to overcome fear of quitting substances": "Change feels scary, but freedom, health, and real happiness are on the other side!",
    "how to forgive myself for past mistakes with substances": "Mistakes don't define you. What you do next does. You're stronger than your past!",
    "can therapy really help with substance abuse": "Yes! Therapy gives you tools, support, and strength to build a new chapter.",
    "how to avoid boredom without substances": "Discover new hobbies, passions, and experiences. A substance-free life is richer than you think!",
    "is marijuana safe because it's legal": "Legal doesn’t always mean safe. Marijuana can still harm memory, focus, and motivation, especially in young brains.",
    "how to stay motivated to stay clean": "Keep reminding yourself of your 'why' — health, dreams, relationships. Motivation grows with purpose!",
    "what happens if i mix substances": "Mixing substances can be extremely dangerous and unpredictable. It's never worth the risk!",
    "how to be patient with the recovery process": "Healing takes time. Every step, even tiny ones, is progress worth celebrating!",
    "what are signs someone needs help with substances": "Isolation, mood changes, secretive behavior, slipping grades — early help makes a big difference!",
    "how to support a sibling struggling with substances": "Listen with love, encourage help, and remind them they are not alone in this fight.",
    "what's the difference between use and abuse": "Use becomes abuse when it causes harm to your health, relationships, studies, or safety.",
    "how to stay substance-free when everyone around me isn't": "Be the leader, not the follower. Inspire others with your strength!",
    "how does alcohol affect the brain": "It slows thinking, weakens memory, and damages decision-making over time. Protect your superpower—your brain!",
    "what are natural highs i can experience": "Exercise, laughter, music, nature, art, achieving goals—all these give real, healthy highs!",
    "how to celebrate victories without substances": "Dance, call a friend, enjoy a hobby, or reward yourself with something special!",
    "how does substance abuse affect families": "It creates hurt, stress, and distance. Healing yourself helps heal relationships too.",
    "can hobbies help prevent substance use": "Definitely! Staying busy with positive passions protects your mind and heart.",
    "how to build self-esteem without substances": "Achieve small goals, surround yourself with positivity, and treat yourself with kindness!",
    "what to say if someone pressures me to try drugs": "A confident 'No thanks, not my thing' is powerful enough. Stand tall!",
    "what are healthy coping mechanisms for stress": "Journaling, exercising, deep breathing, or talking it out are amazing ways to cope.",
    "is addiction a weakness": "No. Addiction is a health issue, not a character flaw. Seeking help is a sign of incredible strength!",
    "how to forgive a friend who hurt me because of substances": "Forgiveness helps you heal, even if trust needs time to rebuild. Choose peace for yourself.",
    "how to move forward after substance struggles": "Focus on growth, seek support, and believe in your ability to create a brighter future!",
    "can exercise help during recovery": "Yes! Exercise boosts mood, energy, and confidence. It's a powerful recovery tool!",
    "how to set goals during recovery": "Start small, make them meaningful, and celebrate every milestone along the way!",
    "how to avoid toxic friendships encouraging substance use": "Choose friends who lift you up, not drag you down. You deserve real support!",
    "how does substance use affect decision making": "It clouds judgment, increases risk-taking, and leads to regret. Clear minds make strong choices!",
    "what if i feel lonely after quitting substances": "New, positive connections will come. Healing sometimes means making space for better people.",
    "how to build a support system during recovery": "Find trusted friends, family, mentors, or support groups who encourage your growth!",
    "can meditation help with cravings": "Absolutely! Meditation trains your mind to ride through cravings calmly.",
    "how to recognize progress in recovery": "Even tiny steps forward—like saying no once—are huge victories!",
    "what if i feel ashamed of my past substance use": "Your past is just one chapter, not your whole story. Your future is being written right now!",
    "how to practice self-care during recovery": "Prioritize rest, positive activities, mental health, and rewarding yourself kindly!",
    "how to handle setbacks without giving up": "A setback isn’t failure. It's a setup for a stronger comeback. Stay in the fight!",
    "how to believe i can change": "Change is always possible. One step, one decision at a time. You’re capable of amazing things!",
    "what is a trigger in substance abuse recovery": "A trigger is something that tempts you to use. Knowing your triggers helps you beat them!",
    "how to rebuild confidence after addiction": "Confidence grows with every good choice you make. Every small victory matters!",
    "how to replace bad habits with good ones": "Focus on adding positive activities instead of just removing old ones. New habits bloom with time!",
    "is it normal to feel scared about quitting": "Yes, it's completely normal. Fear means you're stepping into growth. You’re not alone!",
    "how to stay hopeful during tough times": "Hope is a muscle—feed it with positive action, support, and self-kindness!",
    "how to live a substance-free happy life": "Fill your days with passion, connection, creativity, and self-love. True happiness doesn’t need substances!",

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
    
@app.route('/inner1')
def inner1():
    template_path = os.path.join(app.root_path, 'templates', 'emotional.html')
    if not os.path.exists(template_path):
        logger.error(f"Template file not found at: {template_path}")
        return "Error: Template 'emotional.html' not found.", 500
    try:
        return render_template('emotional.html')
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