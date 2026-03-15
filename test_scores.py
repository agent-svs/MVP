from core.nlu.intent import classify_intent, _load_model
import os
os.environ["INTENT_FINE_TUNED"] = "true"
os.environ["INTENT_MODEL"] = "models/intent_classifier"
_load_model()

print(classify_intent("Hey, I would to make an appointment."))
print(classify_intent("cardiology"))
