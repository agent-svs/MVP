import json
import random

doctors = ["Dr. Sharma", "Dr. Patel", "Dr. Reddy", "Dr. Gupta", "Dr. Ali", "Dr. Vivek", "Dr. Mehta", "Dr. Nair", "Dr. Singh", "Dr. Kumar", "Dr. Desai", "Dr. Fernandez", "Dr. Lee"]
services = ["cardiology", "dermatology", "pediatrics", "orthopedics", "neurology", "ENT", "dental", "general checkup", "ophthalmology", "psychiatry", "gastroenterology", "gynecology"]
patients = ["Amit", "Priya", "Rahul", "Neha", "Suresh", "Ramesh", "Anjali", "Vikram", "Sarah", "John", "Ahmed", "David"]
dates = ["tomorrow", "today", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "the 15th", "next week", "22nd", "this Friday"]
times = ["10am", "2pm", "3:30", "evening", "morning", "5pm", "11:15", "4 o clock", "12 noon", "1:30", "6pm"]
wait_contexts = ["How can I help you?", "Welcome to City Clinic.", "What can I do for you?", "How may I assist?", "City Clinic reception.", "Is there anything else?", "", "Good morning.", "Hello, how can I help?"]

def rand_phone():
    return f"9{random.randint(100000000, 999999999)}"

new_entries = []

def gen(templates, intent, context_list=wait_contexts, count=100):
    for i in range(count):
        template, expected_slots = random.choice(templates)
        slots = {}
        formatted_str = template
        if "{doctor_name}" in template:
            d = random.choice(doctors)
            formatted_str = formatted_str.replace("{doctor_name}", d)
            if "doctor_name" in expected_slots: slots["doctor_name"] = d
        if "{service_type}" in template:
            s = random.choice(services)
            formatted_str = formatted_str.replace("{service_type}", s)
            if "service_type" in expected_slots: slots["service_type"] = s
        if "{patient_name}" in template:
            p = random.choice(patients)
            formatted_str = formatted_str.replace("{patient_name}", p)
            if "patient_name" in expected_slots: slots["patient_name"] = p
        if "{date}" in template:
            d = random.choice(dates)
            formatted_str = formatted_str.replace("{date}", d)
            if "date" in expected_slots: slots["date"] = d
        if "{time}" in template:
            t = random.choice(times)
            formatted_str = formatted_str.replace("{time}", t)
            if "time" in expected_slots: slots["time"] = t
        if "{phone}" in template:
            ph = rand_phone()
            formatted_str = formatted_str.replace("{phone}", ph)
            if "phone" in expected_slots: slots["phone"] = ph
            
        new_entries.append({
            "context": random.choice(context_list),
            "utterance": formatted_str,
            "intent": intent,
            "slots": slots
        })

complaint_tmpls = [
    ("I want to register a very strong complaint against {doctor_name}.", ["doctor_name"]),
    ("{doctor_name} was extremely rude during my visit.", ["doctor_name"]),
    ("The {service_type} department has the worst service I have ever seen.", ["service_type"]),
    ("I had a terrible experience with {doctor_name} in {service_type}.", ["doctor_name", "service_type"]),
    ("Why is the {service_type} waiting area so dirty?", ["service_type"]),
    ("I need to file a formal complaint about {doctor_name}.", ["doctor_name"]),
    ("Your {service_type} doctor, {doctor_name}, misdiagnosed me.", ["doctor_name", "service_type"]),
    ("I have a huge complaint regarding my {service_type} consultation.", ["service_type"]),
    ("Who manages {doctor_name}? I need to complain right now.", ["doctor_name"]),
    ("The staff in the {service_type} wing ignored me completely.", ["service_type"]),
    ("This clinic's {service_type} facilities are pathetic.", ["service_type"]),
    ("I'm leaving a 1-star review for {doctor_name}.", ["doctor_name"]),
    ("Can I speak to the manager about {doctor_name}'s behavior?", ["doctor_name"]),
    ("Refund my money for the {service_type} visit, it was useless.", ["service_type"]),
    ("{doctor_name} charged me for {service_type} but didn't even check me properly.", ["doctor_name", "service_type"])
]
gen(complaint_tmpls, "complaint", count=120)

transfer_tmpls = [
    ("Put a real person on the line please.", []),
    ("I need to speak to a human operator.", []),
    ("Stop the bot, transfer me to the receptionist.", []),
    ("Can I just talk to a human?", []),
    ("Transfer me to a live agent.", []),
    ("Hand over this chat to a real staff member.", []),
    ("I have a complex issue, get me a human.", []),
    ("Operator. Now.", []),
    ("I don't want to talk to an AI.", []),
    ("Is there a human I can speak with?", []),
    ("Route my call to a live person.", []),
    ("Bypass this menu and give me a real human.", []),
    ("Connect me to the front desk lady.", []),
    ("I hate talking to machines.", []),
    ("Real person please.", [])
]
gen(transfer_tmpls, "transfer_to_human", count=120)

emergency_tmpls = [
    ("Help! It's an emergency, I need the {service_type} doctor right away!", ["service_type"]),
    ("My father is collapsing, is your {service_type} ward open?", ["service_type"]),
    ("This is a medical emergency for {service_type}, prepare the ER!", ["service_type"]),
    ("Forget the booking, I need {service_type} emergency care now.", ["service_type"]),
    ("My baby is choking, is the {service_type} specialist there?", ["service_type"]),
    ("He has severe chest pain, we need {service_type} urgently.", ["service_type"]),
    ("I am bleeding heavily, is anyone in {service_type} available?", ["service_type"]),
    ("Call an ambulance for a {service_type} emergency!", ["service_type"]),
    ("My wife's water broke, emergency {service_type} admission needed.", ["service_type"]),
    ("I can't breathe, severe asthma! Get {service_type} doctor.", ["service_type"]),
    ("Send help fast, {service_type} emergency at the gate.", ["service_type"]),
    ("Is the ER equipped for {service_type} trauma?", ["service_type"]),
    ("This is not a regular checkup, I need immediate {service_type} assistance.", ["service_type"])
]
gen(emergency_tmpls, "emergency_request", count=120)

update_tmpls = [
    ("Update my phone number to {phone}.", ["phone"]),
    ("My new mobile is {phone}.", ["phone"]),
    ("Change my contact digit to {phone}.", ["phone"]),
    ("I lost my old sim, replace it with {phone}.", ["phone"]),
    ("My name is {patient_name} and my new number is {phone}.", ["patient_name", "phone"]),
    ("Update the file for {patient_name} with number {phone}.", ["patient_name", "phone"]),
    ("Please note my updated WhatsApp number: {phone}.", ["phone"]),
    ("Remove the old number and use {phone} for {patient_name}.", ["patient_name", "phone"]),
    ("I shifted, my contact is now {phone}.", ["phone"]),
    ("Change {patient_name}'s phone to {phone}.", ["patient_name", "phone"])
]
gen(update_tmpls, "update_contact_info", count=110)

verify_tmpls = [
    ("Yes, my name is {patient_name}.", ["patient_name"]),
    ("I am {patient_name}, verify me.", ["patient_name"]),
    ("You can check under the number {phone}.", ["phone"]),
    ("My number is {phone}.", ["phone"]),
    ("I am {patient_name} and my registered phone is {phone}.", ["patient_name", "phone"]),
    ("It's me, {patient_name}.", ["patient_name"]),
    ("Check the file for {patient_name}.", ["patient_name"]),
    ("The mobile connected is {phone}.", ["phone"]),
    ("Authenticate using {phone}.", ["phone"]),
    ("This is {patient_name} speaking from {phone}.", ["patient_name", "phone"])
]
gen(verify_tmpls, "verify_patient_identity", count=110)

register_tmpls = [
    ("I want to register a new file for {patient_name} with {phone}.", ["patient_name", "phone"]),
    ("Sign up {patient_name} as a new patient, number is {phone}.", ["patient_name", "phone"]),
    ("Create a totally new account for {patient_name}, phone {phone} for {service_type}.", ["patient_name", "phone", "service_type"]),
    ("I am a new patient, my name is {patient_name} and I need {service_type}.", ["patient_name", "service_type"]),
    ("Register me, {patient_name}. Mobile is {phone}.", ["patient_name", "phone"]),
    ("Enroll {patient_name} in your system. Contact: {phone}.", ["patient_name", "phone"]),
    ("I want to open a new file for {service_type}, name is {patient_name}.", ["service_type", "patient_name"]),
    ("First time here. Name: {patient_name}, Number: {phone}.", ["patient_name", "phone"]),
    ("Please add {patient_name} as a new patient for {service_type}.", ["patient_name", "service_type"]),
    ("My son {patient_name} needs to be registered. Number is {phone}.", ["patient_name", "phone"])
]
gen(register_tmpls, "register_new_patient", count=110)

check_my_tmpls = [
    ("When is my next {service_type} appointment?", ["service_type"]),
    ("What appointments do I have with {doctor_name}?", ["doctor_name"]),
    ("Can you list all bookings for {patient_name} on {date}?", ["patient_name", "date"]),
    ("Check my schedule for {date}.", ["date"]),
    ("Tell me my upcoming visits for {service_type}.", ["service_type"]),
    ("Read out my pending appointments under {patient_name}.", ["patient_name"]),
    ("Do I see {doctor_name} on {date}?", ["doctor_name", "date"]),
    ("What are my scheduled dates for {service_type}?", ["service_type"]),
    ("When exactly am I seeing {doctor_name} again?", ["doctor_name"]),
    ("Give me the itinerary for {patient_name} for {service_type}.", ["patient_name", "service_type"])
]
gen(check_my_tmpls, "check_my_appointments", count=100)

chk_status_tmpls = [
    ("Is my {service_type} appointment confirmed for {date}?", ["service_type", "date"]),
    ("Check the status of my booking with {doctor_name}.", ["doctor_name"]),
    ("Did my {time} slot for {service_type} get cancelled?", ["time", "service_type"]),
    ("What is the status of {patient_name}'s appointment on {date}?", ["patient_name", "date"]),
    ("Is {doctor_name} running late for my {time} slot?", ["doctor_name", "time"]),
    ("I need the current status of my {service_type} visit.", ["service_type"]),
    ("Check if my {date} booking with {doctor_name} is active.", ["date", "doctor_name"]),
    ("Is the status for {patient_name} showing confirmed for {service_type}?", ["patient_name", "service_type"]),
    ("Did the doctor approve my {time} slot for {service_type}?", ["time", "service_type"]),
    ("I need to know the status of {patient_name}'s {time} visit.", ["patient_name", "time"])
]
gen(chk_status_tmpls, "check_appointment_status", count=100)

cancel_tmpls = [
    ("Cancel my {service_type} appointment.", ["service_type"]),
    ("I want to cancel my visit with {doctor_name}.", ["doctor_name"]),
    ("Please cancel my {service_type} booking on {date}.", ["service_type", "date"]),
    ("Cancel {patient_name}'s appointment with {doctor_name}.", ["patient_name", "doctor_name"]),
    ("Remove my {time} {service_type} appointment.", ["time", "service_type"]),
    ("Abort my checkup with {doctor_name} at {time}.", ["doctor_name", "time"]),
    ("I can't make it to my {service_type} visit with {doctor_name}.", ["service_type", "doctor_name"]),
    ("Cancel the {service_type} consultation scheduled for {date}.", ["service_type", "date"]),
    ("Drop my booking for {doctor_name}.", ["doctor_name"]),
    ("I want to revoke {patient_name}'s {service_type} appointment.", ["patient_name", "service_type"])
]
gen(cancel_tmpls, "cancel_appointment", count=70)

confirm_tmpls = [
    ("Yes, confirm my {time} appointment.", ["time"]),
    ("Confirm my booking with {doctor_name} for {service_type}.", ["doctor_name", "service_type"]),
    ("Yes, {time} with {doctor_name} is perfect.", ["time", "doctor_name"]),
    ("I am confirming {patient_name}'s {service_type} appointment at {time}.", ["patient_name", "service_type", "time"]),
    ("Lock in my {time} slot for {service_type}.", ["time", "service_type"]),
    ("Yes, confirming the {service_type} visit with {doctor_name}.", ["service_type", "doctor_name"]),
    ("Please confirm the {time} slot with {doctor_name}.", ["time", "doctor_name"]),
    ("Yes, that {service_type} appointment at {time} works for me.", ["service_type", "time"]),
    ("Confirm {patient_name} for {time} with {doctor_name}.", ["patient_name", "time", "doctor_name"]),
    ("Absolutely, mark my {time} {service_type} slot as confirmed.", ["time", "service_type"])
]
gen(confirm_tmpls, "confirm_appointment", count=70)

ambig_confirm_tmpls = [
    ("Okay", []), ("Right", []), ("Sure", []), ("Yes", []), ("Exactly", []), 
    ("Make it so", []), ("Go ahead", []), ("Sounds good", []), ("Perfect", []), ("Done", [])
]
ambig_confirm_contexts = ["Your appointment is for 4pm. Do you confirm?", "Shall I lock this booking?", "I am confirming it now, okay?"]
gen(ambig_confirm_tmpls, "confirmation", context_list=ambig_confirm_contexts, count=25)

ambig_goodbye_tmpls = [
    ("Okay", []), ("Right", []), ("Done", []), ("That's it", []), ("Thanks", [])
]
ambig_goodbye_contexts = ["Your booking is confirmed. Anything else?", "I have cancelled it. Let me know if you need more help.", "Your file is updated."]
gen(ambig_goodbye_tmpls, "goodbye", context_list=ambig_goodbye_contexts, count=25)

negation_tmpls = [
    ("I do NOT want to cancel it.", []),
    ("Don't cancel it.", []),
    ("Wait, do not reschedule.", []),
    ("I am not cancelling.", []),
    ("No, don't drop the appointment.", []),
    ("Stop, I didn't say cancel.", []),
    ("No no, keep the appointment active.", []),
    ("Do not change anything.", []),
    ("I absolutely do NOT want it cancelled.", []),
    ("Wait don't delete my booking.", [])
]
negation_contexts = ["You want to cancel this booking, right?", "I am cancelling your appointment now, okay?", "Should I remove your slot?"]
gen(negation_tmpls, "denial", context_list=negation_contexts, count=40)

asr_entries = [
    {"context": "What time?", "utterance": "Book for tree pm", "intent": "slot_answer", "slots": {"time": "3pm"}},
    {"context": "Which doctor?", "utterance": "I want to see Dr. Pedal", "intent": "slot_answer", "slots": {"doctor_name": "Dr. Patel"}},
    {"context": "What is the service?", "utterance": "Cancel my den tell appointment", "intent": "cancel_appointment", "slots": {"service_type": "dental"}},
    {"context": "Which department?", "utterance": "I need the cart biology doctor", "intent": "book_appointment", "slots": {"service_type": "cardiology"}},
    {"context": "What date?", "utterance": "No, confirm my true stay booking", "intent": "confirm_appointment", "slots": {"date": "Tuesday"}},
    {"context": "What is the patient name?", "utterance": "Yes, my name is Ah meat", "intent": "slot_answer", "slots": {"patient_name": "Amit"}},
    {"context": "What date?", "utterance": "Change the date to the ate", "intent": "change_date", "slots": {"date": "the 8th"}},
    {"context": "What service?", "utterance": "I need to see an eye dog door", "intent": "book_appointment", "slots": {"service_type": "ophthalmology"}},
    {"context": "What is your phone?", "utterance": "nine ate seven six five for tree too one oh", "intent": "slot_answer", "slots": {"phone": "9876543210"}},
    {"context": "What is your phone?", "utterance": "six ate for too won won 0", "intent": "slot_answer", "slots": {"phone": "6842110"}}
]
for _ in range(5):
    for entry in asr_entries:
        new_entry = dict(entry)
        new_entry["slots"] = dict(entry["slots"])
        new_entries.append(new_entry)

book_phone_tmpls = [
    ("Book an appointment for {phone}.", ["phone"]),
    ("I want to schedule a visit, contact is {phone}.", ["phone"]),
    ("Please book a slot for {phone} on {date}.", ["phone", "date"]),
    ("Book {service_type} for number {phone}.", ["service_type", "phone"]),
    ("Can you schedule {doctor_name} for the patient with number {phone}?", ["doctor_name", "phone"]),
    ("I need an appointment, call me at {phone} when confirmed.", ["phone"]),
    ("Book for {patient_name}, phone {phone} tomorrow.", ["patient_name", "phone"]),
    ("Schedule {service_type} for {patient_name} at {phone}.", ["service_type", "patient_name", "phone"])
]
gen(book_phone_tmpls, "book_appointment", count=60)

with open("dataset.jsonl", "a") as f:
    for entry in new_entries:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Successfully appended {len(new_entries)} augmented entries to dataset.jsonl")
